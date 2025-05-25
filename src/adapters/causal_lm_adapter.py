import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from typing import Dict, Any
import asyncio
from .base_adapter import ModelServiceAdapter

logger = logging.getLogger(__name__)

class CausalLMAdapter(ModelServiceAdapter):
    """Adapter for causal language models (e.g., CodeLlama)."""
    async def load(self, timeout: float = 1200.0) -> bool:
        """Load causal LM with device verification."""
        success = await self.load_model(AutoModelForCausalLM, AutoTokenizer, timeout)
        if success:
            device_map = self.createDeviceMap()
            logger.info("Verifying model parameter devices...")
            for name, param in self.model.named_parameters():
                if param.device.type == 'cpu':
                    logger.warning(f"Parameter {name} is on CPU, moving to device_map")
                    device = device_map.get(name, 'cuda:0')
                    param.data = param.data.to(device)
            for i in range(len(self.model.model.layers)):
                rotary_emb = self.model.model.layers[i].self_attn.rotary_emb
                try:
                    module_device = next(rotary_emb.parameters()).device if list(rotary_emb.parameters()) else 'unknown'
                    logger.info(f"Layer {i} rotary_emb module device: {module_device}")
                    if i <= 5:
                        rotary_emb.to('cuda:1')
                        logger.info(f"Moved layer {i} rotary_emb to cuda:1")
                    else:
                        rotary_emb.to('cuda:0')
                        logger.info(f"Moved layer {i} rotary_emb to cuda:0")
                except (AttributeError, StopIteration):
                    logger.warning(f"Layer {i} rotary_emb has no parameters or device info, skipping")
                if hasattr(rotary_emb, 'inv_freq'):
                    logger.info(f"Layer {i} rotary_emb.inv_freq device: {rotary_emb.inv_freq.device}")
                    if rotary_emb.inv_freq.device.type == 'cpu':
                        logger.warning(f"Layer {i} rotary_emb.inv_freq on CPU, moving to cuda:1")
                        rotary_emb.inv_freq = rotary_emb.inv_freq.to('cuda:1')
            logger.info("Model parameter and module device verification complete")
            if self.tokenizer:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        return success

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle text generation request with Llama-specific patching."""
        try:
            request = await self.preprocess_input(request)
            params = request.get('params', {})
            prompt = request.get('input', '')
            if not prompt:
                raise ValueError("Prompt input required")

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA not available")

            inputs = self.tokenizer(prompt, return_tensors="pt")
            logger.info(f"Initial input tensor devices: {[k + ': ' + str(v.device) for k, v in inputs.items()]}")

            if not hasattr(self.model, 'get_submodule'):
                raise ValueError("Model does not support get_submodule")
            embed_device = self.model.get_submodule("model.embed_tokens").weight.device
            logger.info(f"Embedding layer device: {embed_device}")

            inputs = {k: v.to(embed_device) for k, v in inputs.items()}
            logger.info(f"Input tensor devices after move: {[k + ': ' + str(v.device) for k, v in inputs.items()]}")

            if any(tensor.device.type == 'cpu' for tensor in inputs.values()):
                raise ValueError(f"Input tensors on CPU after attempted move to {embed_device}")

            seq_length = inputs['input_ids'].size(1)
            logger.info(f"Input sequence length: {seq_length}")

            self.model.eval()
            for device in self.createDeviceMap().values():
                torch.cuda.synchronize(device)

            def patched_rotary_emb_forward(self, hidden_states, position_ids):
                seq_len = hidden_states.size(1)
                pos_id = position_ids[0, -1].item()
                should_log = seq_len > 1 or pos_id == seq_length
                inv_freq = self.inv_freq.to(hidden_states.device)
                position_ids = position_ids.float()
                inv_freq_expanded = inv_freq[None, :, None].float().expand(position_ids.size(0), -1, 1)
                position_ids_expanded = position_ids[:, None, :]
                freqs = (inv_freq_expanded @ position_ids_expanded).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos()
                sin = emb.sin()
                if should_log:
                    logger.info(f"Rotary embedding: position_ids shape: {position_ids.shape}, values: {position_ids.tolist()}, cos shape: {cos.shape}")
                return cos.to(hidden_states.dtype), sin.to(hidden_states.dtype)
            
            setattr(LlamaRotaryEmbedding, 'forward', patched_rotary_emb_forward)
            logger.info(f"Patched LlamaRotaryEmbedding class forward method: {LlamaRotaryEmbedding.forward.__qualname__}")

            logger.info(f"All input tensors to generate: {[k + ': ' + str(v.device) for k, v in inputs.items()]}")
            for i in range(len(self.model.model.layers)):
                rotary_emb = self.model.model.layers[i].self_attn.rotary_emb
                try:
                    module_device = next(rotary_emb.parameters()).device if list(rotary_emb.parameters()) else 'unknown'
                    logger.info(f"Layer {i} rotary_emb module device before generate: {module_device}")
                except (AttributeError, StopIteration):
                    logger.info(f"Layer {i} rotary_emb has no parameters, device unknown")
                if hasattr(rotary_emb, 'inv_freq'):
                    logger.info(f"Layer {i} rotary_emb.inv_freq device before generate: {rotary_emb.inv_freq.device}")
                logger.info(f"Layer {i} rotary_emb forward method: {rotary_emb.forward.__qualname__}")
                try:
                    test_hidden_states = torch.zeros(1, seq_length, self.model.config.hidden_size, device=embed_device)
                    test_position_ids = torch.arange(seq_length, dtype=torch.long, device=embed_device).unsqueeze(0)
                    cos, sin = rotary_emb(test_hidden_states, test_position_ids)
                    logger.info(f"Layer {i} rotary_emb forward method test successful, cos device: {cos.device}, sin device: {sin.device}, cos shape: {cos.shape}")
                except Exception as e:
                    logger.warning(f"Layer {i} rotary_emb forward method test failed: {str(e)}")

            logger.info("Starting model generation...")
            outputs = self.model.generate(
                **inputs,
                max_length=params.get('max_length', 1000),
                temperature=params.get('temperature', 0.7),
                top_p=params.get('top_p', 0.9),
                do_sample=True,
                use_cache=True,
                output_scores=False,
                return_dict_in_generate=False
            )
            logger.info("Generation complete")
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return await self.postprocess_output({'text': response}, output_type='text/plain')
        except Exception as e:
            return self._handle_error(e)