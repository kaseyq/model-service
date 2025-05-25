import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
from typing import Dict, Any
from .base_adapter import ModelServiceAdapter
import time
import subprocess
import os
from src.common.config_utils import load_config
import types

#from src.model_manager import ModelManager

logger = logging.getLogger(__name__)

class CausalLMAdapter(ModelServiceAdapter):
    """Adapter for causal language models (e.g., CodeLlama)."""

    def _check_vram(self, device_index):
        try:
            torch.cuda.set_device(device_index)
            vram_total = torch.cuda.get_device_properties(device_index).total_memory
            vram_used = torch.cuda.memory_allocated(device_index)
            vram_available = vram_total - vram_used
            return vram_total / 1e9, vram_used / 1e9, vram_available / 1e9  # Convert to GB
        except Exception as e:
            logger.error(f"Error checking VRAM for device {device_index}: {e}")
            return 0, 0, 0

    def check_vram_requirements(self) -> Dict[str, Any]:
        globalConfig = load_config()
        max_memory_config = self.config.get('max_memory', {})
        min_vram_needed_config = globalConfig.get('min_vram_needed', {})

        for device in max_memory_config:
            device_id = int(device.split(":")[1]) if ":" in device else device
            _, vram_used, vram_available = self._check_vram(device_id)
            min_vram_needed = globalConfig.get(f"cuda:{device_id}", 0) / 1e9  # Convert to GB
            if vram_available < min_vram_needed:
                logger.error(f"Insufficient VRAM on device cuda:{device_id}: {vram_available:.2f} GB available, {min_vram_needed:.2f} GB needed")
                return False
        return True

    def createMaxMemory(self) -> Dict[str, Any]:
        max_memory_config = self.config.get('max_memory', {})
        max_memory = {}

        for device, memory in max_memory_config.items():
            device_id = int(device.split(":")[1]) if ":" in device else device
            if isinstance(memory, str):
                if memory.endswith("GB"):
                    memory = int(float(memory[:-2]) * 1e9)  # Convert GB to bytes
                elif memory.endswith("MB"):
                    memory = int(float(memory[:-2]) * 1e6)  # Convert MB to bytes
            max_memory[device_id] = memory

        return max_memory

    def createDeviceMap(self) -> Dict[str, Any]:
        device_map_config = self.config['device_map']
        # Build device_map
        device_map = {}
        # Top-level components
        for component in ["model.embed_tokens", "model.norm", "lm_head"]:
            if component in device_map_config:
                device = device_map_config[component]
                device_id = int(device.split(":")[1]) if ":" in device else device
                device_map[component] = device_id

        # Layers
        layers_config = device_map_config.get("layers", {})
        for layer_range, device in layers_config.items():
            start, end = map(int, layer_range.split("-"))
            device_id = int(device.split(":")[1]) if ":" in device else device
            for i in range(start, end + 1):
                device_map[f"model.layers.{i}"] = device_id
                device_map[f"model.layers.{i}.input_layernorm"] = device_id
                device_map[f"model.layers.{i}.post_attention_layernorm"] = device_id
                device_map[f"model.layers.{i}.self_attn"] = device_id
                device_map[f"model.layers.{i}.mlp"] = device_id
        return device_map

    def load(self, timeout: float = 1200.0) -> bool:
        start_time = time.time()
        try:
            logger.info(f"Loading causal LM: {self.config['model_config_id']} ({self.config['model_name']})")
            logger.info(f"Config: {self.config}")
            
            if not self.check_vram_requirements():
                return False

            local_exists = os.path.exists(self.config['local_path'])

            # Build device_map from config
            device_map = self.createDeviceMap()
            max_memory = self.createMaxMemory()

            if local_exists :
                logger.info("Loading model...")
                # Load model in FP16
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.config['local_path'],
                    torch_dtype=torch.float16,
                    device_map=device_map,
                    max_memory=max_memory,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    local_files_only=True
                )
                #self.model = AutoModelForCausalLM.from_pretrained(
                #    self.config['local_path'],
                #    quantization_config=quantization_config)

                logger.info("Model loaded, loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config['local_path'],
                    trust_remote_code=True,
                    local_files_only=True)

                # Set pad token to avoid warning
                self.tokenizer.pad_token = self.tokenizer.eos_token

                logger.info("Tokenizer loaded")

                # Ensure all model parameters and modules are on the correct devices
                logger.info("Verifying model parameter devices...")
                for name, param in self.model.named_parameters():
                    if param.device.type == 'cpu':
                        logger.warning(f"Parameter {name} is on CPU, moving to device_map")
                        device = device_map.get(name, 'cuda:0')  # Default to cuda:0 if not in device_map
                        param.data = param.data.to(device)
                # Move rotary embedding modules
                for i in range(len(self.model.model.layers)):
                    rotary_emb = self.model.model.layers[i].self_attn.rotary_emb
                    try:
                        # Get module device (may raise AttributeError if not a standard module)
                        module_device = next(rotary_emb.parameters()).device if list(rotary_emb.parameters()) else 'unknown'
                        logger.info(f"Layer {i} rotary_emb module device: {module_device}")
                        if i <= 5:  # Layers 0-5 are on cuda:1
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
            else:
                logger.info("No local model exists") 
                return False
                #self.model = AutoModelForCausalLM.from_pretrained(self.config['local_path'], local_files_only=True)
                ##logger.info("Model loaded, loading tokenizer...")
                #self.tokenizer = AutoTokenizer.from_pretrained(
                #    self.config['model_name'],
                #    quantization_config=quantization_config,
                #    local_files_only=True)
                #logger.info("Tokenizer loaded")
                #self._save_model_locally()

            #torch.cuda.synchronize(device_index)
            #vram_used = torch.cuda.memory_allocated(device_index) / 1e9
            
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
            
            #logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds, VRAM used: {vram_used:.2f} GB")
            #logger.info(f"CUDA memory summary after loading:\n{torch.cuda.memory_summary(device_index)}")
            
            try:
                device_index = 0  # Define device_index for GPU utilization check
                logger.info(f"GPU utilization after loading: {torch.cuda.utilization(device_index)}%")
            except Exception as e:
                logger.warning(f"Failed to retrieve GPU utilization after loading: {str(e)}")
            return True
        except ImportError as e:
            logger.error(f"Import error during model loading (possibly missing bitsandbytes): {str(e)}", exc_info=True)
            self.model = None
            self.tokenizer = None
            return False
        except subprocess.TimeoutExpired:
            logger.error(f"Model loading timed out after {timeout} seconds")
            self.model = None
            self.tokenizer = None
            return False
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            self.model = None
            self.tokenizer = None
            return False

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            params = request.get('params', {})
            prompt = request.get('input', '')
            if not prompt:
                raise ValueError("Prompt input required")
            
            # Check CUDA availability
            if not torch.cuda.is_available():
                logger.error("CUDA not available")
                raise RuntimeError("CUDA not available")

            # Tokenize input without specifying device
            inputs = self.tokenizer(prompt, return_tensors="pt")
            logger.info(f"Initial input tensor devices: {[k + ': ' + str(v.device) for k, v in inputs.items()]}")

            # Get the device of the model's embedding layer
            if not hasattr(self.model, 'get_submodule'):
                logger.error("Model does not support get_submodule, cannot determine embedding device")
                raise ValueError("Model does not support get_submodule")
            embed_device = self.model.get_submodule("model.embed_tokens").weight.device
            logger.info(f"Embedding layer device: {embed_device}")

            # Move inputs to the embedding device
            inputs = {k: v.to(embed_device) for k, v in inputs.items()}
            logger.info(f"Input tensor devices after move: {[k + ': ' + str(v.device) for k, v in inputs.items()]}")

            # Verify no CPU tensors
            if any(tensor.device.type == 'cpu' for tensor in inputs.values()):
                logger.error(f"Input tensors still on CPU after move to {embed_device}")
                raise ValueError(f"Input tensors on CPU after attempted move to {embed_device}")

            # Let model.generate handle position_ids
            seq_length = inputs['input_ids'].size(1)
            logger.info(f"Input sequence length: {seq_length}")

            # Ensure model is in eval mode and synchronize GPUs
            self.model.eval()
            for device in self.createDeviceMap().values():
                torch.cuda.synchronize(device)

            # Patch LlamaRotaryEmbedding class forward method with minimal logging
            def patched_rotary_emb_forward(self, hidden_states, position_ids):
                seq_len = hidden_states.size(1)
                pos_id = position_ids[0, -1].item()
                # Log only for initial sequence or first generated token
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
            
            # Apply class-level patch
            setattr(LlamaRotaryEmbedding, 'forward', patched_rotary_emb_forward)
            logger.info(f"Patched LlamaRotaryEmbedding class forward method: {LlamaRotaryEmbedding.forward.__qualname__}")

            # Log all input tensors and rotary embedding devices for debugging
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
                # Verify forward method
                logger.info(f"Layer {i} rotary_emb forward method: {rotary_emb.forward.__qualname__}")
                # Test patch execution
                try:
                    test_hidden_states = torch.zeros(1, seq_length, self.model.config.hidden_size, device=embed_device)
                    test_position_ids = torch.arange(seq_length, dtype=torch.long, device=embed_device).unsqueeze(0)
                    cos, sin = rotary_emb(test_hidden_states, test_position_ids)
                    logger.info(f"Layer {i} rotary_emb forward method test successful, cos device: {cos.device}, sin device: {sin.device}, cos shape: {cos.shape}")
                except Exception as e:
                    logger.warning(f"Layer {i} rotary_emb forward method test failed: {str(e)}")

            # Generate output
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
            return {
                'status': 'okay',
                'message': 'Prompt processed successfully',
                'result': {
                    'text': response
                }
            }
        except Exception as e:
            logger.error(f"Error processing prompt: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f"Prompt processing failed: {str(e)}",
                'result': {}
            }

    def _save_model_locally(self):
         if not os.path.exists(self.config['local_path']):
             os.makedirs(self.config['local_path'])
         # Create a clean quantization_config dictionary
         if hasattr(self.model, 'config') and hasattr(self.model.config, 'quantization_config'):
             quant_config = self.model.config.quantization_config
             clean_quant_config = {
                 'load_in_4bit': getattr(quant_config, 'load_in_4bit', False),
                 'bnb_4bit_compute_dtype': str(getattr(quant_config, 'bnb_4bit_compute_dtype', 'float16')),
                 'bnb_4bit_quant_type': getattr(quant_config, 'bnb_4bit_quant_type', 'nf4'),
                 'bnb_4bit_use_double_quant': getattr(quant_config, 'bnb_4bit_use_double_quant', False)
             }
             self.model.config.quantization_config = clean_quant_config
         logger.info(f"Saving model to local path: {self.config['local_path']}")
         self.model.save_pretrained(self.config['local_path'])
         self.tokenizer.save_pretrained(self.config['local_path'])
         logger.info("Model and tokenizer saved")