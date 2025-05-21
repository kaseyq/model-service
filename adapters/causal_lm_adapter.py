import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Any
import os
import gc
from .base_adapter import ModelServiceAdapter

logger = logging.getLogger(__name__)

class CausalLMAdapter(ModelServiceAdapter):
    """Adapter for causal language models (e.g., CodeLlama)."""
    def load(self, timeout: float = 300.0) -> bool:
        from time import time
        start_time = time()
        try:
            logger.info(f"Loading causal LM: {self.config['model_config_id']} ({self.config['model_name']})")
            local_exists = os.path.exists(self.config['local_path'])
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config['local_path'] if local_exists else self.config['model_name'],
                load_in_4bit=self.config.get('load_in_4bit', False),
                device_map={'': 0},
                local_files_only=local_exists
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['local_path'] if local_exists else self.config['model_name'],
                local_files_only=local_exists
            )
            if not local_exists:
                self.model.save_pretrained(self.config['local_path'])
                self.tokenizer.save_pretrained(self.config['local_path'])
            logger.info(f"Model loaded in {time() - start_time:.2f} seconds")
            return True
        except Exception as e:
            if time() - start_time > timeout:
                logger.error(f"Model loading timed out after {timeout} seconds")
            else:
                logger.error(f"Failed to load model: {str(e)}")
            self.model = None
            self.tokenizer = None
            return False

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Causal LM unloaded and VRAM cleared")

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        params = request.get('params', {})
        prompt = request.get('input', '')
        if not prompt:
            raise ValueError("Prompt input required")
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda:0')
        outputs = self.model.generate(
            **inputs,
            max_length=params.get('max_length', 1000),
            temperature=params.get('temperature', 0.7),
            top_p=params.get('top_p', 0.9),
            do_sample=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {'status': 'success', 'response': response}