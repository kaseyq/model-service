import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, Any
from .base_adapter import ModelServiceAdapter
import time
import subprocess
import os

logger = logging.getLogger(__name__)

class CausalLMAdapter(ModelServiceAdapter):
    """Adapter for causal language models (e.g., CodeLlama)."""
    def load(self, timeout: float = 1200.0) -> bool:
        start_time = time.time()
        try:
            logger.info(f"Loading causal LM: {self.config['model_config_id']} ({self.config['model_name']})")
            logger.info(f"Config: {self.config}")
            logger.info(f"load_in_4bit: {self.config.get('load_in_4bit', False)}")
            logger.info(f"Loading causal LM: {self.config['model_config_id']} ({self.config['model_name']})")
            device_index = int(self.config['device'].split(':')[-1]) if 'cuda' in self.config['device'] else 0
            _, vram_used, vram_available = self._check_vram(device_index)
            
            # Check minimum VRAM requirement
            if vram_available < self.config.get('min_vram_needed', 10.0):
                logger.error(f"Insufficient VRAM: {vram_available:.2f} GB available, {self.config.get('min_vram_needed', 10.0)} GB needed")
                return False

            # Configure quantization if enabled
            quantization_config = None
            if self.config.get('load_in_4bit', False):
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,  # Use float16 for computation
                    bnb_4bit_quant_type="nf4",  # Normal float 4-bit quantization
                    bnb_4bit_use_double_quant=True  # Double quantization for better precision
                )

            local_exists = os.path.exists(self.config['local_path'])
            #logger.info("Loading model directly...")
            #self.model = AutoModelForCausalLM.from_pretrained(
            #    self.config['local_path'] if local_exists else self.config['model_name'],
            #    quantization_config=quantization_config,  # Pass quantization config
            #    device_map="auto",  # Let accelerate handle device placement
            #    torch_dtype=torch.float16 if not quantization_config else None,  # Use float16 for non-quantized models
            #    local_files_only=local_exists
            #)

            #tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            #tokenizer.save_pretrained(DUMP_PATH + MODEL_NAME)
            #model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, force_download=True)
            #model.save_pretrained(DUMP_PATH + MODEL_NAME)
            
            if local_exists != True:
                logger.info("Loading model...")
                self.model = AutoModelForCausalLM.from_pretrained(self.config['local_path'])
                #    self.config['model_name'])
                #    quantization_config=quantization_config,  # Pass quantization config
                #)
                #    device_map="auto",  # Let accelerate handle device placement
                
                #    torch_dtype=torch.float16 if not quantization_config else None,  # Use float16 for non-quantized models
                #    local_files_only=local_exists


                logger.info("Model loaded, loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config['local_path'],
                    #self.config['model_name'],
                    local_files_only=False)

                logger.info("Tokenizer loaded")
                self._save_model_locally()
            else:
                logger.info("Loading model...") 
                self.model = AutoModelForCausalLM.from_pretrained(self.config['local_path'], local_files_only=True)
                logger.info("Model loaded, loading tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config['local_path'],
                    local_files_only=True)
                logger.info("Tokenizer loaded")


            self.model = self.model.eval().to(device=self.config['device'])


            torch.cuda.synchronize(device_index)
            vram_used = torch.cuda.memory_allocated(device_index) / 1e9
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds, VRAM used: {vram_used:.2f} GB")
            logger.info(f"CUDA memory summary after loading:\n{torch.cuda.memory_summary(device_index)}")
            try:
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
            inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda:0')
            outputs = self.model.generate(
                **inputs,
                max_length=params.get('max_length', 1000),
                temperature=params.get('temperature', 0.7),
                top_p=params.get('top_p', 0.9),
                do_sample=True
            )
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