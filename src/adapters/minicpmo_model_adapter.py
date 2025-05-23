import logging
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any
import os
import tempfile
import uuid
import base64
from .base_adapter import ModelServiceAdapter
import time

logger = logging.getLogger(__name__)

class MiniCPMoModelAdapter(ModelServiceAdapter):
    """Adapter for MiniCPM-o-2_6 model."""
    def load(self, timeout: float = 1200.0) -> bool:
        start_time = time.time()
        try:
            logger.info(f"Loading MiniCPM-o model: {self.config['model_config_id']} ({self.config['model_name']})")
            device_index = int(self.config['device'].split(':')[-1]) if 'cuda' in self.config['device'] else 0
            _, vram_used, vram_available = self._check_vram(device_index)
            
            # Check minimum VRAM requirement
            if vram_available < self.config.get('min_vram_needed', 10.0):
                logger.error(f"Insufficient VRAM: {vram_available:.2f} GB available, {self.config.get('min_vram_needed', 10.0)} GB needed")
                return False

            local_exists = os.path.exists(self.config['local_path'])
            logger.info("Loading model directly...")
            self.model = AutoModel.from_pretrained(
                self.config['local_path'] if local_exists else self.config['model_name'],
                trust_remote_code=True,
                attn_implementation='sdpa',
                torch_dtype=self.config.get('torch_dtype', torch.float16),
                local_files_only=local_exists
            )
            logger.info("Initializing TTS...")
            self.model.init_tts()
            logger.info(f"Moving model to device {self.config['device']}...")
            self.model = self.model.eval().to(device=self.config['device'])
            torch.cuda.synchronize(device_index)
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['local_path'] if local_exists else self.config['model_name'],
                trust_remote_code=True,
                local_files_only=local_exists
            )
            logger.info("Tokenizer loaded")
            self._save_model_locally()

            torch.cuda.synchronize(device_index)
            vram_used = torch.cuda.memory_allocated(device_index) / 1e9
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds, VRAM used: {vram_used:.2f} GB")
            logger.info(f"CUDA memory summary after loading:\n{torch.cuda.memory_summary(device_index)}")
            try:
                logger.info(f"GPU utilization after loading: {torch.cuda.utilization(device_index)}%")
            except Exception as e:
                logger.warning(f"Failed to retrieve GPU utilization after loading: {str(e)}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}", exc_info=True)
            self.model = None
            self.tokenizer = None
            return False

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            params = request.get('params', {})
            messages = request.get('messages', [])
            if not messages:
                raise ValueError("Messages required for MiniCPM-o model")
            processed_messages = []
            for msg in messages:
                if 'role' not in msg or 'content' not in msg:
                    raise ValueError("Each message must have 'role' and 'content'")
                content = []
                for item in msg['content']:
                    if isinstance(item, str) and item.startswith("base64:"):
                        audio_data = base64.b64decode(item[7:])
                        audio_array = np.frombuffer(audio_data, dtype=np.float32)
                        content.append(audio_array)
                    else:
                        content.append(item)
                processed_messages.append({'role': msg['role'], 'content': content})

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_audio_path = os.path.join(temp_dir, f"output_{uuid.uuid4()}.wav")
                params['output_audio_path'] = temp_audio_path
                response = self.model.chat(
                    msgs=processed_messages,
                    tokenizer=self.tokenizer,
                    **params
                )
                if not os.path.exists(temp_audio_path):
                    raise ValueError(f"Audio file not found at {temp_audio_path}")
                with open(temp_audio_path, 'rb') as f:
                    audio_data = f.read()
                    audio_base64 = base64.b64encode(audio_data).decode('utf-8')
                return {
                    'status': 'okay',
                    'message': 'Prompt processed successfully',
                    'result': {
                        'files': {
                            'output_audio_path': {
                                'path': temp_audio_path,
                                'data': f"base64:{audio_base64}"
                            }
                        }
                    }
                }
        except Exception as e:
            logger.error(f"Error processing prompt: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f"Prompt processing failed: {str(e)}",
                'result': {}
            }