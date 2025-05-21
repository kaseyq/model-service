import logging
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any
import os
import tempfile
import uuid
import base64
import gc
from .base_adapter import ModelServiceAdapter

logger = logging.getLogger(__name__)

class MiniCPMoModelAdapter(ModelServiceAdapter):
    """Adapter for MiniCPM-o-2_6 model."""
    def load(self, timeout: float = 300.0) -> bool:
        from time import time
        start_time = time()
        try:
            logger.info(f"Loading MiniCPM-o model: {self.config['model_config_id']} ({self.config['model_name']})")
            local_exists = os.path.exists(self.config['local_path'])
            self.model = AutoModel.from_pretrained(
                self.config['local_path'] if local_exists else self.config['model_name'],
                trust_remote_code=True,
                attn_implementation='sdpa',
                torch_dtype=self.config.get('torch_dtype', torch.float16),
                local_files_only=local_exists
            )
            self.model.init_tts()
            self.model = self.model.eval().to(device=self.config['device'])
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config['local_path'] if local_exists else self.config['model_name'],
                trust_remote_code=True,
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
        logger.info("MiniCPM-o model unloaded and VRAM cleared")

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
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
                'status': 'success',
                'files': {
                    'output_audio_path': {
                        'path': temp_audio_path,
                        'data': f"base64:{audio_base64}"
                    }
                }
            }