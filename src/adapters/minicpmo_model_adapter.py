import logging
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, Any
import os
import tempfile
import uuid
import asyncio
import gc
from .base_adapter import ModelServiceAdapter

logger = logging.getLogger(__name__)

class MiniCPMoModelAdapter(ModelServiceAdapter):
    """Adapter for MiniCPM-o-2_6 model."""
    async def load(self, timeout: float = 1200.0) -> bool:
        """Load MiniCPM-o model with configuration from config.yml and TTS initialization."""
        try:
            logger.info(f"Loading MiniCPM-o model: {self.config['model_config_id']} ({self.config['model_name']})")
            # Debug config values
            logger.debug(f"Config parameters: {self.config.get('parameters', {})}")
            if not self.config.get('parameters', {}).get('load_in_4bit', False):
                logger.warning("load_in_4bit not set in config, forcing 4-bit quantization")
                self.config['parameters'] = self.config.get('parameters', {})
                self.config['parameters']['load_in_4bit'] = True

            success = await self.load_model_internal(AutoModel, AutoTokenizer, timeout)
            if success:
                logger.info("Initializing TTS...")
                self.model.init_tts()
                logger.info(f"Model and TTS loaded on {self.config['device']}")
                # Debug VRAM after TTS
                device_index = int(self.device.split(':')[-1]) if 'cuda' in self.device else 0
                _, vram_used, _ = self._check_vram(device_index)
                logger.debug(f"VRAM after TTS init: {vram_used:.2f} GB")
            return success
        except Exception as e:
            logger.error(f"Failed to load MiniCPM-o model: {str(e)}", exc_info=True)
            return False

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle audio model request."""
        try:
            request = await self.preprocess_input(request)
            params = request.get('params', {})
            messages = request.get('messages', [])
            if not messages:
                raise ValueError("Messages required for MiniCPM-o model")

            processed_messages = []
            for msg in messages:
                if 'role' not in msg :
                    msg['role'] = 'user'

                if 'content' not in msg:
                    msg['content'] = ''
                
                #if content != None:
                content = []
                for item in msg['content']:
                    if isinstance(item, str) and item.startswith("base64:"):
                        content.append(np.frombuffer(request.get('decoded_data', b''), dtype=np.float32))
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
                return await self.postprocess_output(audio_data, output_type='audio/wav')
        except Exception as e:
            return self._handle_error(e)

    async def unload(self) -> None:
        """Unload the model and free resources, including TTS."""
        try:
            logger.info("Cleaning up TTS resources for minicpm-o_2_6")
            if hasattr(self.model, 'tts'):
                del self.model.tts
                self.model.tts = None
                gc.collect()
                if 'cuda' in self.device:
                    torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error cleaning up TTS resources: {str(e)}", exc_info=True)
        await super().unload()