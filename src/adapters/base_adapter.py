import logging
import torch
import os
import gc
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import asyncio
from src.common.config_utils import load_config

logger = logging.getLogger(__name__)

class ModelServiceAdapter(ABC):
    """Base adapter class for model services."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = config.get('device', 'cpu')

    @abstractmethod
    async def load(self, timeout: float = 1200.0) -> bool:
        """Asynchronously load the model and tokenizer."""
        pass

    @abstractmethod
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a model request asynchronously."""
        pass

    def _check_vram(self, device_index: int) -> tuple[float, float, float]:
        """Check CUDA device VRAM status."""
        try:
            torch.cuda.set_device(device_index)
            vram_total = torch.cuda.get_device_properties(device_index).total_memory
            vram_used = torch.cuda.memory_allocated(device_index)
            vram_available = vram_total - vram_used
            logger.debug(f"VRAM for cuda:{device_index}: {vram_used/1e9:.2f}/{vram_total/1e9:.2f} GB used")
            return vram_total / 1e9, vram_used / 1e9, vram_available / 1e9
        except Exception as e:
            logger.error(f"Error checking VRAM for device {device_index}: {e}")
            return 0, 0, 0

    def check_vram_requirements(self) -> bool:
        """Verify if available VRAM meets minimum requirements."""
        global_config = load_config()
        max_memory_config = self.config.get('max_memory', {})
        min_vram_needed_config = global_config.get('min_vram_needed', {})
        default_min_vram = self.config.get('min_vram_needed', 10.0)

        if max_memory_config:
            for device in max_memory_config:
                device_id = int(device.split(":")[1]) if ":" in device else device
                _, vram_used, vram_available = self._check_vram(device_id)
                min_vram_needed = min_vram_needed_config.get(f"cuda:{device_id}", default_min_vram) / 1e9
                if vram_available < min_vram_needed:
                    logger.error(f"Insufficient VRAM on device cuda:{device_id}: {vram_available:.2f} GB available, {min_vram_needed:.2f} GB needed")
                    return False
        else:
            device_index = int(self.device.split(':')[-1]) if 'cuda' in self.device else 0
            _, vram_used, vram_available = self._check_vram(device_index)
            if vram_available < default_min_vram:
                logger.error(f"Insufficient VRAM on device {self.device}: {vram_available:.2f} GB available, {default_min_vram:.2f} GB needed")
                return False
        return True

    def createDeviceMap(self) -> Dict[str, Any]:
        """Build device map from config for model components."""
        device_map_config = self.config.get('device_map', {})
        if not device_map_config:
            device_id = int(self.device.split(":")[1]) if ":" in self.device else self.device
            return {'': device_id}

        device_map = {}
        for component in ["model.embed_tokens", "model.norm", "lm_head"]:
            if component in device_map_config:
                device = device_map_config[component]
                device_id = int(device.split(":")[1]) if ":" in device else device
                device_map[component] = device_id

        layers_config = device_map_config.get("layers", {})
        for layer_range, device in layers_config.items():
            try:
                start, end = map(int, layer_range.split("-"))
                device_id = int(device.split(":")[1]) if ":" in device else device
                for i in range(start, end + 1):
                    device_map[f"model.layers.{i}"] = device_id
                    device_map[f"model.layers.{i}.input_layernorm"] = device_id
                    device_map[f"model.layers.{i}.post_attention_layernorm"] = device_id
                    device_map[f"model.layers.{i}.self_attn"] = device_id
                    device_map[f"model.layers.{i}.mlp"] = device_id
            except ValueError as e:
                logger.error(f"Invalid device_map layer range {layer_range}: {e}")
                raise ValueError(f"Invalid device_map configuration: {e}")
        return device_map

    async def load_model(self, model_class: Any, tokenizer_class: Any = None, timeout: float = 1200.0) -> bool:
        """Load model and optional tokenizer/processor with common steps."""
        try:
            if not self.check_vram_requirements():
                return False

            local_exists = os.path.exists(self.config['local_path'])
            device_map = self.createDeviceMap()
            max_memory = self.config.get('max_memory', {})

            logger.info(f"Loading model: {self.config['model_config_id']} ({self.config['model_name']})")
            self.model = model_class.from_pretrained(
                self.config['local_path'] if local_exists else self.config['model_name'],
                torch_dtype=self.config.get('parameters', {}).get('torch_dtype', torch.float16),
                device_map=device_map,
                max_memory=max_memory,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                local_files_only=local_exists
            )
            logger.info(f"Model loaded to device(s): {device_map}")

            if tokenizer_class:
                logger.info("Loading tokenizer/processor...")
                # Handle CLIPProcessor with slow_image_processor_class
                if 'CLIPProcessor' in str(tokenizer_class):
                    self.tokenizer = tokenizer_class.from_pretrained(
                        self.config['local_path'] if local_exists else self.config['model_name'],
                        trust_remote_code=True,
                        local_files_only=local_exists,
                        slow_image_processor_class=True
                    )
                else:
                    self.tokenizer = tokenizer_class.from_pretrained(
                        self.config['local_path'] if local_exists else self.config['model_name'],
                        trust_remote_code=True,
                        local_files_only=local_exists
                    )
                logger.info("Tokenizer/processor loaded")

            device_index = int(self.device.split(':')[-1]) if 'cuda' in self.device else 0
            torch.cuda.synchronize(device_index)
            await self._log_vram_status(device_index)

            if not local_exists:
                await self._save_model_locally()
            return True
        except Exception as e:
            logger.error(f"Failed to load model {self.config['model_config_id']}: {str(e)}", exc_info=True)
            self.model = None
            self.tokenizer = None
            return False

    async def _save_model_locally(self) -> None:
        """Save model and tokenizer to local path if not already present."""
        if not os.path.exists(self.config['local_path']):
            logger.info(f"Saving model to local path: {self.config['local_path']}")
            os.makedirs(self.config['local_path'], exist_ok=True)
            if self.model:
                self.model.save_pretrained(self.config['local_path'])
            if self.tokenizer:
                self.tokenizer.save_pretrained(self.config['local_path'])
            logger.info("Model and tokenizer saved")

    async def _log_vram_status(self, device_index: int) -> None:
        """Log VRAM and GPU utilization status."""
        _, vram_used, vram_available = self._check_vram(device_index)
        logger.info(f"VRAM used: {vram_used:.2f} GB, available: {vram_available:.2f} GB")
        logger.info(f"CUDA memory summary:\n{torch.cuda.memory_summary(device_index)}")
        try:
            logger.info(f"GPU utilization: {torch.cuda.utilization(device_index)}%")
        except Exception as e:
            logger.warning(f"Failed to retrieve GPU utilization: {str(e)}")

    def get_metadata(self) -> Dict[str, Any]:
        """Return model metadata from configuration."""
        try:
            metadata = {
                'input_types': self.config.get('input_types', []),
                'output_types': self.config.get('output_types', []),
                'model_type': self.config.get('type', 'unknown'),
                'version': str(self.config.get('version', '1.0'))
            }
            if not metadata['input_types'] or not metadata['output_types']:
                logger.warning(f"Incomplete metadata for model {self.config.get('model_config_id', 'unknown')}: input_types or output_types missing")
            return metadata
        except KeyError as e:
            logger.error(f"Missing required config field for metadata: {e}")
            raise ValueError(f"Invalid configuration: missing required field {e}")

    async def unload(self) -> None:
        """Unload the model and free resources."""
        try:
            model_id = self.config.get('model_config_id', 'unknown')
            logger.info(f"Unloading model {model_id}")
            if self.model is not None:
                # Explicitly delete model to release references
                del self.model
                self.model = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            if 'cuda' in self.device:
                device_index = int(self.device.split(':')[-1]) if ':' in self.device else 0
                # Force garbage collection
                gc.collect()
                # Clear CUDA memory cache
                torch.cuda.empty_cache()
                # Reset peak memory stats
                torch.cuda.reset_peak_memory_stats(device_index)
                # Synchronize to ensure all operations complete
                torch.cuda.synchronize(device_index)
                # Log VRAM after cleanup
                await self._log_vram_status(device_index)
        except Exception as e:
            logger.error(f"Error during unload of model {self.config.get('model_config_id', 'unknown')}: {str(e)}", exc_info=True)

    async def preprocess_input(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and preprocess input data, including base64 decoding."""
        supported_inputs = self.get_metadata()['input_types']
        input_type = request.get('input_type', 'text/plain')
        if input_type not in supported_inputs:
            raise ValueError(f"Unsupported input type: {input_type}. Supported: {supported_inputs}")

        if input_type in ['audio/wav', 'image/jpeg', 'image/png'] and 'data' in request:
            from src.common.file_utils import decode_base64_data
            try:
                request['decoded_data'] = decode_base64_data(request['data'])
            except Exception as e:
                raise ValueError(f"Failed to decode base64 data: {str(e)}")
        return request

    async def postprocess_output(self, output: Any, output_type: str = 'text/plain') -> Dict[str, Any]:
        """Format output for client response, including base64 encoding if needed."""
        if output_type in ['audio/wav', 'image/jpeg', 'image/png']:
            from src.common.file_utils import encode_base64_data
            try:
                output = encode_base64_data(output)
            except Exception as e:
                raise ValueError(f"Failed to encode output as base64: {str(e)}")
        return {
            'status': 'okay',
            'message': 'Request processed successfully',
            'result': output
        }

    def _handle_error(self, error: Exception) -> Dict[str, Any]:
        """Format error response."""
        logger.error(f"Error processing request: {str(error)}", exc_info=True)
        return {
            'status': 'error',
            'message': f"Request failed: {str(error)}",
            'result': {}
        }