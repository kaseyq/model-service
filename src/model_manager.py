import importlib
import logging
import threading
import torch
import time
from typing import Dict, Any
from src.common.config_utils import load_config
from src.adapters.base_adapter import ModelServiceAdapter
import os
import asyncio

logger = logging.getLogger(__name__)

class ModelManager:
    current_model_id = None
    current_adapter = None
    model_lock = threading.Lock()
    server_state = "down"
    busy_reason = None
    server_start_time = time.time()
    config = load_config()
    API_VERSION = config['api_version']
    MODEL_REGISTRY = config.get('models', [])
    MODEL_LOOKUP = {cfg['model_config_id']: cfg for cfg in MODEL_REGISTRY}
    ADAPTER_REGISTRY = {}

    @classmethod
    def register_adapter(cls, adapter_name: str, adapter_class: Any):
        """Register a new adapter class."""
        cls.ADAPTER_REGISTRY[adapter_name] = adapter_class
        logger.info(f"Registered adapter: {adapter_name}")

    @classmethod
    def discover_adapters(cls, adapters_dir: str = "src.adapters"):
        """Dynamically discover adapter classes."""
        adapters_dir = adapters_dir.replace('.', '/')
        for filename in os.listdir(adapters_dir):
            if filename.endswith('_adapter.py'):
                module_name = filename[:-3]
                try:
                    module = importlib.import_module(f"src.adapters.{module_name}")
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, ModelServiceAdapter) and attr != ModelServiceAdapter:
                            cls.register_adapter(attr_name, attr)
                except Exception as e:
                    logger.error(f"Failed to import adapter module {module_name}: {str(e)}", exc_info=True)

    @classmethod
    async def load(cls, model_config_id: str, timeout: float = 999999) -> tuple[bool, str]:
        """Load a model using its adapter."""
        if model_config_id not in cls.MODEL_LOOKUP:
            logger.error(f"Model config ID {model_config_id} not found")
            return False, f"Model config ID {model_config_id} not found"

        cfg = cls.MODEL_LOOKUP[model_config_id]
        with cls.model_lock:
            if model_config_id == cls.current_model_id and cls.current_adapter is not None:
                cls.server_state = "ready"
                cls.busy_reason = None
                return True, f"Model {model_config_id} is already loaded"

            cls.server_state = "busy"
            cls.busy_reason = f"loading model {model_config_id}"
            logger.info(f"Attempting to load model {model_config_id}")

            # Log VRAM before loading
            device_index = int(cfg.get('device', 'cuda:0').split(':')[-1]) if 'cuda' in cfg.get('device', 'cuda:0') else 0
            _, vram_used, _ = cls._check_vram_static(device_index)
            logger.info(f"VRAM before loading {model_config_id}: {vram_used:.2f} GB")

            # Unload current model to free resources
            if cls.current_adapter is not None:
                try:
                    success, msg = await cls.unload()
                    if not success:
                        logger.warning(f"Failed to unload current model: {msg}")
                except Exception as e:
                    logger.error(f"Error unloading current model: {str(e)}", exc_info=True)

            try:
                adapter_class_name = cfg.get('adapter_class')
                adapter_class = cls.ADAPTER_REGISTRY.get(adapter_class_name)
                if not adapter_class:
                    raise ValueError(f"Adapter {adapter_class_name} not registered")
                adapter = adapter_class(cfg)
                if await adapter.load(timeout):
                    cls.current_adapter = adapter
                    cls.current_model_id = model_config_id
                    cls.server_state = "ready"
                    cls.busy_reason = None
                    logger.info(f"Successfully loaded model: {model_config_id}")
                    # Log VRAM after loading
                    _, vram_used, _ = cls._check_vram_static(device_index)
                    logger.info(f"VRAM after loading {model_config_id}: {vram_used:.2f} GB")
                    return True, f"Successfully loaded model: {model_config_id}"
                else:
                    logger.error(f"Model {model_config_id} failed to load")
                    return False, f"Model {model_config_id} failed to load"
            except Exception as e:
                cls.server_state = "error"
                cls.busy_reason = None
                logger.error(f"Failed to load model {model_config_id}: {str(e)}", exc_info=True)
                return False, f"Failed to load model {model_config_id}: {str(e)}"

    @classmethod
    async def unload(cls) -> tuple[bool, str]:
        """Unload the current model and free resources."""
        with cls.model_lock:
            if cls.server_state == "busy":
                msg = f"Cannot unload: server is busy due to {cls.busy_reason or 'unknown reason'}"
                logger.warning(msg)
                return False, msg
            if cls.current_adapter is None:
                msg = "No model loaded"
                logger.info(msg)
                return False, msg

            cls.server_state = "busy"
            cls.busy_reason = f"unloading model {cls.current_model_id}"
            logger.info(f"Server state: {cls.server_state} due to {cls.busy_reason}")

            try:
                # Log VRAM before unloading
                device_index = 0
                if cls.current_adapter and 'cuda' in cls.current_adapter.device:
                    device_index = int(cls.current_adapter.device.split(':')[-1])
                _, vram_used, _ = cls._check_vram_static(device_index)
                logger.info(f"VRAM before unload: {vram_used:.2f} GB")

                await cls.current_adapter.unload()
                cls.current_adapter = None
                cls.current_model_id = None
                cls.server_state = "ready"
                cls.busy_reason = None
                msg = "Model unloaded"
                logger.info(msg)

                # Log VRAM after unloading
                _, vram_used, _ = cls._check_vram_static(device_index)
                logger.info(f"VRAM after unload: {vram_used:.2f} GB")
                return True, msg
            except Exception as e:
                cls.server_state = "error"
                cls.busy_reason = None
                msg = f"Failed to unload model: {str(e)}"
                logger.error(msg, exc_info=True)
                return False, msg

    @classmethod
    def _check_vram_static(cls, device_index: int) -> tuple[float, float, float]:
        """Static method to check VRAM status."""
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