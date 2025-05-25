import logging
import torch
import threading
from typing import Dict, Any
import time

from src.common.file_utils import load_yaml
from src.common.path_utils import PathUtil
from src.adapters.causal_lm_adapter import CausalLMAdapter
from src.adapters.minicpmo_model_adapter import MiniCPMoModelAdapter
from src.common.config_utils import load_config

logger = logging.getLogger(__name__)
class ModelManager():
    current_model_id = None
    current_adapter = None
    model_lock = threading.Lock()

    server_state = "down"  # error, ready, busy
    busy_reason = None  # Track reason for busy state
    server_start_time = time.time()
    
    config = load_config()
    #config = load_yaml([PathUtil.get_path("config.yml"), PathUtil.get_path("config.yaml")])
    API_VERSION = config['api_version']
    MODEL_REGISTRY = config['models']
    MODEL_LOOKUP = {cfg['model_config_id']: cfg for cfg in MODEL_REGISTRY}
    API_VERSION = config['api_version']

    def unload() -> tuple[bool, str]:
        #global server_state, busy_reason, server_start_time, API_VERSION, config
        success = False
        msg = "unknown error"
        with ModelManager.model_lock:
            if ModelManager.server_state == "busy":
                msg = f"Cannot unload: server is busy due to {ModelManager.busy_reason or 'unknown reason'}"
                success = False
            elif ModelManager.current_adapter is not None:
                ModelManager.server_state = "busy"
                ModelManager.busy_reason = "unloading model via CLI"
                logger.info(f"Server state: {ModelManager.server_state} due to {ModelManager.busy_reason}")
                try:
                    ModelManager.current_adapter.unload()
                    ModelManager.current_adapter = None
                    ModelManager.current_model_id = None
                    ModelManager.server_state = "ready"
                    ModelManager.busy_reason = None
                    logger.info("Model unloaded via CLI")
                    #print("Model unloaded")

                    msg = f"Model unloaded"
                    success = True

                except Exception as e:
                    msg = f"{str(e)}"
                    success = False

                    logger.error(f"Failed to unload model via CLI: {str(e)}", exc_info=True)
                    #print(f"Failed to unload model: {str(e)}")
            else:
                msg = "No model loaded"
                success = False

        return success,msg

    def load(model_config_id: str, timeout: float = 999999) -> tuple[bool, str]:
        success = False
        msg = "unknown error"
        """Load a model using its adapter."""

        if model_config_id not in ModelManager.MODEL_LOOKUP:
            success = False
            msg = f"Model config ID {model_config_id} not found in registry"
            logger.error(msg)
            return success, msg
        
        cfg = ModelManager.MODEL_LOOKUP[model_config_id]
        with ModelManager.model_lock:
            logger.info(f"Acquired model_lock for loading {model_config_id}")
            try:
                # Check if the requested model is already loaded
                if model_config_id == ModelManager.current_model_id and ModelManager.current_adapter is not None:
                    logger.info(f"Model {model_config_id} is already loaded")
                    ModelManager.server_state = "ready"
                    ModelManager.busy_reason = None
                    success = True
                    msg = f"Model {model_config_id} is already loaded"
                    return success, msg

                ModelManager.server_state = "busy"
                ModelManager.busy_reason = f"loading model {model_config_id}"
                logger.info(f"Server state: {ModelManager.server_state} due to {ModelManager.busy_reason}")
                device_index = int(cfg['device'].split(':')[-1]) if 'cuda' in cfg['device'] else 0
                
                try:
                    logger.info(f"Checking CUDA device {device_index} status...")
                    torch.cuda.init()
                    logger.info(f"CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(device_index)}")
                except Exception as e:
                    ModelManager.server_state = "error"
                    ModelManager.busy_reason = None
                    success = False
                    msg =f"Failed to check VRAM or CUDA status: {str(e)}"
                    logger.error(msg, exc_info=True)

                if ModelManager.current_adapter is not None:
                    logger.info(f"Unloading current model: {ModelManager.current_model_id}")
                    try:
                        ModelManager.current_adapter.unload()
                        ModelManager.current_adapter = None
                        ModelManager.current_model_id = None
                    except Exception as e:
                        ModelManager.busy_reason = None
                        ModelManager.server_state = "error"
                        success = False
                        msg = f"Failed to unload model {ModelManager.current_model_id}: {str(e)}"
                        logger.error(msg, exc_info=True)

                if ModelManager.server_state != "error" :
                    logger.info(f"Creating adapter for {model_config_id}")
                    try:
                        adapter_class_map = {
                            'CausalLMAdapter': CausalLMAdapter,
                            'MiniCPMoModelAdapter': MiniCPMoModelAdapter
                        }
                        adapter = adapter_class_map[cfg['adapter_class']](cfg)
                    except Exception as e:
                        success = False
                        msg = f"Failed to create adapter for {model_config_id}: {str(e)}"
                        logger.error(msg, exc_info=True)
                
                if ModelManager.server_state != "error" :
                    logger.info(f"Calling adapter.load for {model_config_id}")
                    try:
                        if adapter.load(timeout):
                            ModelManager.current_adapter = adapter
                            ModelManager.current_model_id = model_config_id
                            ModelManager.server_state = "ready"
                            ModelManager.busy_reason = None
                            success = True
                            msg =f"Successfully loaded model: {model_config_id}"
                            logger.info(msg)

                        else:
                            success = False
                            msg = f"Model {model_config_id} failed to load"
                            logger.error(msg)
                    except Exception as e:
                        success = False
                        msg = f"Exception during adapter.load for {model_config_id}: {str(e)}"
                        logger.error(msg, exc_info=True)

            except Exception as e:
                success = False
                msg = f"Unexpected error loading model {model_config_id}: {str(e)}"
                logger.error(msg, exc_info=True)

            if success != True :
                ModelManager.current_adapter = None
                ModelManager.current_model_id = None
                ModelManager.server_state = "error"
                ModelManager.busy_reason = None

        return success,msg