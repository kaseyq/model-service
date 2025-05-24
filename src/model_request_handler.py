import json
import logging
import socketserver
import time
import torch
from typing import Dict, Any
from src.model_manager import ModelManager
from src.common.logging_utils import truncate_log

logger = logging.getLogger(__name__)

class ModelRequestHandler(socketserver.BaseRequestHandler):
    """Handle client requests for model tasks."""
    def handle(self):
        try:
            # Receive data in chunks
            data = b""
            max_data_size = ModelManager.config['max_data_size']
            while True:
                chunk = self.request.recv(1048576)  # 1 MB chunks
                if len(data) + len(chunk) > max_data_size:
                    raise ValueError(f"Received data exceeds maximum size of {max_data_size} bytes")
                data += chunk
                try:
                    json.loads(data.decode('utf-8'))
                    break
                except json.JSONDecodeError:
                    if not chunk:
                        raise ValueError("Incomplete JSON received")
                    continue

            raw_data = data.decode('utf-8')
            logger.info(f"Raw received data: {truncate_log(raw_data)} [total length: {len(raw_data)}]")

            # Parse JSON request
            request = json.loads(raw_data)
            logger.info(f"Request: {truncate_log(request)}")

            # Initialize response
            response_data = {
                "version": ModelManager.API_VERSION,
                "server_state": ModelManager.server_state,
                "model_config_id": ModelManager.current_model_id,
                "message": "",
                "status": "okay",
                "payload": {}
            }

            # Validate version
            if "version" not in request or str(request["version"]) != str(ModelManager.API_VERSION):
                logger.warning(f"Invalid or missing version: expected {ModelManager.API_VERSION}, got {request.get('version')}")
                response_data.update({
                    "status": "error",
                    "message": f"Invalid or missing version: expected {ModelManager.API_VERSION}"
                })
            else:
                task = request.get('task')
                if task not in ['list_models', 'status'] and 'model_config_id' not in request:
                    response_data.update({
                        "status": "error",
                        "message": "model_config_id required for all tasks except list_models and status"
                    })
                else:
                    model_config_id = request.get('model_config_id', None)

                    if task in ['load_model', 'unload_model'] and ModelManager.server_state == "busy":
                        logger.warning(f"Rejected {task} request: server is busy due to {ModelManager.busy_reason or 'unknown reason'}")
                        response_data.update({
                            "status": "error",
                            "message": f"Cannot perform {task}: server is busy due to {ModelManager.busy_reason or 'unknown reason'}"
                        })
                    elif task == 'load_model':
                        if model_config_id not in ModelManager.MODEL_LOOKUP:
                            response_data.update({
                                "status": "error",
                                "message": f"Invalid model_config_id: {model_config_id}"
                            })
                        else:
                            success = ModelManager.load(model_config_id, ModelManager.config['timeout'])  # Fixed: Changed load_model to ModelManager.load
                            response_data.update({
                                "server_state": ModelManager.server_state,
                                "model_config_id": ModelManager.current_model_id,
                                "status": "okay" if success else "error",
                                "message": f"Model {model_config_id} {'already loaded' if success and model_config_id == ModelManager.current_model_id else 'loaded' if success else 'failed to load'}"
                            })
                    elif task == 'unload_model':
                        if model_config_id != ModelManager.current_model_id:
                            response_data.update({
                                "status": "error",
                                "message": f"Cannot unload {model_config_id}; current model is {ModelManager.current_model_id or 'none'}"
                            })
                        else:
                            ModelManager.server_state = "busy"
                            ModelManager.busy_reason = f"unloading model {model_config_id}"
                            logger.info(f"Server state: {ModelManager.server_state} due to {ModelManager.busy_reason}")
                            try:
                                if ModelManager.current_adapter is not None:
                                    ModelManager.current_adapter.unload()
                                    ModelManager.current_adapter = None
                                    ModelManager.current_model_id = None
                                ModelManager.server_state = "ready"
                                ModelManager.busy_reason = None
                                response_data.update({
                                    "server_state": ModelManager.server_state,
                                    "model_config_id": ModelManager.current_model_id,
                                    "status": "okay",
                                    "message": "Model unloaded"
                                })
                            except Exception as e:
                                logger.error(f"Failed to unload model {model_config_id}: {str(e)}", exc_info=True)
                                response_data.update({
                                    "server_state": "error",
                                    "model_config_id": ModelManager.current_model_id,
                                    "status": "error",
                                    "message": f"Failed to unload model: {str(e)}"
                                })
                    elif task == 'list_models':
                        current_cfg = ModelManager.MODEL_LOOKUP.get(ModelManager.current_model_id, {}) if ModelManager.current_model_id else {}
                        current_cfg = {k: str(v) for k, v in current_cfg.items() if k != 'local_path' and k != 'adapter_class'}
                        response_data.update({
                            "status": "okay",
                            "message": "Model list retrieved",
                            "payload": {
                                "models": [cfg['model_config_id'] for cfg in ModelManager.MODEL_REGISTRY],
                                "current_model": ModelManager.current_model_id,
                                "current_config": current_cfg
                            }
                        })
                    elif task == 'prompt':
                        if ModelManager.server_state != "ready":
                            logger.warning(f"Rejected prompt request: server is busy due to {ModelManager.busy_reason or 'unknown reason'}")
                            response_data.update({
                                "status": "error",
                                "message": f"Cannot process prompt: server is busy due to {ModelManager.busy_reason or 'unknown reason'}"
                            })
                        elif model_config_id != ModelManager.current_model_id or ModelManager.current_adapter is None:
                            logger.warning(f"Rejected prompt request: model {model_config_id} not loaded")
                            response_data.update({
                                "status": "error",
                                "message": f"Cannot process prompt: model {model_config_id} not loaded"
                            })
                        else:
                            try:
                                adapter_response = ModelManager.current_adapter.handle_request(request)
                                response_data.update({
                                    "server_state": ModelManager.server_state,
                                    "model_config_id": ModelManager.current_model_id,
                                    "status": adapter_response.get("status", "error"),
                                    "message": adapter_response.get("message", "Prompt processing failed"),
                                    "payload": {"result": adapter_response.get("result", {})}
                                })
                            except Exception as e:
                                logger.error(f"Adapter failed to process prompt: {str(e)}", exc_info=True)
                                response_data.update({
                                    "server_state": ModelManager.server_state,
                                    "model_config_id": ModelManager.current_model_id,
                                    "status": "error",
                                    "message": f"Prompt processing failed: {str(e)}",
                                    "payload": {"result": {}}
                                })
                    elif task == 'status':
                        cfg = ModelManager.MODEL_LOOKUP.get(ModelManager.current_model_id, {}) if ModelManager.current_model_id else {'device': 'cuda:0'}
                        device_index = int(cfg['device'].split(':')[-1]) if 'cuda' in cfg['device'] else 0
                        try:
                            vram_total = torch.cuda.get_device_properties(device_index).total_memory / 1e9
                            vram_used = torch.cuda.memory_allocated(device_index) / 1e9
                        except Exception as e:
                            logger.error(f"Failed to retrieve VRAM status: {str(e)}", exc_info=True)
                            vram_total = vram_used = 0.0
                        uptime = time.time() - ModelManager.server_start_time
                        response_data.update({
                            "status": "okay",
                            "message": "Server status retrieved",
                            "payload": {
                                "health": ModelManager.server_state,
                                "busy_reason": ModelManager.busy_reason,
                                "current_model": ModelManager.current_model_id,
                                "vram_usage": {
                                    "total_gb": round(vram_total, 2),
                                    "used_gb": round(vram_used, 2),
                                    "free_gb": round(vram_total - vram_used, 2)
                                },
                                "uptime_seconds": round(uptime, 2)
                            }
                        })
                    else:
                        response_data.update({
                            "status": "error",
                            "message": f"Invalid task: {task}"
                        })

            logger.info(f"Response data: {truncate_log(response_data)}")
            response_json = json.dumps(response_data)
            logger.info(f"Raw sent data length: {len(response_json)}")
            self.request.sendall(response_json.encode('utf-8'))
            logger.info("Response sent")

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}", exc_info=True)
            ModelManager.server_state = "error"
            ModelManager.busy_reason = None
            response_data = {
                "version": ModelManager.API_VERSION,
                "server_state": ModelManager.server_state,
                "model_config_id": ModelManager.current_model_id,
                "message": f"Error processing request: {str(e)}",
                "status": "error",
                "payload": {"result": {}}
            }
            logger.info(f"Response data: {truncate_log(response_data)}")
            response_json = json.dumps(response_data)
            logger.info(f"Raw sent data length: {len(response_json)}")
            try:
                self.request.sendall(response_json.encode('utf-8'))
            except Exception as send_error:
                logger.error(f"Failed to send error response: {str(send_error)}", exc_info=True)