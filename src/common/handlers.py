import socket
import socketserver



class ModelRequestHandler(socketserver.BaseRequestHandler):
    """Handle client requests for model tasks."""
    def handle(self):
        global current_model_id, current_adapter, server_state, busy_reason
        try:
            # Receive data in chunks
            data = b""
            max_data_size = config['max_data_size']  # From config
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
                "version": API_VERSION,
                "server_state": server_state,
                "model_config_id": current_model_id,
                "message": "",
                "status": "okay",
                "payload": {}
            }

            # Validate version
            if "version" not in request or request["version"] != str(API_VERSION):
                logger.warning(f"Invalid or missing version: expected {API_VERSION}, got {request.get('version')}")
                response_data.update({
                    "status": "error",
                    "message": f"Invalid or missing version: expected {API_VERSION}"
                })
                response_json = json.dumps(response_data)
                logger.info(f"Raw sent data: {truncate_log(response_json)} [total length: {len(response_json)}]")
                self.request.sendall(response_json.encode('utf-8'))
                logger.info("Response sent")
                return

            with model_lock:
                task = request.get('task')
                if task not in ['list_models', 'status'] and 'model_config_id' not in request:
                    response_data.update({
                        "status": "error",
                        "message": "model_config_id required for all tasks except list_models and status"
                    })
                else:
                    model_config_id = request.get('model_config_id', None)

                    if task in ['load_model', 'unload_model'] and server_state == "busy":
                        logger.warning(f"Rejected {task} request: server is busy due to {busy_reason or 'unknown reason'}")
                        response_data.update({
                            "status": "error",
                            "message": f"Cannot perform {task}: server is busy due to {busy_reason or 'unknown reason'}"
                        })
                    elif task == 'load_model':
                        if model_config_id == current_model_id:
                            success = true
                        elif model_config_id not in MODEL_LOOKUP:
                            response_data.update({
                                "status": "error",
                                "message": f"Invalid model_config_id: {model_config_id}"
                            })
                        else:
                            success = load_model(model_config_id, config['timeout'])
                            response_data.update({
                                "server_state": server_state,
                                "model_config_id": current_model_id,
                                "status": "okay" if success else "error",
                                "message": f"Model {model_config_id} {'loaded' if success else 'failed to load'}"
                            })
                    elif task == 'unload_model':
                        if model_config_id != current_model_id:
                            response_data.update({
                                "status": "error",
                                "message": f"Cannot unload {model_config_id}; current model is {current_model_id or 'none'}"
                            })
                        else:
                            server_state = "busy"
                            busy_reason = f"unloading model {model_config_id}"
                            logger.info(f"Server state: {server_state} due to {busy_reason}")
                            if current_adapter is not None:
                                current_adapter.unload()
                                current_adapter = None
                                current_model_id = None
                            server_state = "ready"
                            busy_reason = None
                            response_data.update({
                                "server_state": server_state,
                                "model_config_id": current_model_id,
                                "status": "okay",
                                "message": "Model unloaded"
                            })
                    elif task == 'list_models':
                        current_cfg = MODEL_LOOKUP.get(current_model_id, {}) if current_model_id else {}
                        current_cfg = {k: str(v) for k, v in current_cfg.items() if k != 'local_path' and k != 'adapter_class'}
                        response_data.update({
                            "status": "okay",
                            "message": "Model list retrieved",
                            "payload": {
                                "models": [cfg['model_config_id'] for cfg in MODEL_REGISTRY],
                                "current_model": current_model_id,
                                "current_config": current_cfg
                            }
                        })
                    elif task == 'prompt':
                        if server_state != "ready":
                            logger.warning(f"Rejected prompt request: server is busy due to {busy_reason or 'unknown reason'}")
                            response_data.update({
                                "status": "error",
                                "message": f"Cannot process prompt: server is busy due to {busy_reason or 'unknown reason'}"
                            })
                        elif model_config_id != current_model_id or current_adapter is None:
                            logger.warning(f"Rejected prompt request: model {model_config_id} not loaded")
                            response_data.update({
                                "status": "error",
                                "message": f"Cannot process prompt: model {model_config_id} not loaded"
                            })
                        else:
                            try:
                                adapter_response = current_adapter.handle_request(request)
                                response_data.update({
                                    "server_state": server_state,
                                    "model_config_id": current_model_id,
                                    "status": adapter_response.get("status", "error"),
                                    "message": adapter_response.get("message", "Prompt processing failed"),
                                    "payload": {"result": adapter_response.get("result", {})}
                                })
                            except Exception as e:
                                logger.error(f"Adapter failed to process prompt: {str(e)}")
                                response_data.update({
                                    "server_state": server_state,
                                    "model_config_id": current_model_id,
                                    "status": "error",
                                    "message": f"Prompt processing failed: {str(e)}",
                                    "payload": {"result": {}}
                                })
                    elif task == 'status':
                        cfg = MODEL_LOOKUP.get(current_model_id, {}) if current_model_id else {'device': 'cuda:0'}
                        device_index = int(cfg['device'].split(':')[-1]) if 'cuda' in cfg['device'] else 0
                        vram_total = torch.cuda.get_device_properties(device_index).total_memory / 1e9
                        vram_used = torch.cuda.memory_allocated(device_index) / 1e9
                        uptime = time.time() - server_start_time
                        response_data.update({
                            "status": "okay",
                            "message": "Server status retrieved",
                            "payload": {
                                "health": server_state,
                                "busy_reason": busy_reason,
                                "current_model": current_model_id,
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

            response_json = json.dumps(response_data)
            logger.info(f"response : {truncate_log(response_json)} [total length: {len(response_json)}]")
            self.request.sendall(response_json.encode('utf-8'))
            logger.info("Response sent")

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            server_state = "error"
            busy_reason = None
            response_data = {
                "version": API_VERSION,
                "server_state": server_state,
                "model_config_id": current_model_id,
                "message": f"Error processing request: {str(e)}",
                "status": "error",
                "payload": {"result": {}}
            }
            response_json = json.dumps(response_data)
            logger.info(f"Raw sent data: {truncate_log(response_json)} [total length: {len(response_json)}]")
            self.request.sendall(response_json.encode('utf-8'))

