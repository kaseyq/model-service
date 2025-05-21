import json
import logging
import socketserver
import torch
from typing import Dict, Any
import threading
import os
import gc
from queue import Queue
import argparse
import yaml
from adapters.base_adapter import ModelServiceAdapter
from adapters.causal_lm_adapter import CausalLMAdapter
from adapters.minicpmo_model_adapter import MiniCPMoModelAdapter

# Set up logging (static, to be addressed in Docker Compose)
logging.basicConfig(level=logging.INFO, filename="model_server.log")
logger = logging.getLogger(__name__)

# Load configuration from config.yml or config.yaml
def load_config(config_paths: list = ["config.yml", "config.yaml"]) -> Dict[str, Any]:
    """Load configuration from YAML file, trying multiple paths."""
    for config_path in config_paths:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            if not config or 'models' not in config:
                raise ValueError(f"Invalid {config_path}: 'models' key missing")
            logger.info(f"Loaded configuration from {config_path}")
            
            # Validate required top-level config
            required_keys = ['host', 'port', 'timeout']
            for key in required_keys:
                if key not in config:
                    raise ValueError(f"Missing required config key: {key}")
            
            # Validate and process models
            for model_config in config['models']:
                if 'model_config_id' not in model_config or 'model_name' not in model_config or 'adapter_class' not in model_config:
                    raise ValueError("Each model must have 'model_config_id', 'model_name', and 'adapter_class'")
                # Convert torch_dtype string to torch type
                if 'torch_dtype' in model_config:
                    dtype_map = {
                        'bfloat16': torch.bfloat16,
                        'float16': torch.float16,
                        'float32': torch.float32
                    }
                    model_config['torch_dtype'] = dtype_map.get(model_config['torch_dtype'], torch.float16)
                # Resolve adapter class
                adapter_class_map = {
                    'CausalLMAdapter': CausalLMAdapter,
                    'MiniCPMoModelAdapter': MiniCPMoModelAdapter
                }
                adapter_class = adapter_class_map.get(model_config['adapter_class'])
                if not adapter_class:
                    raise ValueError(f"Invalid adapter_class: {model_config['adapter_class']}")
                model_config['adapter_class'] = adapter_class
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, trying next")
        except Exception as e:
            logger.error(f"Failed to load {config_path}: {str(e)}")
    raise FileNotFoundError("No valid config file found (tried config.yml, config.yaml)")

# Global model state
config = load_config()
MODEL_REGISTRY = config['models']
MODEL_LOOKUP = {cfg['model_config_id']: cfg for cfg in MODEL_REGISTRY}
current_model_id = None
current_adapter = None
model_lock = threading.Lock()
request_queue = Queue()

def truncate_log(data: Any, base64_max_length: int = 50, other_max_length: int = 200, max_total: int = 1000) -> Any:
    """Recursively truncate long strings for logging."""
    if isinstance(data, str):
        max_length = base64_max_length if data.startswith("base64:") else other_max_length
        if len(data) > max_length:
            return f"{data[:max_length]}...[truncated, len={len(data)}]"
        return data
    elif isinstance(data, dict):
        return {k: truncate_log(v, base64_max_length, other_max_length, max_total) for k, v in data.items()}
    elif isinstance(data, list):
        return [truncate_log(item, base64_max_length, other_max_length, max_total) for item in data]
    elif isinstance(data, (int, float, bool, type(None))):
        return data
    else:
        str_data = str(data)
        if len(str_data) > other_max_length:
            return f"{str_data[:other_max_length]}...[truncated, type={type(data).__name__}]"
        return str_data

def load_model(model_config_id: str, timeout: float) -> bool:
    """Load a model using its adapter."""
    global current_model_id, current_adapter
    if model_config_id not in MODEL_LOOKUP:
        logger.error(f"Model config ID {model_config_id} not found in registry")
        return False

    cfg = MODEL_LOOKUP[model_config_id]
    with model_lock:
        vram_available = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        logger.info(f"Available VRAM: {vram_available / 1e9:.2f} GB")
        if vram_available < 5e9:
            logger.error("Insufficient VRAM for model loading")
            return False

        # Unload current model
        if current_adapter is not None:
            current_adapter.unload()
            current_adapter = None
            current_model_id = None

        # Load new model via adapter
        adapter = cfg['adapter_class'](cfg)
        if adapter.load(timeout):
            current_adapter = adapter
            current_model_id = model_config_id
            return True
        return False

class ModelRequestHandler(socketserver.BaseRequestHandler):
    """Handle client requests for model tasks."""
    def handle(self):
        try:
            # Receive data in chunks
            data = b""
            while True:
                chunk = self.request.recv(65536)
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

            # Handle requests
            response_data = {'model_config_id': current_model_id}
            with model_lock:
                task = request.get('task')
                if task != 'list_models' and 'model_config_id' not in request:
                    raise ValueError("model_config_id required for all tasks except list_models")
                
                model_config_id = request.get('model_config_id', None)

                if task == 'load_model':
                    if model_config_id not in MODEL_LOOKUP:
                        raise ValueError(f"Invalid model_config_id: {model_config_id}")
                    success = load_model(model_config_id, config['timeout'])
                    response_data.update({
                        'status': 'success' if success else 'error',
                        'message': f"Model {model_config_id} {'loaded' if success else 'failed to load'}",
                        'model_config_id': current_model_id
                    })
                elif task == 'unload_model':
                    if model_config_id != current_model_id:
                        raise ValueError(f"Cannot unload {model_config_id}; current model is {current_model_id or 'none'}")
                    if current_adapter is not None:
                        current_adapter.unload()
                        current_adapter = None
                        current_model_id = None
                    response_data.update({
                        'status': 'success',
                        'message': 'Model unloaded',
                        'model_config_id': None
                    })
                elif task == 'list_models':
                    current_cfg = MODEL_LOOKUP.get(current_model_id, {}) if current_model_id else {}
                    current_cfg = {k: str(v) for k, v in current_cfg.items() if k != 'local_path' and k != 'adapter_class'}
                    response_data.update({
                        'status': 'success',
                        'models': [cfg['model_config_id'] for cfg in MODEL_REGISTRY],
                        'current_model': current_model_id,
                        'current_config': current_cfg
                    })
                elif task == 'prompt':
                    if model_config_id not in MODEL_LOOKUP:
                        raise ValueError(f"Invalid model_config_id: {model_config_id}")
                    if current_model_id is None or current_adapter is None or current_model_id != model_config_id:
                        request_queue.put((self.request, request))
                        response_data.update({
                            'status': 'pending',
                            'message': f"Model {model_config_id} not loaded; request queued",
                            'model_config_id': current_model_id
                        })
                    else:
                        response_data.update(current_adapter.handle_request(request))

                else:
                    raise ValueError(f"Invalid task: {task}")

            response_data['model_config_id'] = current_model_id
            self.request.sendall(json.dumps(response_data).encode('utf-8'))
            logger.info("Response sent")

            # Process queued requests after model load
            if task == 'load_model' and response_data['status'] == 'success':
                while not request_queue.empty():
                    client_request, queued_request = request_queue.get()
                    handler = ModelRequestHandler(client_request, client_request.getsockname(), self.server)
                    handler.handle()

        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            error_response = {'error': str(e), 'model_config_id': current_model_id}
            self.request.sendall(json.dumps(error_response).encode('utf-8'))

def run_server(startup_model_id: str = None):
    """Run the TCP server with startup model."""
    # Load model at startup if specified
    global current_model_id
    if startup_model_id and startup_model_id in MODEL_LOOKUP:
        logger.info(f"Attempting to load startup model: {startup_model_id}")
        if not load_model(startup_model_id, config['timeout']):
            logger.warning(f"Failed to load startup model {startup_model_id}; falling back to index 0")
            load_model(MODEL_REGISTRY[0]['model_config_id'], config['timeout'])
    else:
        logger.info(f"Loading default model at index 0: {MODEL_REGISTRY[0]['model_config_id']}")
        load_model(MODEL_REGISTRY[0]['model_config_id'], config['timeout'])

    server = socketserver.ThreadingTCPServer((config['host'], config['port']), ModelRequestHandler)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    logger.info(f"Server running on {config['host']}:{config['port']}")
    server.serve_forever()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Server")
    parser.add_argument('--model-config-id', type=str, help="Model config ID to load at startup")
    args = parser.parse_args()
    
    import socket
    run_server(startup_model_id=args.model_config_id)