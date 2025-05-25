import argparse
import logging
import socket
import socketserver
import threading

from src.common.path_utils import PathUtil
from src.common.logging_utils import setup_logging
#from src.model_manager import load_model, server_state, busy_reason
from src.model_request_handler import ModelRequestHandler
from src.model_manager import ModelManager
from src.command_line_interface import command_line_interface
from src.common.file_utils import load_yaml
import time

global server_state, busy_reason, server_start_time, API_VERSION, config

# Global server state
#config = load_yaml([PathUtil.get_path("config.yml"), PathUtil.get_path("config.yaml")])
#API_VERSION = config['api_version']
#server_state = "down"  # error, ready, busy
#busy_reason = None  # Track reason for busy state
#server_start_time = time.time()

# Set up logging with timestamps
setup_logging()



logger = logging.getLogger(__name__)

def run_server(startup_model_id: str = None):
    """Run the TCP server with startup model and CLI."""
    #global server_state, busy_reason, server_start_time, API_VERSION, config
    ModelManager.server_state = "busy"
    ModelManager.busy_reason = "initializing server"
    logger.info(f"Server state: {ModelManager.server_state} due to {ModelManager.busy_reason}")

    config = load_yaml([PathUtil.get_path("config.yml"), PathUtil.get_path("config.yaml")])

    if startup_model_id:
        logger.info(f"Attempting to load startup model: {startup_model_id}")
        ModelManager.load(startup_model_id, config['timeout'])

    #if startup_model_id and startup_model_id in config['models']:
        #if not ModelRequestHandler.load_model(startup_model_id, config['timeout']):
        #    logger.warning(f"Failed to load startup model {startup_model_id}; falling back to index 0")
        #ModelManager.load(config['models'][0]['model_config_id'], config['timeout'])
    else:
        logger.info(f"Loading default model at index 0: {config['models'][0]['model_config_id']}")
        ModelManager.load(config['models'][0]['model_config_id'], config['timeout'])

    # Start CLI in a separate thread
    cli_thread = threading.Thread(target=command_line_interface, daemon=True)
    cli_thread.start()

    try:
        server = socketserver.ThreadingTCPServer((config['host'], config['port']), ModelRequestHandler)
        server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_state = "ready"
        busy_reason = None
        logger.info(f"Server running on {config['host']}:{config['port']}")
        server.serve_forever()
    except Exception as e:
        logger.error(f"Server failed to start: {str(e)}", exc_info=True)
        server_state = "error"
        busy_reason = None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Server")
    parser.add_argument('--model-config-id', type=str, help="Model config ID to load at startup")
    args = parser.parse_args()
    
    run_server(startup_model_id=args.model_config_id)