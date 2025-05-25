import argparse
import socket
import socketserver
import threading
import logging
import asyncio
from fastapi import FastAPI, HTTPException
from src.common.path_utils import PathUtil
from src.common.logging_utils import setup_logging
from src.model_manager import ModelManager
from src.model_request_handler import ModelRequestHandler
from src.command_line_interface import command_line_interface
from src.common.file_utils import load_yaml
import uvicorn

setup_logging()
logger = logging.getLogger(__name__)
app = FastAPI()

@app.post("/api/v{version}/task")
async def handle_task(version: int, request: dict):
    if str(version) != str(ModelManager.API_VERSION):
        raise HTTPException(status_code=400, detail=f"Invalid version: expected {ModelManager.API_VERSION}")
    handler = ModelRequestHandler(None)  # Socket not used for HTTP
    handler.request_data = request  # Inject request data
    response = await handler.handle_async()  # Use async handle method
    return response

async def run_server(startup_model_id: str = None):
    config = load_yaml([PathUtil.get_path("config.yml"), PathUtil.get_path("config.yaml")])
    ModelManager.server_state = "busy"
    ModelManager.busy_reason = "initializing server"

    # Discover adapters dynamically
    ModelManager.discover_adapters()

    if startup_model_id:
        await ModelManager.load(startup_model_id, config['timeout'])
    else:
        await ModelManager.load(config['models'][0]['model_config_id'], config['timeout'])

    # Start CLI in a separate thread
    cli_thread = threading.Thread(target=command_line_interface, daemon=True)
    cli_thread.start()

    # Start TCP server in a separate thread
    tcp_server = socketserver.ThreadingTCPServer((config['host'], config['port']), ModelRequestHandler)
    tcp_server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcp_thread = threading.Thread(target=tcp_server.serve_forever, daemon=True)
    tcp_thread.start()

    # Start HTTP server in the current event loop
    uvicorn_config = uvicorn.Config(
        app=app,
        host=config['host'],
        port=config.get('http_port', 8000),
        log_level="info"
    )
    server = uvicorn.Server(uvicorn_config)
    await server.serve()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Server")
    parser.add_argument('--model-config-id', type=str, help="Model config ID to load at startup")
    args = parser.parse_args()
    asyncio.run(run_server(startup_model_id=args.model_config_id))