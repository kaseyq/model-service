# src/common/socket_utils.py
import socket
import json
import logging
from typing import Dict, Any
from src.common.config_utils import load_config

logger = logging.getLogger(__name__)

class ModelServiceClient:
    def __init__(self, host: str, port: int, timeout: int = None):
        self.host = host
        self.port = port
        config = load_config()
        self.timeout = timeout if timeout is not None else config.get('client_socket_timeout', 90)
    
    def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout)
                sock.connect((self.host, self.port))
                logger.debug(f"Sending request to model-service: {json.dumps(request)}")
                sock.sendall(json.dumps(request).encode('utf-8') + b'\n')
                data = ''
                while True:
                    chunk = sock.recv(4096).decode('utf-8')
                    if not chunk:
                        break
                    data += chunk
                    if '\n' in data:
                        break
                logger.debug(f"Received raw response from model-service: {data}")
                response = json.loads(data.strip())
                logger.debug(f"Parsed response: {response}")
                return response
        except socket.timeout:
            logger.error("Socket communication timed out")
            return {"status": "error", "message": "Request timed out", "payload": {}}
        except Exception as e:
            logger.error(f"Socket communication error: {str(e)}")
            return {"status": "error", "message": str(e), "payload": {}}