import socket
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class ModelServiceClient:
    def __init__(self, host: str, port: int, timeout: int):
        self.host = host
        self.port = port
        self.timeout = 30  # 30 seconds for faster failure on hangs

    def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        try:
            # Ensure version is sent as integer 1
            #if "version" in request:
            #    request["version"] = int(request["version"])
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.settimeout(self.timeout)
                sock.connect((self.host, self.port))
                # Log the exact request being sent
                logger.debug(f"Sending request to model-service: {json.dumps(request)}")
                # Send request as JSON
                sock.sendall(json.dumps(request).encode('utf-8') + b'\n')
                # Receive response
                data = ''
                while True:
                    chunk = sock.recv(4096).decode('utf-8')
                    if not chunk:
                        break
                    data += chunk
                    if '\n' in data:
                        break
                # Log the raw response
                logger.debug(f"Received raw response from model-service: {data}")
                # Parse response
                response = json.loads(data.strip())
                logger.debug(f"Parsed response: {response}")
                return response
        except socket.timeout:
            logger.error("Socket communication timed out")
            return {"status": "error", "message": "Request timed out", "payload": {}}
        except Exception as e:
            logger.error(f"Socket communication error: {str(e)}")
            return {"status": "error", "message": str(e), "payload": {}}
