import logging
import json
from typing import Any

def setup_logging():
    """Set up logging with timestamps."""
    logging.basicConfig(
        level=logging.INFO,
        filename="storage/logs/model_server.log",
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def truncate_log(data: Any, base64_max_length: int = 100, other_max_length: int = 200, max_total: int = 1000) -> str:
    """Truncate log data to prevent excessive log size."""
    if isinstance(data, dict):
        data = json.dumps(data)
    if isinstance(data, bytes):
        data = data.decode('utf-8', errors='ignore')
    if not isinstance(data, str):
        data = str(data)
    
    if data.startswith("base64:"):
        return data[:base64_max_length] + "..." if len(data) > base64_max_length else data
    return data[:other_max_length] + "..." if len(data) > other_max_length else data