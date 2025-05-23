import logging
import sys
import torch
from src.model_manager import ModelManager
import time
#, MODEL_LOOKUP, current_model_id, current_adapter, server_state, busy_reason, model_lock

logger = logging.getLogger(__name__)
server_start_time = time.time()

def command_line_interface():
    """Listen for command-line input to control the server."""
    #global server_state, busy_reason, API_VERSION, config

    if not sys.stdin.isatty():
        logger.info("CLI disabled: not running in a terminal")
        return
    print("Model-Service Command Line Interface")
    print("Commands: load <model_config_id>, unload, list, status, exit")
    while True:
        try:
            cmd = input("> ").strip().lower()
            if cmd == "exit":
                logger.info("Shutting down server via CLI")
                sys.exit(0)
            elif cmd == "list":
                print(f"Available models: {[cfg['model_config_id'] for cfg in ModelManager.MODEL_LOOKUP.values()]}")
                #print(f"Available models: {[cfg['model_config_id'] for cfg in ModelManager.MODEL_LOOKUP]}")
                print(f"Current model: {ModelManager.current_model_id or 'none'}")
            elif cmd == "unload":
                success, msg = ModelManager.unload()
                
                if success == True:
                    logger.info(f"unload {success} {msg}")
                else:
                    logger.error(f"unload {success} {msg}")

                print(f"unload: {success} {msg}")

            elif cmd.startswith("load "):
                model_id = cmd[5:].strip()
                success, msg = ModelManager.load(model_id, ModelManager.config['timeout'])

                if success == True:
                    logger.info(f"load {success} {msg}")
                else:
                    logger.error(f"load {success} {msg}")

                print(f"unload: {success} {msg}")

            elif cmd == "status":
                cfg = ModelManager.MODEL_LOOKUP.get(ModelManager.current_model_id, {}) if ModelManager.current_model_id else {'device': 'cuda:0'}
                device_index = int(cfg['device'].split(':')[-1]) if 'cuda' in cfg['device'] else 0
                try:
                    vram_total = torch.cuda.get_device_properties(device_index).total_memory / 1e9
                    vram_used = torch.cuda.memory_allocated(device_index) / 1e9
                except Exception as e:
                    logger.error(f"Failed to retrieve VRAM status: {str(e)}", exc_info=True)
                    vram_total = vram_used = 0.0
                uptime = time.time() - ModelManager.server_start_time
                print(f"Health: {ModelManager.server_state}")
                if ModelManager.server_state == "busy":
                    print(f"Busy reason: {ModelManager.busy_reason or 'unknown'}")
                print(f"Current model: {ModelManager.current_model_id or 'none'}")
                print(f"VRAM: {round(vram_used, 2)} GB used / {round(vram_total, 2)} GB total")
                print(f"Uptime: {round(uptime, 2)} seconds")
            else:
                print("Unknown command. Use: load <model_config_id>, unload, list, status, exit")
        except KeyboardInterrupt:
            logger.info("CLI interrupted, continuing")
        except Exception as e:
            logger.error(f"CLI error: {str(e)}", exc_info=True)
            print(f"Error: {str(e)}")