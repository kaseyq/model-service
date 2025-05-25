# src/command_line_interface.py
import logging
import sys
import torch
from src.model_manager import ModelManager
from src.common.socket_utils import ModelServiceClient
from src.common.config_utils import load_config
import time
import json

logger = logging.getLogger(__name__)
server_start_time = time.time()

def command_line_interface():
    """Listen for command-line input to control the server."""
    config = load_config()
    client = ModelServiceClient(host=config['host'], port=config['port'])
    
    if not sys.stdin.isatty():
        logger.info("CLI disabled: not running in a terminal")
        return
    print("Model-Service Command Line Interface")
    print("Commands: load <model_config_id>, unload, list, status, prompt <input>, exit")
    print("You can also enter a prompt directly if a model is loaded (e.g., 'write code...')")
    while True:
        try:
            cmd = input("> ").strip()
            cmd_lower = cmd.lower()
            if cmd_lower == "exit":
                logger.info("Shutting down server via CLI")
                sys.exit(0)
            elif cmd_lower == "list":
                print(f"Available models: {[cfg['model_config_id'] for cfg in ModelManager.MODEL_LOOKUP.values()]}")
                print(f"Current model: {ModelManager.current_model_id or 'none'}")
            elif cmd_lower == "unload":
                success, msg = ModelManager.unload()
                if success:
                    logger.info(f"unload {success} {msg}")
                else:
                    logger.error(f"unload {success} {msg}")
                print(f"unload: {success} {msg}")
            elif cmd_lower.startswith("load "):
                model_id = cmd_lower[5:].strip()
                success, msg = ModelManager.load(model_id, ModelManager.config['timeout'])
                if success:
                    logger.info(f"load {success} {msg}")
                else:
                    logger.error(f"load {success} {msg}")
                print(f"load: {success} {msg}")
            elif cmd_lower.startswith("prompt "):
                if not ModelManager.current_model_id:
                    print("Error: No model loaded. Use 'load <model_config_id>' first.")
                    logger.error("Prompt command failed: No model loaded")
                    continue
                prompt_input = cmd[7:].strip()
                if not prompt_input:
                    print("Error: Prompt input required")
                    logger.error("Prompt command failed: No input provided")
                    continue
                request = {
                    "version": ModelManager.API_VERSION,
                    "task": "prompt",
                    "model_config_id": ModelManager.current_model_id
                }
                if ModelManager.current_model_id == "codellama-13b":
                    request.update({
                        "input": prompt_input,
                        "params": {
                            "max_length": 300,  # Reduced for concise output
                            "temperature": 0.5,  # Lowered for focused output
                            "top_p": 0.9
                        }
                    })
                elif ModelManager.current_model_id == "minicpm-o_2_6":
                    request.update({
                        "messages": [
                            {
                                "role": "user",
                                "content": [prompt_input]
                            }
                        ],
                        "params": {}
                    })
                else:
                    print(f"Error: Prompt not supported for model {ModelManager.current_model_id}")
                    logger.error(f"Prompt not supported for model {ModelManager.current_model_id}")
                    continue
                response = client.send_request(request)
                if response.get("status") == "okay":
                    result = response.get("payload", {}).get("result", {})
                    if ModelManager.current_model_id == "codellama-13b":
                        text = result.get("text", "No text returned")
                        max_display_length = 2000  # Limit to avoid terminal overload
                        print(f"Response: {text[:max_display_length]}{'...' if len(text) > max_display_length else ''}")
                        logger.info(f"Prompt response: {text[:200]}...")
                    elif ModelManager.current_model_id == "minicpm-o_2_6":
                        audio_data = result.get("files", {}).get("output_audio_path", {}).get("data", "")
                        if audio_data.startswith("base64:"):
                            print("Response: Audio output generated (base64 encoded).")
                            logger.info(f"Prompt response: Audio output (base64 length: {len(audio_data)})")
                        else:
                            print("Response: No audio output generated")
                            logger.info("Prompt response: No audio output")
                    else:
                        print("Response: Unknown response format")
                        logger.info(f"Prompt response: Unknown format {result}")
                else:
                    error_msg = response.get("message", "Unknown error")
                    print(f"Error: {error_msg}")
                    logger.error(f"Prompt failed: {error_msg}")
            elif cmd_lower == "status":
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
                if ModelManager.current_model_id:
                    prompt_input = cmd.strip()
                    request = {
                        "version": ModelManager.API_VERSION,
                        "task": "prompt",
                        "model_config_id": ModelManager.current_model_id
                    }
                    if ModelManager.current_model_id == "codellama-13b":
                        request.update({
                            "input": prompt_input,
                            "params": {
                                "max_length": 300,  # Reduced for concise output
                                "temperature": 0.5,  # Lowered for focused output
                                "top_p": 0.9
                            }
                        })
                    elif ModelManager.current_model_id == "minicpm-o_2_6":
                        request.update({
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [prompt_input]
                                }
                            ],
                            "params": {}
                        })
                    else:
                        print(f"Error: Prompt not supported for model {ModelManager.current_model_id}")
                        logger.error(f"Prompt not supported for model {ModelManager.current_model_id}")
                        continue
                    response = client.send_request(request)
                    if response.get("status") == "okay":
                        result = response.get("payload", {}).get("result", {})
                        if ModelManager.current_model_id == "codellama-13b":
                            text = result.get("text", "No text returned")
                            max_display_length = 2000  # Limit to avoid terminal overload
                            print(f"Response: {text[:max_display_length]}{'...' if len(text) > max_display_length else ''}")
                            logger.info(f"Prompt response: {text[:200]}...")
                        elif ModelManager.current_model_id == "minicpm-o_2_6":
                            audio_data = result.get("files", {}).get("output_audio_path", {}).get("data", "")
                            if audio_data.startswith("base64:"):
                                print("Response: Audio output generated (base64 encoded).")
                                logger.info(f"Prompt response: Audio output (base64 length: {len(audio_data)})")
                            else:
                                print("Response: No audio output generated")
                                logger.info("Prompt response: No audio output")
                        else:
                            print("Response: Unknown response format")
                            logger.info(f"Prompt response: Unknown format {result}")
                    else:
                        error_msg = response.get("message", "Unknown error")
                        print(f"Error: {error_msg}")
                        logger.error(f"Prompt failed: {error_msg}")
                else:
                    print("Unknown command. Use: load <model_config_id>, unload, list, status, prompt <input>, exit")
                    logger.error(f"Unknown command: {cmd}")
        except KeyboardInterrupt:
            logger.info("CLI interrupted, continuing")
        except Exception as e:
            logger.error(f"CLI error: {str(e)}", exc_info=True)
            print(f"Error: {str(e)}")