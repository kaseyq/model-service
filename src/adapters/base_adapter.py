import logging
import torch
import gc
import subprocess
import pickle
import os
from abc import ABC, abstractmethod
from typing import Dict, Any
from transformers import PreTrainedModel, PreTrainedTokenizer
import time

logger = logging.getLogger(__name__)

class ModelServiceAdapter(ABC):
    """Base adapter class for model services."""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model: PreTrainedModel = None
        self.tokenizer: PreTrainedTokenizer = None

    @abstractmethod
    def load(self, timeout: float = 1200.0) -> bool:
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a request to the model."""
        pass

    def unload(self) -> None:
        """Unload the model and tokenizer, freeing CUDA memory."""
        device_index = int(self.config['device'].split(':')[-1]) if 'cuda' in self.config['device'] else 0
        logger.info(f"Clearing model and tokenizer for device {device_index}...")
        self.model = None
        self.tokenizer = None
        logger.info("Forcing CUDA memory cleanup...")
        for _ in range(5):  # Aggressive cleanup
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated(device_index)
            torch.cuda.reset_max_memory_cached(device_index)
            gc.collect()
            torch.cuda.synchronize(device_index)
        vram_used = torch.cuda.memory_allocated(device_index) / 1e9
        logger.info(f"Model unloaded, VRAM used: {vram_used:.2f} GB")
        logger.info(f"CUDA memory summary after unloading:\n{torch.cuda.memory_summary(device_index)}")
        try:
            logger.info(f"GPU utilization after unloading: {torch.cuda.utilization(device_index)}%")
        except Exception as e:
            logger.warning(f"Failed to retrieve GPU utilization after unloading: {str(e)}")

    def _check_vram(self, device_index: int) -> tuple[float, float, float]:
        """Check CUDA device VRAM status."""
        torch.cuda.init()
        torch.cuda.synchronize(device_index)
        vram_total = torch.cuda.get_device_properties(device_index).total_memory / 1e9
        vram_used = torch.cuda.memory_allocated(device_index) / 1e9
        vram_available = vram_total - vram_used
        logger.info(f"VRAM status: {vram_used:.2f}/{vram_total:.2f} GB used, {vram_available:.2f} GB available")
        logger.info(f"CUDA memory summary:\n{torch.cuda.memory_summary(device_index)}")
        try:
            logger.info(f"GPU utilization: {torch.cuda.utilization(device_index)}%")
        except Exception as e:
            logger.warning(f"Failed to retrieve GPU utilization: {str(e)}")
        return vram_total, vram_used, vram_available

    def _load_model_in_subprocess(self, model_class: str, tokenizer_class: str, device_index: int, timeout: float) -> bool:
        """Load model and tokenizer in a subprocess to isolate CUDA context."""
        local_exists = os.path.exists(self.config['local_path'])
        logger.info("Starting model load in subprocess...")
        cmd = [
            "python3", "-c",
            f"""
import pickle
import torch
from transformers import {model_class}, {tokenizer_class}
torch.cuda.init()
torch.cuda.set_per_process_memory_fraction(0.9, {device_index})
print("Loading model...")
model = {model_class}.from_pretrained(
    '{self.config['local_path']}' if {local_exists} else '{self.config['model_name']}',
    load_in_4bit={self.config.get('load_in_4bit', False)} if '{model_class}' == 'AutoModelForCausalLM' else False,
    device_map={{'': 0}} if '{model_class}' == 'AutoModelForCausalLM' else None,
    trust_remote_code=True if '{model_class}' == 'AutoModel' else False,
    attn_implementation='sdpa' if '{model_class}' == 'AutoModel' else None,
    torch_dtype=torch.float16 if '{model_class}' == 'AutoModel' else None,
    local_files_only={local_exists}
)
if '{model_class}' == 'AutoModel':
    print("Model loaded, initializing TTS...")
    model.init_tts()
    print("TTS initialized, moving model to device...")
    model = model.eval().to('cuda:{device_index}')
print("Model loaded, loading tokenizer...")
tokenizer = {tokenizer_class}.from_pretrained(
    '{self.config['local_path']}' if {local_exists} else '{self.config['model_name']}',
    trust_remote_code=True if '{tokenizer_class}' == 'AutoTokenizer' else False,
    local_files_only={local_exists}
)
print("Tokenizer loaded")
torch.cuda.synchronize({device_index})
with open('/tmp/model.pkl', 'wb') as f:
    pickle.dump((model, tokenizer), f)
"""
        ]
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Monitor subprocess progress
        elapsed = 0
        while process.poll() is None:
            if elapsed >= timeout:
                process.terminate()
                logger.error(f"Subprocess timed out after {timeout} seconds")
                return False
            time.sleep(30)
            elapsed += 30
            logger.info(f"Subprocess still running, elapsed time: {elapsed} seconds")
            try:
                logger.info(f"GPU utilization: {torch.cuda.utilization(device_index)}%")
            except Exception as e:
                logger.warning(f"Failed to retrieve GPU utilization: {str(e)}")
            vram_used = torch.cuda.memory_allocated(device_index) / 1e9
            logger.info(f"Current VRAM usage: {vram_used:.2f} GB")
            try:
                stdout_line = process.stdout.readline().strip()
                if stdout_line:
                    logger.info(f"Subprocess stdout: {stdout_line}")
            except:
                pass
            try:
                stderr_line = process.stderr.readline().strip()
                if stderr_line:
                    logger.error(f"Subprocess stderr: {stderr_line}")
            except:
                pass

        stdout, stderr = process.communicate()
        if process.returncode != 0:
            logger.error(f"Subprocess failed: stdout: {stdout}, stderr: {stderr}")
            return False

        # Load model and tokenizer from pickle
        logger.info("Loading model and tokenizer from subprocess output...")
        with open('/tmp/model.pkl', 'rb') as f:
            self.model, self.tokenizer = pickle.load(f)
        os.remove('/tmp/model.pkl')
        logger.info("Model and tokenizer loaded from subprocess")
        return True

    def _save_model_locally(self):
        """Save model and tokenizer to local path if not already present."""
        if not os.path.exists(self.config['local_path']):
            logger.info("Saving model to local path...")
            self.model.save_pretrained(self.config['local_path'])
            self.tokenizer.save_pretrained(self.config['local_path'])
            logger.info("Model and tokenizer saved")