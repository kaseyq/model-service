import logging
from typing import Dict, Any, List, Callable
import base64
import io
from PIL import Image
import numpy as np
import librosa
from fastapi import UploadFile
from .path_utils import PathUtil
import os
import yaml
import tempfile

logger = logging.getLogger(__name__)

class DataProcessorRegistry:
    PROCESSORS: Dict[str, Callable] = {}

    @classmethod
    def register_processor(cls, data_type: str, processor: Callable):
        cls.PROCESSORS[data_type] = processor

    @classmethod
    async def process(cls, data_type: str, data: Any) -> bytes:
        processor = cls.PROCESSORS.get(data_type)
        if not processor:
            raise ValueError(f"No processor for data type: {data_type}")
        return await processor(data)

async def process_audio(file: UploadFile) -> bytes:
    audio, sr = librosa.load(file.file, sr=16000, mono=True)
    return audio.astype(np.float32).tobytes()

async def process_image(file: UploadFile) -> bytes:
    image = Image.open(file.file).convert('RGB')
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG")
    return output.getvalue()

async def process_text(data: str) -> bytes:
    return data.encode('utf-8')

DataProcessorRegistry.register_processor('audio/wav', process_audio)
DataProcessorRegistry.register_processor('image/jpeg', process_image)
DataProcessorRegistry.register_processor('text/plain', process_text)

async def process_audio_file(file: UploadFile) -> bytes:
    """Process audio file to mono, 16000 Hz, and return as bytes."""
    try:
        if not file.filename.lower().endswith('.wav'):
            raise ValueError("Only WAV files are supported for audio processing")
        temp_dir = PathUtil.get_path("tmp")
        os.makedirs(temp_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_dir, delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name
        audio, sr = librosa.load(temp_file_path, sr=16000, mono=True)
        audio_data = audio.astype(np.float32).tobytes()
        os.unlink(temp_file_path)
        logger.info(f"Processed audio: size={len(audio_data)}, sample_rate={sr}")
        return audio_data
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        raise

async def process_image_file(file: UploadFile) -> bytes:
    """Process image to max 1024x1024 and return as bytes."""
    try:
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise ValueError("Only PNG or JPEG files are supported for image processing")
        image = Image.open(io.BytesIO(await file.read()))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        max_size = 1024
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG")
        image_data = output.getvalue()
        logger.info(f"Processed image: size={len(image_data)}, dimensions={image.size}")
        return image_data
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise

def decode_base64_data(data: str) -> bytes:
    """Decode base64-encoded data, removing 'base64:' prefix if present."""
    try:
        if data.startswith("base64:"):
            data = data[7:]
        return base64.b64decode(data)
    except Exception as e:
        raise ValueError(f"Invalid base64 data: {str(e)}")

def encode_base64_data(data: bytes) -> str:
    """Encode data as base64 with 'base64:' prefix."""
    try:
        encoded = base64.b64encode(data).decode('utf-8')
        return f"base64:{encoded}"
    except Exception as e:
        raise ValueError(f"Failed to encode base64 data: {str(e)}")

def load_yaml(config_paths: List[str]) -> Dict[str, Any]:
    """Load configuration from YAML file, trying multiple paths."""
    for config_path in config_paths:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            if not config or 'models' not in config:
                raise ValueError(f"Invalid {config_path}: 'models' key missing")
            return config
        except FileNotFoundError:
            continue
        except Exception as e:
            raise ValueError(f"Failed to load {config_path}: {str(e)}")
    raise FileNotFoundError("No valid config file found")