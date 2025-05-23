import logging
import tempfile
import numpy as np
import librosa
import soundfile as sf
from PIL import Image
import io
import ffmpeg
from fastapi import UploadFile
from .path_utils import PathUtil
import os
import yaml
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

async def process_audio(file: UploadFile) -> bytes:
    """Process audio file to mono, 16000 Hz, and return as base64-encoded np.ndarray."""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.wav'):
            raise ValueError("Only WAV files are supported for audio processing")

        # Save uploaded file to temporary location
        temp_dir = PathUtil.get_path("tmp")
        os.makedirs(temp_dir, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=temp_dir, delete=False) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        # Apply ffmpeg filters (highpass, lowpass)
        output_path = tempfile.mktemp(suffix=".wav", dir=temp_dir)
        stream = ffmpeg.input(temp_file_path)
        stream = ffmpeg.filter(stream, 'highpass', f=200)
        stream = ffmpeg.filter(stream, 'lowpass', f=3000)
        stream = ffmpeg.output(stream, output_path, ac=1, ar=16000)
        ffmpeg.run(stream, overwrite_output=True, quiet=True)

        # Load with librosa
        audio, sr = librosa.load(output_path, sr=16000, mono=True)
        audio_data = audio.astype(np.float32).tobytes()

        # Clean up
        os.unlink(temp_file_path)
        os.unlink(output_path)

        logger.info(f"Processed audio: size={len(audio_data)}, sample_rate={sr}")
        return audio_data
    except Exception as e:
        logger.error(f"Audio processing error: {str(e)}")
        raise

async def process_image(file: UploadFile) -> bytes:
    """Process image to max 1024x1024 and return as base64-encoded bytes."""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            raise ValueError("Only PNG or JPEG files are supported for image processing")

        image = Image.open(io.BytesIO(await file.read()))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Resize if necessary
        max_size = 1024
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        # Save to bytes
        output = io.BytesIO()
        image.save(output, format="JPEG")
        image_data = output.getvalue()

        logger.info(f"Processed image: size={len(image_data)}, dimensions={image.size}")
        return image_data
    except Exception as e:
        logger.error(f"Image processing error: {str(e)}")
        raise



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