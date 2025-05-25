import unittest
import asyncio
from unittest.mock import patch, AsyncMock, MagicMock
from src.common.file_utils import DataProcessorRegistry, process_audio, process_image, process_text, process_audio_file, process_image_file, decode_base64_data, encode_base64_data, load_yaml
from fastapi import UploadFile
import io
import os

class TestFileUtils(unittest.IsolatedAsyncioTestCase):
    def test_data_processor_registry(self):
        registry = DataProcessorRegistry()
        async def dummy_processor(data):
            return b"test"
        registry.register_processor("test/type", dummy_processor)
        self.assertIn("test/type", registry.PROCESSORS)
        result = asyncio.run(registry.process("test/type", None))
        self.assertEqual(result, b"test")
        with self.assertRaises(ValueError):
            asyncio.run(registry.process("invalid/type", None))

    @patch('librosa.load')
    async def test_process_audio(self, mock_load):
        mock_load.return_value = (MagicMock(), 16000)
        file = MagicMock(spec=UploadFile)
        file.file = io.BytesIO()
        result = await process_audio(file)
        self.assertIsInstance(result, bytes)
        mock_load.assert_called_with(file.file, sr=16000, mono=True)

    @patch('librosa.load', side_effect=Exception("Load error"))
    async def test_process_audio_error(self, mock_load):
        file = MagicMock(spec=UploadFile)
        file.file = io.BytesIO()
        with self.assertRaises(Exception):
            await process_audio(file)

    @patch('PIL.Image.open')
    async def test_process_image(self, mock_open):
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_image.thumbnail = MagicMock()
        mock_open.return_value = mock_image
        file = MagicMock(spec=UploadFile)
        file.file = io.BytesIO()
        result = await process_image(file)
        self.assertIsInstance(result, bytes)
        mock_open.assert_called()
        mock_image.convert.assert_called_with('RGB')

    @patch('PIL.Image.open', side_effect=Exception("Image error"))
    async def test_process_image_error(self, mock_open):
        file = MagicMock(spec=UploadFile)
        file.file = io.BytesIO()
        with self.assertRaises(Exception):
            await process_image(file)

    async def test_process_text(self):
        result = await process_text("test")
        self.assertEqual(result, b"test")
        self.assertIsInstance(result, bytes)

    @patch('librosa.load')
    @patch('os.makedirs')
    @patch('tempfile.NamedTemporaryFile')
    async def test_process_audio_file(self, mock_tempfile, mock_makedirs, mock_load):
        mock_load.return_value = (MagicMock(), 16000)
        mock_file = MagicMock()
        mock_file.write = MagicMock()
        mock_tempfile.return_value.__enter__.return_value = mock_file
        mock_file.name = "/tmp/test.wav"
        file = MagicMock(spec=UploadFile)
        file.filename = "test.wav"
        file.read = AsyncMock(return_value=b"audio data")
        result = await process_audio_file(file)
        self.assertIsInstance(result, bytes)
        mock_load.assert_called_with(mock_file.name, sr=16000, mono=True)

    async def test_process_audio_file_invalid_type(self):
        file = MagicMock(spec=UploadFile)
        file.filename = "test.mp3"
        file.read = AsyncMock(return_value=b"audio data")
        with self.assertRaises(ValueError) as cm:
            await process_audio_file(file)
        self.assertIn("Only WAV files", str(cm.exception))

    async def test_process_audio_file_io_error(self):
        file = MagicMock(spec=UploadFile)
        file.filename = "test.wav"
        file.read = AsyncMock(side_effect=IOError("File error"))
        with self.assertRaises(Exception) as cm:
            await process_audio_file(file)
        self.assertIn("Audio processing error", str(cm.exception))

    @patch('PIL.Image.open')
    @patch('os.makedirs')
    async def test_process_image_file(self, mock_makedirs, mock_open):
        mock_image = MagicMock()
        mock_image.convert.return_value = mock_image
        mock_image.thumbnail = MagicMock()
        mock_open.return_value = mock_image
        file = MagicMock(spec=UploadFile)
        file.filename = "test.jpg"
        file.read = AsyncMock(return_value=b"image data")
        result = await process_image_file(file)
        self.assertIsInstance(result, bytes)
        mock_open.assert_called()

    async def test_process_image_file_invalid_type(self):
        file = MagicMock(spec=UploadFile)
        file.filename = "test.gif"
        file.read = AsyncMock(return_value=b"image data")
        with self.assertRaises(ValueError) as cm:
            await process_image_file(file)
        self.assertIn("Only PNG or JPEG", str(cm.exception))

    async def test_process_image_file_io_error(self):
        file = MagicMock(spec=UploadFile)
        file.filename = "test.jpg"
        file.read = AsyncMock(side_effect=IOError("File error"))
        with self.assertRaises(Exception) as cm:
            await process_image_file(file)
        self.assertIn("Image processing error", str(cm.exception))

    def test_decode_base64_data(self):
        encoded = "base64:dGVzdA=="
        result = decode_base64_data(encoded)
        self.assertEqual(result, b"test")
        with self.assertRaises(ValueError):
            decode_base64_data("invalid")

    def test_encode_base64_data(self):
        data = b"test"
        result = encode_base64_data(data)
        self.assertEqual(result, "base64:dGVzdA==")
        with self.assertRaises(ValueError):
            encode_base64_data(None)

    @patch('builtins.open')
    def test_load_yaml(self, mock_open):
        mock_open.return_value.__enter__.return_value.read.return_value = """
        host: "0.0.0.0"
        models: []
        """
        config = load_yaml(["config.yml"])
        self.assertEqual(config["host"], "0.0.0.0")
        with self.assertRaises(FileNotFoundError):
            load_yaml(["nonexistent.yml"])

    @patch('builtins.open')
    def test_load_yaml_invalid(self, mock_open):
        mock_open.return_value.__enter__.return_value.read.return_value = """
        host: "0.0.0.0"
        """
        with self.assertRaises(ValueError) as cm:
            load_yaml(["config.yml"])
        self.assertIn("'models' key missing", str(cm.exception))

if __name__ == "__main__":
    unittest.main()