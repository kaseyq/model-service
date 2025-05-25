import unittest
import asyncio
import torch
import os
from unittest.mock import patch, MagicMock, AsyncMock
from src.adapters.minicpmo_model_adapter import MiniCPMoModelAdapter
from src.adapters.causal_lm_adapter import CausalLMAdapter
from src.adapters.clip_vision_adapter import CLIPVisionAdapter
from src.common.file_utils import load_yaml
from src.common.path_utils import PathUtil

class TestAdapters(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = load_yaml([PathUtil.get_path("config.yml")])
        self.models = {m['model_config_id']: m for m in self.config['models']}
        # Mock torch.cuda to simulate CUDA availability
        self.patcher_cuda = patch('torch.cuda.is_available', return_value=True)
        self.patcher_cuda.start()
        # Mock transformers to prevent model downloads
        self.patcher_model = patch('transformers.AutoModel.from_pretrained', return_value=MagicMock())
        self.patcher_model.start()
        self.patcher_model_causal = patch('transformers.AutoModelForCausalLM.from_pretrained', return_value=MagicMock())
        self.patcher_model_causal.start()
        self.patcher_model_clip = patch('transformers.CLIPModel.from_pretrained', return_value=MagicMock())
        self.patcher_model_clip.start()
        self.patcher_tokenizer = patch('transformers.AutoTokenizer.from_pretrained', return_value=MagicMock())
        self.patcher_tokenizer.start()
        self.patcher_processor = patch('transformers.CLIPProcessor.from_pretrained', return_value=MagicMock())
        self.patcher_processor.start()

    def tearDown(self):
        self.patcher_cuda.stop()
        self.patcher_model.stop()
        self.patcher_model_causal.stop()
        self.patcher_model_clip.stop()
        self.patcher_tokenizer.stop()
        self.patcher_processor.stop()

    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.set_device')
    @patch('torch.cuda.synchronize')
    async def test_minicpm_load_success(self, mock_sync, mock_set_device, mock_memory, mock_properties):
        mock_device = MagicMock()
        mock_device.total_memory = 20 * 10**9
        mock_device.major = 8
        mock_device.minor = 0
        mock_properties.return_value = mock_device
        mock_memory.return_value = 5 * 10**9

        config = self.models['minicpm-o_2_6']
        adapter = MiniCPMoModelAdapter(config)
        with patch('os.path.exists', return_value=True), \
             patch('src.adapters.base_adapter.ModelServiceAdapter._save_model_locally', new=AsyncMock()):
            adapter.model.init_tts = MagicMock()
            success = await adapter.load(timeout=10.0)
            self.assertTrue(success)
            adapter.model.init_tts.assert_called()
            mock_sync.assert_called()

    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.set_device')
    async def test_minicpm_load_insufficient_vram(self, mock_set_device, mock_memory, mock_properties):
        mock_device = MagicMock()
        mock_device.total_memory = 5 * 10**9
        mock_device.major = 8
        mock_device.minor = 0
        mock_properties.return_value = mock_device
        mock_memory.return_value = 4 * 10**9

        config = self.models['minicpm-o_2_6']
        adapter = MiniCPMoModelAdapter(config)
        success = await adapter.load(timeout=10.0)
        self.assertFalse(success)

    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.set_device')
    async def test_minicpm_load_no_local_model(self, mock_set_device, mock_memory, mock_properties):
        mock_device = MagicMock()
        mock_device.total_memory = 20 * 10**9
        mock_device.major = 8
        mock_device.minor = 0
        mock_properties.return_value = mock_device
        mock_memory.return_value = 5 * 10**9

        config = self.models['minicpm-o_2_6']
        adapter = MiniCPMoModelAdapter(config)
        with patch('os.path.exists', return_value=False):
            success = await adapter.load(timeout=10.0)
            self.assertFalse(success)

    async def test_minicpm_handle_request_success(self):
        config = self.models['minicpm-o_2_6']
        adapter = MiniCPMoModelAdapter(config)
        adapter.model = MagicMock()
        adapter.tokenizer = MagicMock()
        with patch('tempfile.TemporaryDirectory') as mock_tempdir:
            mock_tempdir.return_value.__enter__.return_value = '/tmp'
            adapter.model.chat = MagicMock(return_value=None)
            with patch('builtins.open', new_callable=MagicMock) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = b'test_audio'
                request = {
                    "messages": [{"role": "user", "content": ["test"]}],
                    "params": {},
                    "input_type": "text/plain"
                }
                response = await adapter.handle_request(request)
                self.assertEqual(response["status"], "okay")
                self.assertIn("files", response["result"])
                self.assertTrue(response["result"]["files"]["output_audio_path"]["data"].startswith("base64:"))

    async def test_minicpm_handle_request_invalid_input(self):
        config = self.models['minicpm-o_2_6']
        adapter = MiniCPMoModelAdapter(config)
        adapter.model = MagicMock()
        adapter.tokenizer = MagicMock()
        request = {
            "messages": [],
            "params": {},
            "input_type": "text/plain"
        }
        response = await adapter.handle_request(request)
        self.assertEqual(response["status"], "error")
        self.assertIn("Messages required", response["message"])

    async def test_minicpm_handle_request_invalid_base64(self):
        config = self.models['minicpm-o_2_6']
        adapter = MiniCPMoModelAdapter(config)
        adapter.model = MagicMock()
        adapter.tokenizer = MagicMock()
        request = {
            "messages": [{"role": "user", "content": ["base64:invalid"]}],
            "params": {},
            "input_type": "audio/wav"
        }
        response = await adapter.handle_request(request)
        self.assertEqual(response["status"], "error")
        self.assertIn("Invalid base64 data", response["message"])

    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.set_device')
    @patch('torch.cuda.synchronize')
    async def test_causallm_load_success(self, mock_sync, mock_set_device, mock_memory, mock_properties):
        mock_device = MagicMock()
        mock_device.total_memory = 20 * 10**9
        mock_device.major = 8
        mock_device.minor = 0
        mock_properties.return_value = mock_device
        mock_memory.return_value = 5 * 10**9

        config = self.models['codellama-13b']
        adapter = CausalLMAdapter(config)
        with patch('os.path.exists', return_value=True), \
             patch('src.adapters.base_adapter.ModelServiceAdapter._save_model_locally', new=AsyncMock()):
            adapter.model.named_parameters.return_value = [('param', MagicMock(device=torch.device('cuda:0')))]
            adapter.model.model.layers = [MagicMock() for _ in range(40)]
            for layer in adapter.model.model.layers:
                layer.self_attn.rotary_emb = MagicMock()
                layer.self_attn.rotary_emb.parameters.return_value = []
            success = await adapter.load(timeout=10.0)
            self.assertTrue(success)
            adapter.model.assert_called()
            adapter.tokenizer.assert_called()
            mock_sync.assert_called()

    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.set_device')
    async def test_causallm_load_no_local_model(self, mock_set_device, mock_memory, mock_properties):
        mock_device = MagicMock()
        mock_device.total_memory = 20 * 10**9
        mock_device.major = 8
        mock_device.minor = 0
        mock_properties.return_value = mock_device
        mock_memory.return_value = 5 * 10**9

        config = self.models['codellama-13b']
        adapter = CausalLMAdapter(config)
        with patch('os.path.exists', return_value=False):
            success = await adapter.load(timeout=10.0)
            self.assertFalse(success)

    async def test_causallm_handle_request_success(self):
        config = self.models['codellama-13b']
        adapter = CausalLMAdapter(config)
        adapter.model = MagicMock()
        adapter.tokenizer = MagicMock()
        adapter.tokenizer.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        adapter.model.get_submodule.return_value.weight.device = torch.device('cuda:0')
        adapter.model.generate = MagicMock(return_value=torch.tensor([[1, 2, 3, 4]]))
        adapter.tokenizer.decode = MagicMock(return_value="generated text")
        request = {
            "input": "test prompt",
            "params": {"max_length": 50},
            "input_type": "text/plain"
        }
        response = await adapter.handle_request(request)
        self.assertEqual(response["status"], "okay")
        self.assertIn("text", response["result"])
        self.assertEqual(response["result"]["text"], "generated text")

    async def test_causallm_handle_request_no_submodule(self):
        config = self.models['codellama-13b']
        adapter = CausalLMAdapter(config)
        adapter.model = MagicMock()
        adapter.tokenizer = MagicMock()
        adapter.model.get_submodule.side_effect = AttributeError("No submodule")
        request = {
            "input": "test prompt",
            "params": {},
            "input_type": "text/plain"
        }
        response = await adapter.handle_request(request)
        self.assertEqual(response["status"], "error")
        self.assertIn("does not support get_submodule", response["message"])

    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.set_device')
    @patch('torch.cuda.synchronize')
    async def test_clipvision_load_success(self, mock_sync, mock_set_device, mock_memory, mock_properties):
        mock_device = MagicMock()
        mock_device.total_memory = 20 * 10**9
        mock_device.major = 8
        mock_device.minor = 0
        mock_properties.return_value = mock_device
        mock_memory.return_value = 5 * 10**9

        config = self.models['clip-vision']
        adapter = CLIPVisionAdapter(config)
        with patch('os.path.exists', return_value=True):
            success = await adapter.load(timeout=10.0)
            self.assertTrue(success)
            adapter.model.assert_called()
            adapter.tokenizer.assert_called()
            mock_sync.assert_called()

    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.set_device')
    async def test_clipvision_load_failed(self, mock_set_device, mock_memory, mock_properties):
        mock_device = MagicMock()
        mock_device.total_memory = 20 * 10**9
        mock_device.major = 8
        mock_device.minor = 0
        mock_properties.return_value = mock_device
        mock_memory.return_value = 5 * 10**9

        config = self.models['clip-vision']
        adapter = CLIPVisionAdapter(config)
        with patch('transformers.CLIPModel.from_pretrained', side_effect=Exception("Load error")):
            success = await adapter.load(timeout=10.0)
            self.assertFalse(success)

    async def test_clipvision_handle_request_success(self):
        config = self.models['clip-vision']
        adapter = CLIPVisionAdapter(config)
        adapter.model = MagicMock()
        adapter.tokenizer = MagicMock()
        adapter.model.get_image_features = MagicMock(return_value=torch.tensor([[0.1, 0.2]]))
        with patch('PIL.Image.open', return_value=MagicMock()):
            request = {
                "data": "base64:iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=",
                "input_type": "image/jpeg"
            }
            response = await adapter.handle_request(request)
            self.assertEqual(response["status"], "okay")
            self.assertIn("features", response["result"])
            self.assertEqual(len(response["result"]["features"]), 2)

    async def test_clipvision_handle_request_invalid_base64(self):
        config = self.models['clip-vision']
        adapter = CLIPVisionAdapter(config)
        adapter.model = MagicMock()
        adapter.tokenizer = MagicMock()
        request = {
            "data": "invalid_base64",
            "input_type": "image/jpeg"
        }
        response = await adapter.handle_request(request)
        self.assertEqual(response["status"], "error")
        self.assertIn("Invalid base64 data", response["message"])

if __name__ == "__main__":
    unittest.main()