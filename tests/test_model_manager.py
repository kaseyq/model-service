import unittest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from src.model_manager import ModelManager
from src.common.file_utils import load_yaml
from src.common.path_utils import PathUtil
import os

class TestModelManager(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = load_yaml([PathUtil.get_path("config.yml")])
        ModelManager.MODEL_REGISTRY = self.config['models']
        ModelManager.MODEL_LOOKUP = {cfg['model_config_id']: cfg for cfg in ModelManager.MODEL_REGISTRY}
        ModelManager.current_model_id = None
        ModelManager.current_adapter = None
        ModelManager.server_state = "ready"
        ModelManager.busy_reason = None
        ModelManager.config = self.config

    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    @patch('torch.cuda.set_device')
    @patch('torch.cuda.synchronize')
    @patch('torch.cuda.is_available', return_value=True)
    async def test_load_model_success(self, mock_cuda, mock_sync, mock_set_device, mock_memory, mock_properties):
        mock_device = MagicMock()
        mock_device.total_memory = 20 * 10**9
        mock_device.major = 8
        mock_device.minor = 0
        mock_properties.return_value = mock_device
        mock_memory.return_value = 5 * 10**9

        with patch('src.adapters.minicpmo_model_adapter.MiniCPMoModelAdapter.load', new=AsyncMock(return_value=True)), \
             patch('src.model_manager.MiniCPMoModelAdapter', new=MagicMock()):
            success, msg = await ModelManager.load("minicpm-o_2_6", timeout=10.0)
            self.assertTrue(success)
            self.assertEqual(msg, "Successfully loaded model: minicpm-o_2_6")

    async def test_load_invalid_model(self):
        success, msg = await ModelManager.load("invalid-model", timeout=10.0)
        self.assertFalse(success)
        self.assertIn("not found", msg)

    async def test_load_already_loaded(self):
        ModelManager.current_model_id = "minicpm-o_2_6"
        ModelManager.current_adapter = MagicMock()
        success, msg = await ModelManager.load("minicpm-o_2_6", timeout=10.0)
        self.assertTrue(success)
        self.assertIn("already loaded", msg)

    @patch('src.adapters.minicpmo_model_adapter.MiniCPMoModelAdapter.unload', new=AsyncMock())
    async def test_unload_model(self):
        ModelManager.current_model_id = "minicpm-o_2_6"
        ModelManager.current_adapter = MagicMock()
        success, msg = await ModelManager.unload()
        self.assertTrue(success)
        self.assertEqual(msg, "Model unloaded")
        self.assertIsNone(ModelManager.current_model_id)
        self.assertIsNone(ModelManager.current_adapter)

    async def test_unload_no_model(self):
        ModelManager.current_model_id = None
        ModelManager.current_adapter = None
        success, msg = await ModelManager.unload()
        self.assertFalse(success)
        self.assertEqual(msg, "No model loaded")

    async def test_unload_busy_server(self):
        ModelManager.server_state = "busy"
        ModelManager.busy_reason = "loading"
        success, msg = await ModelManager.unload()
        self.assertFalse(success)
        self.assertIn("server is busy", msg)

    def test_register_adapter(self):
        class DummyAdapter:
            pass
        ModelManager.register_adapter("DummyAdapter", DummyAdapter)
        self.assertIn("DummyAdapter", ModelManager.ADAPTER_REGISTRY)
        self.assertEqual(ModelManager.ADAPTER_REGISTRY["DummyAdapter"], DummyAdapter)

    @patch('os.listdir', return_value=['minicpmo_model_adapter.py', 'causal_lm_adapter.py'])
    @patch('importlib.import_module')
    def test_discover_adapters(self, mock_import, mock_listdir):
        mock_module = MagicMock()
        mock_module.MiniCPMoModelAdapter = MiniCPMoModelAdapter
        mock_module.CausalLMAdapter = CausalLMAdapter
        mock_import.return_value = mock_module
        ModelManager.discover_adapters()
        self.assertIn("MiniCPMoModelAdapter", ModelManager.ADAPTER_REGISTRY)
        self.assertIn("CausalLMAdapter", ModelManager.ADAPTER_REGISTRY)

    @patch('os.listdir', return_value=[])
    def test_discover_no_adapters(self, mock_listdir):
        ModelManager.ADAPTER_REGISTRY = {}
        ModelManager.discover_adapters()
        self.assertEqual(ModelManager.ADAPTER_REGISTRY, {})

if __name__ == "__main__":
    unittest.main()