import unittest
from unittest.mock import patch, MagicMock
import sys
import io
import asyncio
from src.command_line_interface import command_line_interface
from src.model_manager import ModelManager

class TestCommandLineInterface(unittest.TestCase):
    def setUp(self):
        self.stdout = io.StringIO()
        sys.stdout = self.stdout
        ModelManager.MODEL_LOOKUP = {
            'codellama-13b': {'model_config_id': 'codellama-13b'},
            'minicpm-o_2_6': {'model_config_id': 'minicpm-o_2_6'}
        }
        ModelManager.API_VERSION = 1
        ModelManager.config = {'host': '0.0.0.0', 'port': 9999, 'timeout': 300.0}
        ModelManager.current_model_id = None
        ModelManager.server_state = "ready"

    def tearDown(self):
        sys.stdout = sys.__stdout__

    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', side_effect=['list', 'exit'])
    def test_list_command(self, mock_input, mock_isatty):
        with patch('sys.exit') as mock_exit:
            command_line_interface()
            output = self.stdout.getvalue()
            self.assertIn("Available models: ['codellama-13b', 'minicpm-o_2_6']", output)
            mock_exit.assert_called_with(0)

    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', side_effect=['unload', 'exit'])
    @patch('src.model_manager.ModelManager.unload', new=AsyncMock(return_value=(True, "Model unloaded")))
    def test_unload_command(self, mock_unload, mock_input, mock_isatty):
        with patch('sys.exit') as mock_exit:
            command_line_interface()
            output = self.stdout.getvalue()
            self.assertIn("unload: True Model unloaded", output)
            mock_unload.assert_called_once()
            mock_exit.assert_called_with(0)

    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', side_effect=['unload', 'exit'])
    @patch('src.model_manager.ModelManager.unload', new=AsyncMock(return_value=(False, "No model loaded")))
    def test_unload_no_model(self, mock_unload, mock_input, mock_isatty):
        with patch('sys.exit') as mock_exit:
            command_line_interface()
            output = self.stdout.getvalue()
            self.assertIn("unload: False No model loaded", output)
            mock_unload.assert_called_once()
            mock_exit.assert_called_with(0)

    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', side_effect=['load codellama-13b', 'exit'])
    @patch('src.model_manager.ModelManager.load', new=AsyncMock(return_value=(True, "Successfully loaded model: codellama-13b")))
    def test_load_command(self, mock_load, mock_input, mock_isatty):
        with patch('sys.exit') as mock_exit:
            command_line_interface()
            output = self.stdout.getvalue()
            self.assertIn("load: True Successfully loaded model: codellama-13b", output)
            mock_load.assert_called_with("codellama-13b", 300.0)
            mock_exit.assert_called_with(0)

    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', side_effect=['load invalid-model', 'exit'])
    @patch('src.model_manager.ModelManager.load', new=AsyncMock(return_value=(False, "Model config ID invalid-model not found")))
    def test_load_invalid_model(self, mock_load, mock_input, mock_isatty):
        with patch('sys.exit') as mock_exit:
            command_line_interface()
            output = self.stdout.getvalue()
            self.assertIn("load: False Model config ID invalid-model not found", output)
            mock_load.assert_called_with("invalid-model", 300.0)
            mock_exit.assert_called_with(0)

    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', side_effect=['status', 'exit'])
    @patch('torch.cuda.get_device_properties')
    @patch('torch.cuda.memory_allocated')
    def test_status_command(self, mock_memory, mock_properties, mock_input, mock_isatty):
        mock_device = MagicMock()
        mock_device.total_memory = 20 * 10**9
        mock_properties.return_value = mock_device
        mock_memory.return_value = 5 * 10**9
        with patch('sys.exit') as mock_exit:
            command_line_interface()
            output = self.stdout.getvalue()
            self.assertIn("Health: ready", output)
            self.assertIn("Current model: none", output)
            self.assertIn("VRAM", output)
            mock_exit.assert_called_with(0)

    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', side_effect=['prompt test', 'exit'])
    @patch('src.common.socket_utils.ModelServiceClient.send_request')
    def test_prompt_command(self, mock_send_request, mock_input, mock_isatty):
        ModelManager.current_model_id = "codellama-13b"
        mock_send_request.return_value = {
            "status": "okay",
            "payload": {"result": {"text": "test output"}}
        }
        with patch('sys.exit') as mock_exit:
            command_line_interface()
            output = self.stdout.getvalue()
            self.assertIn("Response: test output", output)
            mock_send_request.assert_called()
            mock_exit.assert_called_with(0)

    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', side_effect=['prompt test', 'exit'])
    def test_prompt_no_model(self, mock_input, mock_isatty):
        ModelManager.current_model_id = None
        with patch('sys.exit') as mock_exit:
            command_line_interface()
            output = self.stdout.getvalue()
            self.assertIn("Error: No model loaded", output)
            mock_exit.assert_called_with(0)

    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', side_effect=['prompt', 'exit'])
    @patch('src.common.socket_utils.ModelServiceClient.send_request')
    def test_prompt_empty(self, mock_send_request, mock_input, mock_isatty):
        ModelManager.current_model_id = "codellama-13b"
        with patch('sys.exit') as mock_exit:
            command_line_interface()
            output = self.stdout.getvalue()
            self.assertIn("Error: Prompt input required", output)
            mock_send_request.assert_not_called()
            mock_exit.assert_called_with(0)

    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', side_effect=['invalid_command', 'exit'])
    def test_invalid_command(self, mock_input, mock_isatty):
        with patch('sys.exit') as mock_exit:
            command_line_interface()
            output = self.stdout.getvalue()
            self.assertIn("Unknown command", output)
            mock_exit.assert_called_with(0)

    @patch('sys.stdin.isatty', return_value=True)
    @patch('builtins.input', side_effect=[KeyboardInterrupt, 'exit'])
    def test_keyboard_interrupt(self, mock_input, mock_isatty):
        with patch('sys.exit') as mock_exit:
            command_line_interface()
            output = self.stdout.getvalue()
            self.assertIn("", output)  # CLI continues after interrupt
            mock_exit.assert_called_with(0)

if __name__ == "__main__":
    unittest.main()