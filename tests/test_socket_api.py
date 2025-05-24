import unittest
import json
import socket
import time
import sys
import os

# Ensure project root is in sys.path for imports when running from tests/
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.common.socket_utils import ModelServiceClient
from src.common.file_utils import load_yaml
from src.common.path_utils import PathUtil

class TestModelServiceAPI(unittest.TestCase):
    def setUp(self):
        """Set up the test client using server configuration from config.yml."""
        self.config = load_yaml([PathUtil.get_path("config.yml"), PathUtil.get_path("config.yaml")])
        print(f"Loaded config: host={self.config['host']}, port={self.config['port']}, api_version={self.config['api_version']}")
        self.host = self.config['host']
        self.port = self.config['port']
        self.api_version = self.config['api_version']
        
        self.client = ModelServiceClient(host=self.host, port=self.port, timeout=60)
        print(f"Attempting to unload model to reset server state...")
        self._unload_model()
        print("Unload complete")
        time.sleep(1)

    def tearDown(self):
        """Clean up by unloading any loaded model."""
        print("Tearing down: unloading model...")
        self._unload_model()
        print("Teardown complete")

    def _unload_model(self):
        """Helper method to unload the current model."""
        # First, get the current model
        status_request = {
            "version": self.api_version,
            "task": "status"
        }
        status_response = self.client.send_request(status_request)
        print(f"Status response: {status_response}")
        current_model_id = status_response.get("model_config_id", None)
        
        if current_model_id:
            unload_request = {
                "version": self.api_version,
                "task": "unload_model",
                "model_config_id": current_model_id
            }
            unload_response = self.client.send_request(unload_request)
            print(f"Unload response for {current_model_id}: {unload_response}")
        else:
            print("No model loaded, skipping unload")

    def test_status(self):
        """Test the 'status' task to retrieve server health and VRAM usage."""
        request = {
            "version": self.api_version,
            "task": "status"
        }
        response = self.client.send_request(request)
        self.assertEqual(response["status"], "okay", f"Expected status 'okay', got {response}")
        self.assertEqual(response["server_state"], "ready", f"Expected server_state 'ready', got {response['server_state']}")
        self.assertIn("payload", response, "Response missing payload")
        payload = response["payload"]
        self.assertIn("health", payload, "Payload missing health")
        self.assertIn("vram_usage", payload, "Payload missing vram_usage")
        self.assertIn("uptime_seconds", payload, "Payload missing uptime_seconds")
        self.assertGreaterEqual(payload["uptime_seconds"], 0, "Uptime should be non-negative")

    def test_list_models(self):
        """Test the 'list_models' task to retrieve available models."""
        request = {
            "version": self.api_version,
            "task": "list_models"
        }
        response = self.client.send_request(request)
        self.assertEqual(response["status"], "okay", f"Expected status 'okay', got {response}")
        self.assertIn("payload", response, "Response missing payload")
        payload = response["payload"]
        self.assertIn("models", payload, "Payload missing models")
        self.assertIn("minicpm-o_2_6", payload["models"], "Expected model 'minicpm-o_2_6' in models")
        self.assertIn("codellama-13b", payload["models"], "Expected model 'codellama-13b' in models")
        self.assertIn("current_model", payload, "Payload missing current_model")
        self.assertIn("current_config", payload, "Payload missing current_config")

    def test_load_model(self):
        """Test the 'load_model' task for codellama-13b."""
        request = {
            "version": self.api_version,
            "task": "load_model",
            "model_config_id": "codellama-13b"
        }
        response = self.client.send_request(request)
        self.assertEqual(response["status"], "okay", f"Expected status 'okay', got {response}")
        self.assertEqual(response["server_state"], "ready", f"Expected server_state 'ready', got {response['server_state']}")
        self.assertEqual(response["model_config_id"], "codellama-13b", f"Expected model_config_id 'codellama-13b', got {response['model_config_id']}")
        self.assertIn("Model codellama-13b", response["message"], f"Expected message to mention model loading, got {response['message']}")

    def test_unload_model(self):
        """Test the 'unload_model' task after loading a model."""
        load_request = {
            "version": self.api_version,
            "task": "load_model",
            "model_config_id": "codellama-13b"
        }
        load_response = self.client.send_request(load_request)
        self.assertEqual(load_response["status"], "okay", f"Failed to load model: {load_response}")

        unload_request = {
            "version": self.api_version,
            "task": "unload_model",
            "model_config_id": "codellama-13b"
        }
        response = self.client.send_request(unload_request)
        self.assertEqual(response["status"], "okay", f"Expected status 'okay', got {response}")
        self.assertEqual(response["server_state"], "ready", f"Expected server_state 'ready', got {response['server_state']}")
        self.assertIsNone(response["model_config_id"], f"Expected model_config_id to be None, got {response['model_config_id']}")
        self.assertEqual(response["message"], "Model unloaded", f"Expected message 'Model unloaded', got {response['message']}")

    def test_prompt_codellama(self):
        """Test the 'prompt' task with codellama-13b."""
        load_request = {
            "version": self.api_version,
            "task": "load_model",
            "model_config_id": "codellama-13b"
        }
        load_response = self.client.send_request(load_request)
        self.assertEqual(load_response["status"], "okay", f"Failed to load model: {load_response}")

        prompt_request = {
            "version": self.api_version,
            "task": "prompt",
            "model_config_id": "codellama-13b",
            "input": "def hello_world():",
            "params": {
                "max_length": 50,
                "temperature": 0.7,
                "top_p": 0.9
            }
        }
        response = self.client.send_request(prompt_request)  # Fixed: Use prompt_request
        self.assertEqual(response["status"], "okay", f"Expected status 'okay', got {response}")
        self.assertIn("payload", response, "Response missing payload")
        payload = response["payload"]
        self.assertIn("result", payload, "Payload missing result")
        self.assertIn("text", payload["result"], "Result missing text")
        self.assertIn("hello_world", payload["result"]["text"], "Expected 'hello_world' in response text")

    def test_invalid_version(self):
        """Test request with invalid API version."""
        request = {
            "version": 999,
            "task": "status"
        }
        response = self.client.send_request(request)
        self.assertEqual(response["status"], "error", f"Expected status 'error', got {response}")
        self.assertIn("Invalid or missing version", response["message"], f"Expected version error message, got {response['message']}")

    def test_invalid_task(self):
        """Test request with invalid task."""
        request = {
            "version": self.api_version,
            "task": "invalid_task",
            "model_config_id": "codellama-13b"
        }
        response = self.client.send_request(request)
        self.assertEqual(response["status"], "error", f"Expected status 'error', got {response}")
        self.assertIn("Invalid task", response["message"], f"Expected invalid task message, got {response['message']}")

if __name__ == "__main__":
    unittest.main()