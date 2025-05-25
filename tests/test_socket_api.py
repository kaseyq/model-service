import unittest
import json
import sys
import os
import asyncio
from unittest.mock import patch, MagicMock
from httpx import AsyncClient

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.common.socket_utils import ModelServiceClient
from src.common.file_utils import load_yaml
from src.common.path_utils import PathUtil

class TestModelServiceAPI(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = load_yaml([PathUtil.get_path("config.yml"), PathUtil.get_path("config.yaml")])
        self.host = self.config['host']
        self.port = self.config['port']
        self.api_version = self.config['api_version']
        self.client = ModelServiceClient(host=self.host, port=self.port, timeout=60)
        self.http_client = AsyncClient(base_url=f"http://{self.host}:8000")

    async def asyncSetUp(self):
        await self._unload_model()

    async def asyncTearDown(self):
        await self._unload_model()
        await self.http_client.aclose()

    async def _unload_model(self):
        status_request = {
            "version": self.api_version,
            "task": "status"
        }
        status_response = self.client.send_request(status_request)
        current_model_id = status_response.get("model_config_id", None)
        if current_model_id:
            unload_request = {
                "version": self.api_version,
                "task": "unload_model",
                "model_config_id": current_model_id
            }
            self.client.send_request(unload_request)

    def test_status(self):
        request = {
            "version": self.api_version,
            "task": "status"
        }
        response = self.client.send_request(request)
        self.assertEqual(response["status"], "okay")
        self.assertEqual(response["server_state"], "ready")
        self.assertIn("payload", response)
        payload = response["payload"]
        self.assertIn("health", payload)
        self.assertIn("vram_usage", payload)
        self.assertIn("uptime_seconds", payload)
        self.assertGreaterEqual(payload["uptime_seconds"], 0)

    def test_list_models(self):
        request = {
            "version": self.api_version,
            "task": "list_models"
        }
        response = self.client.send_request(request)
        self.assertEqual(response["status"], "okay")
        self.assertIn("payload", response)
        payload = response["payload"]
        self.assertIn("models", payload)
        self.assertIn("minicpm-o_2_6", payload["models"])
        self.assertIn("codellama-13b", payload["models"])
        self.assertIn("clip-vision", payload["models"])
        self.assertIn("current_model", payload)
        self.assertIn("current_config", payload)

    def test_load_model(self):
        for model_id in ["codellama-13b", "minicpm-o_2_6", "clip-vision"]:
            request = {
                "version": self.api_version,
                "task": "load_model",
                "model_config_id": model_id
            }
            response = self.client.send_request(request)
            self.assertEqual(response["status"], "okay")
            self.assertEqual(response["server_state"], "ready")
            self.assertEqual(response["model_config_id"], model_id)
            self.assertIn(f"Model {model_id}", response["message"])

    def test_load_invalid_model(self):
        request = {
            "version": self.api_version,
            "task": "load_model",
            "model_config_id": "invalid-model"
        }
        response = self.client.send_request(request)
        self.assertEqual(response["status"], "error")
        self.assertIn("not found", response["message"])

    def test_unload_model(self):
        load_request = {
            "version": self.api_version,
            "task": "load_model",
            "model_config_id": "codellama-13b"
        }
        self.client.send_request(load_request)

        unload_request = {
            "version": self.api_version,
            "task": "unload_model",
            "model_config_id": "codellama-13b"
        }
        response = self.client.send_request(unload_request)
        self.assertEqual(response["status"], "okay")
        self.assertEqual(response["server_state"], "ready")
        self.assertIsNone(response["model_config_id"])
        self.assertEqual(response["message"], "Model unloaded")

    def test_unload_wrong_model(self):
        load_request = {
            "version": self.api_version,
            "task": "load_model",
            "model_config_id": "codellama-13b"
        }
        self.client.send_request(load_request)

        unload_request = {
            "version": self.api_version,
            "task": "unload_model",
            "model_config_id": "minicpm-o_2_6"
        }
        response = self.client.send_request(unload_request)
        self.assertEqual(response["status"], "error")
        self.assertIn("Cannot unload minicpm-o_2_6", response["message"])

    def test_prompt_codellama(self):
        load_request = {
            "version": self.api_version,
            "task": "load_model",
            "model_config_id": "codellama-13b"
        }
        self.client.send_request(load_request)

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
        response = self.client.send_request(prompt_request)
        self.assertEqual(response["status"], "okay")
        self.assertIn("payload", response)
        self.assertIn("result", response["payload"])
        self.assertIn("text", response["payload"]["result"])
        self.assertIn("hello_world", response["payload"]["result"]["text"])

    def test_prompt_minicpm(self):
        load_request = {
            "version": self.api_version,
            "task": "load_model",
            "model_config_id": "minicpm-o_2_6"
        }
        self.client.send_request(load_request)

        prompt_request = {
            "version": self.api_version,
            "task": "prompt",
            "model_config_id": "minicpm-o_2_6",
            "messages": [
                {
                    "role": "user",
                    "content": ["test prompt"]
                }
            ],
            "params": {}
        }
        response = self.client.send_request(prompt_request)
        self.assertEqual(response["status"], "okay")
        self.assertIn("payload", response)
        self.assertIn("result", response["payload"])
        self.assertIn("files", response["payload"]["result"])
        self.assertIn("output_audio_path", response["payload"]["result"]["files"])
        self.assertTrue(response["payload"]["result"]["files"]["output_audio_path"]["data"].startswith("base64:"))

    def test_prompt_clip_vision(self):
        load_request = {
            "version": self.api_version,
            "task": "load_model",
            "model_config_id": "clip-vision"
        }
        self.client.send_request(load_request)

        prompt_request = {
            "version": self.api_version,
            "task": "prompt",
            "model_config_id": "clip-vision",
            "input_type": "image/jpeg",
            "data": "base64:iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        }
        response = self.client.send_request(prompt_request)
        self.assertEqual(response["status"], "okay")
        self.assertIn("payload", response)
        self.assertIn("result", response["payload"])
        self.assertIn("features", response["payload"]["result"])

    def test_prompt_invalid_input(self):
        load_request = {
            "version": self.api_version,
            "task": "load_model",
            "model_config_id": "codellama-13b"
        }
        self.client.send_request(load_request)

        prompt_request = {
            "version": self.api_version,
            "task": "prompt",
            "model_config_id": "codellama-13b",
            "input": "",
            "params": {}
        }
        response = self.client.send_request(prompt_request)
        self.assertEqual(response["status"], "error")
        self.assertIn("Prompt input required", response["message"])

    def test_prompt_wrong_model(self):
        load_request = {
            "version": self.api_version,
            "task": "load_model",
            "model_config_id": "codellama-13b"
        }
        self.client.send_request(load_request)

        prompt_request = {
            "version": self.api_version,
            "task": "prompt",
            "model_config_id": "minicpm-o_2_6",
            "messages": [{"role": "user", "content": ["test"]}],
            "params": {}
        }
        response = self.client.send_request(prompt_request)
        self.assertEqual(response["status"], "error")
        self.assertIn("not loaded", response["message"])

    def test_invalid_version(self):
        request = {
            "version": 999,
            "task": "status"
        }
        response = self.client.send_request(request)
        self.assertEqual(response["status"], "error")
        self.assertIn("Invalid or missing version", response["message"])

    def test_invalid_task(self):
        request = {
            "version": self.api_version,
            "task": "invalid_task",
            "model_config_id": "codellama-13b"
        }
        response = self.client.send_request(request)
        self.assertEqual(response["status"], "error")
        self.assertIn("Invalid task", response["message"])

    async def test_http_status(self):
        response = await self.http_client.post(f"/api/v{self.api_version}/task", json={
            "version": self.api_version,
            "task": "status"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "okay")
        self.assertEqual(data["server_state"], "ready")
        self.assertIn("payload", data)

    async def test_http_load_model(self):
        response = await self.http_client.post(f"/api/v{self.api_version}/task", json={
            "version": self.api_version,
            "task": "load_model",
            "model_config_id": "codellama-13b"
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "okay")
        self.assertEqual(data["model_config_id"], "codellama-13b")

    async def test_http_invalid_version(self):
        response = await self.http_client.post("/api/v999/task", json={
            "version": 999,
            "task": "status"
        })
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("Invalid version", data["detail"])

    async def test_http_busy_server(self):
        with patch('src.model_manager.ModelManager.server_state', "busy"), \
             patch('src.model_manager.ModelManager.busy_reason', "loading"):
            response = await self.http_client.post(f"/api/v{self.api_version}/task", json={
                "version": self.api_version,
                "task": "load_model",
                "model_config_id": "codellama-13b"
            })
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["status"], "error")
            self.assertIn("server is busy", data["message"])

    async def test_malformed_request(self):
        response = await self.http_client.post(f"/api/v{self.api_version}/task", json={})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "error")
        self.assertIn("Invalid or missing version", data["message"])

if __name__ == "__main__":
    unittest.main()