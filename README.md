Model Service
A socket-based model server with adapters for specific AI models, supporting client or console input. Deployable via virtual environment (venv) or Docker. Hosted at https://github.com/kaseyq/model-service.
Features

Socket-based TCP server for AI model inference
Adapters for models like MiniCPM-o_2_6 (audio) and CodeLlama-13b (causal LM)
Command-line interface for model management (load, unload, list, status)
Supports venv or Docker deployment
Configurable model storage for HuggingFace and local models

Directory Structure
model-service/
├── config.yml               # Server and model configuration
├── docker-compose.yml       # Docker Compose for production
├── docker-compose-dev.yml   # Docker Compose for development
├── Dockerfile              # Docker image definition
├── __main__.py             # Server entry point
├── requirements.txt         # Python dependencies for venv
├── requirements_docker.txt  # Python dependencies for Docker
├── src/                    # Source code
│   ├── adapters/           # Model-specific adapters
│   │   ├── base_adapter.py
│   │   ├── causal_lm_adapter.py
│   │   ├── minicpmo_model_adapter.py
│   ├── common/             # Shared functionality
│   │   ├── file_utils.py
│   │   ├── handlers.py
│   │   ├── __init__.py
│   │   ├── path_utils.py
│   │   ├── socket_utils.py
│   ├── command_line_interface.py
│   ├── model_manager.py
│   ├── model_request_handler.py
├── storage/                # Model and data storage
│   ├── codellama-13b/
│   ├── file_storage/
│   ├── huggingface_cache/
│   ├── minicpm-o_2_6/
│   ├── models/
│   ├── tmp/

Setup
Prerequisites

Python 3.8+ (for venv)
Docker and Docker Compose (for Docker)
Git
CUDA-capable GPU with at least 5 GB VRAM (for model inference)

Virtual Environment Setup

Clone the repository:
git clone https://github.com/kaseyq/model-service.git
cd model-service


Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:
pip install -r requirements.txt



Docker Setup

Clone the repository:
git clone https://github.com/kaseyq/model-service.git
cd model-service


Build and run with Docker Compose:
docker-compose up --build

For development:
docker-compose -f docker-compose-dev.yml up --build



Usage
Running the Server

Virtual Environment:Start the server with an optional model config ID (e.g., minicpm-o_2_6 or codellama-13b):
python __main__.py --model-config-id minicpm-o_2_6

Defaults to the first model in config.yml if unspecified.

Docker:Server starts automatically with docker-compose up, loading the default model.


The server runs on 0.0.0.0:9999 by default (configurable in config.yml).
Interacting with the Server

Command-Line Interface:Manage models interactively (requires terminal):
python src/command_line_interface.py

Available commands:

load <model_config_id>: Load a model (e.g., load minicpm-o_2_6)
unload: Unload the current model
list: Show available and current models
status: Display server health, current model, VRAM usage, and uptime
exit: Shut down the server


Client Interaction:Connect via a TCP socket client to 0.0.0.0:9999. Request handling details are in src/model_request_handler.py.


Model Compilation
To compile specific AI model file contents:
./util/compile-scoped.sh [file_path ...]

Configuration
Edit config.yml to adjust:

Server settings (host, port, timeout, api_version)
Model configurations (e.g., minicpm-o_2_6, codellama-13b)
Device settings (cuda:0), data types (bfloat16, 4bit), and local model paths

Example model entry:
models:
  - model_config_id: minicpm-o_2_6
    model_name: Openbmb/MiniCPM-o_2_6
    type: audio
    device: cuda:0
    torch_dtype: bfloat16
    local_path: ./storage/models/minicpm-o_2_6
    adapter_class: MiniCPMoModelAdapter

Ensure the storage/ directory contains model weights and cached data.
Contributing
Contributions are welcome! Open an issue or submit a pull request on GitHub.
License
MIT License. See LICENSE file.
