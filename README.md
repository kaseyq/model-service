# Model Service Project

This project implements a WebSocket-based TCP server for AI model inference, supporting models like `minicpm-o_2_6` and `codellama-13b` with CUDA acceleration. It runs on a NAS-NVR Ubuntu server with an NVIDIA GPU (24 GB VRAM, CUDA 12.4) and is deployed using Docker. The setup includes configuration files, Python scripts, adapters, and a symlink for model weights to optimize storage.

## Project Overview
- **Purpose**: Serve AI models for inference via a WebSocket API, supporting audio (`minicpm-o_2_6`) and causal language models (`codellama-13b`).
- **Environment**: Ubuntu server with Python 3.12.3 (host), CUDA 12.4, NVIDIA driver 550.144.03, Docker 28.1.1.
- **Deployment**: Dockerized using `docker-compose.yml` and a `Dockerfile` with Python 3.10 (Ubuntu 22.04).
- **Key Features**:
  - Model weights stored externally and linked via symlinks to `./storage`.
  - No-cloud policy: Local model weights and dependencies.
  - Git integration with SSH authentication (`git@github.com:kaseyq/model-service.git`).

## File Structure and Purpose
| File/Directory | Purpose |
|----------------|---------|
| `.gitignore` | Excludes `storage/`, `venv/`, `setup.log`, `.DS_Store`, `._*`, and temporary files from Git. |
| `.dockerignore` | Excludes `storage/`, `venv/`, `setup.log`, and similar from Docker build context to reduce build size. |
| `Dockerfile` | Defines the Docker image, using `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` (Python 3.10), installs dependencies, and sets up the server. |
| `docker-compose.yml` | Configures the Docker service (`model-service`), maps port 9999, mounts volumes under `./storage`, and enables NVIDIA GPU support. |
| `config.yml` | Specifies server settings (host: `127.0.0.1`, port: 9999) and model configurations (`minicpm-o_2_6`, `codellama-13b`) with paths (`/app/minicpm-o_2_6`, `/app/codellama-13b`). |
| `model_server.py` | WebSocket server script that loads and serves models using adapters, handles tasks like `load_model`. |
| `requirements.txt` | Lists Python dependencies (`torch==2.3.1`, `numpy==1.24.1`, `pyyaml==6.0.2`, etc.) for the container. |
| `setup.sh` | Host setup script: creates `venv`, installs dependencies, sets up `./storage` directories, creates `config.yml`, fixes permissions, and checks system requirements. |
| `clean.sh` | Removes `venv` and `setup.log` for resetting the host environment. |
| `adapters/` | Directory containing adapter scripts for model loading: |
| `adapters/__init__.py` | Marks `adapters/` as a Python package. |
| `adapters/base_adapter.py` | Base class for model adapters. |
| `adapters/causal_lm_adapter.py` | Adapter for causal language models (`codellama-13b`). |
| `adapters/minicpmo_model_adapter.py` | Adapter for audio models (`minicpm-o_2_6`). |
| `storage/` | Directory for volume mounts (ignored in Git), containing symlinks (e.g., `./storage/minicpm-o_2_6` → `/home/kaseyq/projects/repos/MiniCPM-o-2_6`). |
| `venv/` | Host virtual environment (optional, ignored in Git). |
| `setup.log` | Log file for `setup.sh` output (ignored in Git). |

## Prerequisites
- **Hardware**: NVIDIA GPU (e.g., RTX 3090, 24 GB VRAM), 64 GB RAM.
- **Software**:
  - Ubuntu (host) with Python 3.12.3.
  - Docker 28.1.1 with NVIDIA Docker runtime.
  - Git for repository management.
  - CUDA 12.4, NVIDIA driver 550.144.03.
- **Dependencies**: Model weights stored locally (e.g., `/home/kaseyq/projects/repos/MiniCPM-o-2_6`).

## Setup Instructions
### 1. Clone the Repository
Clone the project using SSH:
```bash
git clone git@github.com:kaseyq/model-service.git
cd model-service
```

### 2. Configure Git
Set your Git user details:
```bash
git config user.name "Kasey Quanrud"
git config user.email "kasey@kaseyquanrud.com"
```

### 3. Set Up Symlinks for Model Weights
Link model weights to `./storage`:
```bash
mkdir -p ./storage
ln -s /home/kaseyq/projects/repos/MiniCPM-o-2_6 ./storage/minicpm-o_2_6
# For codellama-13b (if available):
# ln -s /path/to/CodeLlama-13b ./storage/codellama-13b
```
Verify:
```bash
ls -ld ./storage/minicpm-o_2_6
```

### 4. Fix Permissions and Clean Metadata
Ensure correct permissions and remove macOS metadata:
```bash
chmod 644 config.yml docker-compose.yml model_server.py requirements.txt Dockerfile
chmod 755 setup.sh clean.sh
rm -f ./storage/.DS_Store ./storage/._.DS_Store
```

### 5. Run `setup.sh` (Optional, Host Setup)
For host-side setup (e.g., non-Docker testing or directory creation):
```bash
./setup.sh
```
- Creates `./storage` directories, `config.yml`, and checks system requirements.
- Logs output to `setup.log`.

### 6. Install Docker Compose V2
Ensure the Docker Compose V2 plugin is installed:
```bash
sudo apt update
sudo apt install docker-compose-plugin
docker compose version
```

### 7. Build and Run the Docker Service
Build the Docker image and start the service:
```bash
docker compose build --no-cache
docker compose up -d
```
Check container status:
```bash
docker ps
docker logs model-service
```

### 8. Test Model Loading
Load a model via the WebSocket API:
```bash
curl -X POST http://127.0.0.1:9999 -d '{"task": "load_model", "model_config_id": "minicpm-o_2_6"}'
```
Or test manually:
```bash
docker exec -it model-service bash
python3 model_server.py --model-config-id minicpm-o_2_6
```
Logs are written to `./storage/model_server.log`.

## Configuration Details
### `config.yml`
- **Host**: `127.0.0.1`
- **Port**: `9999`
- **Timeout**: 300.0 seconds
- **Models**:
  - `minicpm-o_2_6`: Audio model, uses `/app/minicpm-o_2_6`, `bfloat16`, `MiniCPMoModelAdapter`.
  - `codellama-13b`: Causal language model, uses `/app/codellama-13b`, 4-bit quantization, `CausalLMAdapter`.

### `docker-compose.yml`
- **Service**: `model-service`
- **Container Name**: `model-service`
- **Image**: Built from `Dockerfile` (CUDA 12.4, Ubuntu 22.04, Python 3.10).
- **Volumes**:
  - `./:/app` (code).
  - `./storage/minicpm-o_2_6:/app/minicpm-o_2_6` (model weights).
  - `./storage/codellama-13b:/app/codellama-13b` (model weights).
  - `./storage/huggingface_cache:/root/.cache/huggingface` (cache).
  - `./storage/file_storage:/app/file_storage` (outputs).
  - `./storage/tmp:/tmp` (temporary files).
  - `./storage/model_server.log:/app/model_server.log` (logs).
  - `tmpfs` for `/tmp/cache` (1 GB, in-memory).
- **Port**: `9999:9999`
- **Environment**: `NVIDIA_VISIBLE_DEVICES=0`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility,video`.
- **GPU**: 1 NVIDIA GPU via `deploy.resources.reservations.devices`.
- **Restart**: `unless-stopped`
- **Shared Memory**: `shm_size: 40gb`
- **DNS**: `8.8.8.8`
- **Logging**: JSON, 10 MB, 3 rotations.

### `requirements.txt`
- Dependencies: `torch==2.3.1`, `torchaudio==2.3.1`, `torchvision==0.18.1`, `numpy==1.24.1`, `pyyaml==6.0.2`, `transformers==4.44.2`, `bitsandbytes==0.44.1`.
- Installed in the container using PyPI (`--index-url https://pypi.org/simple`) and PyTorch CUDA 12.1 index (`--extra-index-url https://download.pytorch.org/whl/cu121`).

## Troubleshooting
### Build Failures
- **Error**: Incompatible dependency versions (e.g., `pyyaml==6.0.2` not found).
  - Verify `Dockerfile` uses `--index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu121`.
  - Check `requirements.txt`:
    ```bash
    cat requirements.txt
    ```
  - Rebuild:
    ```bash
    docker compose build --no-cache
    ```

### Model Loading Issues
- **Error**: Model fails to load (`minicpm-o_2_6`).
  - Verify symlink:
    ```bash
    ls -ld ./storage/minicpm-o_2_6
    ```
  - Test in container:
    ```bash
    docker exec -it model-service bash
    python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('/app/minicpm-o_2_6', local_files_only=True)"
    ```
  - Adjust permissions:
    ```bash
    chmod -R 755 /home/kaseyq/projects/repos/MiniCPM-o-2_6
    ```

### Port Conflicts
- Check port `9999`:
  ```bash
  sudo netstat -tuln | grep 9999
  sudo lsof -i :9999
  ```
- Kill conflicting processes:
  ```bash
  sudo kill -9 <pid>
  ```

### Residual Processes
- Kill stalled `systemctl` processes:
  ```bash
  sudo kill -9 473464 473465 473466 475475 475476 475477
  ps aux | grep model-service
  ```

## Notes
- **Python Version**: Host uses Python 3.12.3; container uses Python 3.10 (Ubuntu 22.04). Dependencies are compatible.
- **Symlinks**: Model weights are linked (e.g., `./storage/minicpm-o_2_6` → `/home/kaseyq/projects/repos/MiniCPM-o-2_6`) to avoid duplication (~16.5 GB).
- **No-Cloud Policy**: Local weights and dependencies ensure offline operation.
- **Git**: Repository (`git@github.com:kaseyq/model-service.git`) excludes `storage/`, `venv/`, and `setup.log`.
- **setup.sh**: Useful for host-side setup (directories, symlinks, permissions, system checks), kept unchanged for reference.
- **clean.sh**: Resets host environment by removing `venv` and `setup.log`.

## Contributing
To contribute:
1. Fork the repository.
2. Create a branch (`git checkout -b feature-branch`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push (`git push origin feature-branch`).
5. Open a pull request.