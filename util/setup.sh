#!/bin/bash

# setup.sh: Setup script for model server environment
# - Creates Python virtual environment
# - Installs requirements.txt in venv with CUDA 12.1 index, with retries, no timeout
# - Deactivates venv
# - Creates default config.yml if missing
# - Verifies docker-compose.yml or docker-compose.yaml and Dockerfile exist
# - Creates storage directories for volumes under ./storage
# - Checks system requirements and hardware resources without installing

# Exit on error (except for pip install, handled separately)
set -e

# Log file
LOG_FILE="setup.log"
exec 3>&1 1>>"$LOG_FILE" 2>&1
echo "Setup script started at $(date)" >&3

# Colors for console output
GREEN="\033[0;32m"
RED="\033[0;31m"
NC="\033[0m"

# Helper function to log and print status
log_status() {
    local status=$1
    local message=$2
    if [ "$status" == "OK" ]; then
        echo -e "${GREEN}[OK]${NC} $message" >&3
    else
        echo -e "${RED}[FAIL]${NC} $message" >&3
    fi
}

# Check and fix file permissions, remove macOS metadata
echo "Checking file permissions and metadata..." >&3
for file in config.yml config.yaml docker-compose.yml docker-compose.yaml model_server.py requirements.txt Dockerfile Dockerfile.txt clean.sh; do
    if [ -f "$file" ]; then
        if [[ "$file" == "setup.sh" || "$file" == "clean.sh" ]]; then
            chmod 755 "$file" 2>/dev/null && log_status "OK" "Set permissions for $file" || log_status "FAIL" "Failed to set permissions for $file"
        else
            chmod 644 "$file" 2>/dev/null && log_status "OK" "Set permissions for $file" || log_status "FAIL" "Failed to set permissions for $file"
        fi
    fi
done
if ls ._*.yml ._*.yaml ._.DS_Store .DS_Store >/dev/null 2>&1; then
    rm -f ._*.yml ._*.yaml ._.DS_Store .DS_Store 2>/dev/null && log_status "OK" "Removed macOS metadata files" || log_status "FAIL" "Failed to remove macOS metadata files"
else
    log_status "OK" "No macOS metadata files found"
fi

# Create storage directories for volumes
echo "Creating storage directories for volumes..." >&3
mkdir -p ./storage 2>/dev/null && chmod 755 ./storage 2>/dev/null && log_status "OK" "Created ./storage directory" || log_status "FAIL" "Failed to create ./storage directory"
for dir in minicpm-o_2_6 codellama-13b huggingface_cache file_storage tmp; do
    if [ -d "./storage/$dir" ]; then
        log_status "OK" "Directory ./storage/$dir already exists"
    else
        mkdir -p "./storage/$dir" && chmod 755 "./storage/$dir" 2>/dev/null && log_status "OK" "Created directory ./storage/$dir" || log_status "FAIL" "Failed to create directory ./storage/$dir"
    fi
done

# 1. Create default config.yml if missing
echo "Checking for config.yml or config.yaml..." >&3
if [ -f "config.yml" ] || [ -f "config.yaml" ]; then
    log_status "OK" "Config file (config.yml or config.yaml) already exists, skipping creation"
else
    cat > config.yml << 'EOF'
host: "127.0.0.1"
port: 9999
timeout: 300.0
models:
  - model_config_id: minicpm-o_2_6
    model_name: Openbmb/MiniCPM-o_2_6
    type: audio
    device: cuda:0
    torch_dtype: bfloat16
    local_path: ./minicpm-o_2_6
    adapter_class: MiniCPMoModelAdapter
  - model_config_id: codellama-13b
    model_name: codellama/CodeLlama-13b-hf
    type: causal_lm
    device: cuda:0
    load_in_4bit: true
    local_path: ./codellama-13b
    adapter_class: CausalLMAdapter
EOF
    chmod 644 config.yml 2>/dev/null && log_status "OK" "Created default config.yml" || log_status "FAIL" "Failed to create config.yml"
fi

# 2. Verify docker-compose.yml or docker-compose.yaml exists
echo "Checking for docker-compose.yml or docker-compose.yaml..." >&3
if [ -f "docker-compose.yml" ] || [ -f "docker-compose.yaml" ]; then
    log_status "OK" "Docker Compose file (docker-compose.yml or docker-compose.yaml) exists"
else
    log_status "FAIL" "Docker Compose file (docker-compose.yml or docker-compose.yaml) not found"
    echo "Please create docker-compose.yml or docker-compose.yaml to proceed with Docker setup" >&3
fi

# 3. Verify Dockerfile exists
echo "Checking for Dockerfile..." >&3
if [ -f "Dockerfile" ]; then
    log_status "OK" "Dockerfile exists"
elif [ -f "Dockerfile.txt" ]; then
    log_status "FAIL" "Dockerfile.txt found instead of Dockerfile"
    echo "Please rename Dockerfile.txt to Dockerfile:" >&3
    echo "  mv Dockerfile.txt Dockerfile" >&3
    echo "  chmod 644 Dockerfile" >&3
else
    log_status "FAIL" "Dockerfile not found"
    echo "Please create Dockerfile to proceed with Docker setup" >&3
fi

# 4. Create Python virtual environment
echo "Creating Python virtual environment..." >&3
if [ -d "venv" ]; then
    echo "Virtual environment already exists, skipping creation" >&3
else
    if command -v python3 >/dev/null 2>&1; then
        python3 -m venv venv
        log_status "OK" "Created virtual environment at ./venv"
    else
        log_status "FAIL" "python3 not found"
        exit 1
    fi
fi

# 5. Install requirements in virtual environment
echo "Installing requirements from requirements.txt..." >&3
source venv/bin/activate
if [ -f "requirements.txt" ]; then
    # Retry pip install up to 3 times
    for attempt in {1..3}; do
        echo "Attempt $attempt of 3: Installing PyTorch packages with CUDA 12.1 index..." >&3
        if pip3 install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121 | tee /dev/fd/3 && \
           pip3 install -r requirements.txt | tee /dev/fd/3; then
            log_status "OK" "Installed requirements"
            break
        else
            log_status "FAIL" "Attempt $attempt failed to install requirements"
            if [ $attempt -eq 3 ]; then
                log_status "FAIL" "All attempts failed, check $LOG_FILE for details"
                echo "To debug, activate venv and try manually:" >&3
                echo "  source venv/bin/activate" >&3
                echo "  pip3 install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121" >&3
                echo "  pip3 install -r requirements.txt" >&3
                deactivate
                exit 1
            fi
            sleep 5
        fi
    done
else
    log_status "FAIL" "requirements.txt not found"
    deactivate
    exit 1
fi

# 6. Deactivate virtual environment
deactivate
log_status "OK" "Deactivated virtual environment"

# 7. Check system requirements and hardware resources (no install)
echo "Checking system requirements and hardware resources..." >&3

# Check Python
if command -v python3 >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    log_status "OK" "Python found: $PYTHON_VERSION"
else
    log_status "FAIL" "Python not found"
fi

# Check pip
if command -v pip3 >/dev/null 2>&1; then
    PIP_VERSION=$(pip3 --version 2>&1)
    log_status "OK" "pip found: $PIP_VERSION"
else
    log_status "FAIL" "pip not found"
fi

# Check git
if command -v git >/dev/null 2>&1; then
    GIT_VERSION=$(git --version 2>&1)
    log_status "OK" "git found: $GIT_VERSION"
else
    log_status "FAIL" "git not found"
fi

# Check curl
if command -v curl >/dev/null 2>&1; then
    CURL_VERSION=$(curl --version 2>&1 | head -n 1)
    log_status "OK" "curl found: $CURL_VERSION"
else
    log_status "FAIL" "curl not found"
fi

# Check build-essential
if dpkg -l | grep -q build-essential; then
    log_status "OK" "build-essential found"
else
    log_status "FAIL" "build-essential not found"
fi

# Check NVIDIA driver
if command -v nvidia-smi >/dev/null 2>&1; then
    NVIDIA_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null)
    log_status "OK" "NVIDIA driver found: $NVIDIA_DRIVER"
else
    log_status "FAIL" "NVIDIA driver not found (nvidia-smi missing)"
fi

# Check CUDA version
if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_VERSION=$(nvidia-smi | grep -i cuda | awk '{print $9}' | head -n 1)
    if [[ "$CUDA_VERSION" =~ ^12\.[0-4]$ ]]; then
        log_status "OK" "CUDA version compatible: $CUDA_VERSION (expected 12.x)"
    else
        log_status "FAIL" "CUDA version mismatch: found $CUDA_VERSION, expected 12.x"
    fi
else
    log_status "FAIL" "CUDA version check failed (nvidia-smi missing)"
fi

# Check Docker
if command -v docker >/dev/null 2>&1; then
    DOCKER_VERSION=$(docker --version 2>&1)
    log_status "OK" "Docker found: $DOCKER_VERSION"
else
    log_status "FAIL" "Docker not found"
fi

# Check Docker Compose (V2)
if docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE_VERSION=$(docker compose version 2>&1)
    log_status "OK" "Docker Compose found: $DOCKER_COMPOSE_VERSION"
else
    log_status "FAIL" "Docker Compose not found"
    echo "To install Docker Compose V2 (not performed by this script):" >&3
    echo "  sudo apt update" >&3
    echo "  sudo apt install docker-compose-plugin" >&3
    echo "Or download the binary:" >&3
    echo "  curl -L \"https://github.com/docker/compose/releases/download/v2.29.2/docker-compose-$(uname -s)-$(uname -m)\" -o /usr/local/bin/docker-compose" >&3
    echo "  sudo chmod +x /usr/local/bin/docker-compose" >&3
fi

# Check NVIDIA Docker runtime
if docker info --format '{{.Runtimes}}' | grep -q nvidia 2>/dev/null; then
    log_status "OK" "NVIDIA Docker runtime found"
else
    log_status "FAIL" "NVIDIA Docker runtime not found"
fi

# Check GPU hardware
if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | awk '{print $1/1024 " GB"}')
    log_status "OK" "GPU found: $GPU_NAME, VRAM: $VRAM"
else
    log_status "FAIL" "No GPU detected"
fi

# Check RAM
if command -v free >/dev/null 2>&1; then
    TOTAL_RAM=$(free -h | grep Mem | awk '{print $2}')
    AVAILABLE_RAM=$(free -h | grep Mem | awk '{print $7}')
    log_status "OK" "RAM: Total $TOTAL_RAM, Available $AVAILABLE_RAM"
else
    log_status "FAIL" "Unable to check RAM"
fi

# Check CPU
if command -v lscpu >/dev/null 2>&1; then
    CPU_MODEL=$(lscpu | grep "Model name" | awk -F: '{print $2}' | xargs)
    CPU_CORES=$(lscpu | grep "^CPU(s):" | awk '{print $2}')
    log_status "OK" "CPU: $CPU_MODEL, $CPU_CORES cores"
else
    log_status "FAIL" "Unable to check CPU"
fi

# Check disk space
if command -v df >/dev/null 2>&1; then
    DISK_SPACE=$(df -h . | tail -n 1 | awk '{print $4}')
    log_status "OK" "Available disk space: $DISK_SPACE"
else
    log_status "FAIL" "Unable to check disk space"
fi

echo "Setup completed. Check $LOG_FILE for details." >&3