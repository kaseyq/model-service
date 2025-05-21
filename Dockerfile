FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu20.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Command to run the server
CMD ["python3", "model_server.py"]