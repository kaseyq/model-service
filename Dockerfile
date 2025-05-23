FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

     WORKDIR /app

     # Install system dependencies
     RUN apt-get update && apt-get install -y \
         python3 \
         python3-pip \
	 libsndfile1 \
         && rm -rf /var/lib/apt/lists/*

     # Copy requirements and install
     COPY requirements_docker.txt .
     RUN pip3 install --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu121 -r requirements_docker.txt

     # Copy application code
     COPY . .

     # Command to run the server
     CMD ["python3", "."]
