# Use the NVIDIA CUDA 11.8 base image with Ubuntu 22.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# Install Python 3.10, git, wget, build tools, and pip
RUN apt-get update && apt-get install -y \
    software-properties-common \
    git \
    wget \
    build-essential \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    libavdevice-dev \
    libavfilter-dev \
    libavformat-dev \
    libavcodec-dev \
    libswscale-dev \
    pkg-config \
    ffmpeg \
    libavcodec-extra \
    libgl1 &&\
    rm -rf /var/lib/apt/lists/*

# Set python3.10 as default python and upgrade pip
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    python -m pip install --upgrade pip

# Clone the ReferDINO repository
RUN git clone https://github.com/diegomaureira/ReferDINO.git /ReferDINO
WORKDIR /ReferDINO

# Install PyTorch 2.5.1 with CUDA 11.8 support via pip
RUN python -m pip install torch==2.5.1+cu118 torchvision==0.20.1+cu118 torchaudio==2.5.1 --extra-index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
RUN python -m pip install -r requirements.txt

# Instead, add this in CMD or entrypoint script:
CMD ["bash", "-c", "cd models/GroundingDINO/ops && python setup.py build install && python test.py && cd ../../.. && bash"]