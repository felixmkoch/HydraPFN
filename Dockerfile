FROM nvidia/cuda:12.8.0-devel-ubuntu22.04
#FROM nvidia/cuda:11.8.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app
COPY . .

RUN apt-get update && \
    apt-get install -y dos2unix && \
    apt update && \
    apt upgrade -y && \
    apt install git -y && \
    apt install pip -y && \
    apt update && \
    apt install vim -y 

# Pip PyTorch compatible with CUDA
#pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
#pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
RUN pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128

# Install requirements
RUN pip install -r requirements.txt

RUN pip install wandb==0.25.0

RUN pip install causal-conv1d>=1.4.0

RUN pip install mamba-ssm==2.3.0
