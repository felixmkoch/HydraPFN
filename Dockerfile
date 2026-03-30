FROM nvidia/cuda:12.8.0-devel-ubuntu22.04
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

RUN pip install uv

# install project (replace HYDRAPFN_DIR_PATH with actual path)
RUN cd HYDRAPFN_DIR_PATH
RUN uv sync
RUN uv pip install -e .

# run script(s) (replace MY_SCRIPT with actual file name)
#RUN uv run MY_SCRIPT.py
