# OPSD baseline reproduction image (Qwen3-1.7B).
# Pins the exact deps from environment.yml. Build on a machine with GPUs
# available at *run* time; the build itself only needs nvcc (devel base) for flash-attn.
#
#   docker build -t opsd:repro .
#
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        git wget ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Miniconda ----
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p /opt/conda \
    && rm /tmp/mc.sh
ENV PATH=/opt/conda/bin:$PATH

# ---- opsd conda env (python 3.10 + torch 2.8 / trl 0.26 / vllm 0.11 / deepspeed 0.18 ...) ----
# Accept the Anaconda default channel ToS (required by newer conda before env create).
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
COPY environment.yml /tmp/environment.yml
RUN conda env create -f /tmp/environment.yml && conda clean -afy

# All subsequent RUN commands (and the default runtime) execute inside the `opsd` env.
SHELL ["conda", "run", "--no-capture-output", "-n", "opsd", "/bin/bash", "-c"]

# ---- flash-attn (README: install separately, no build isolation) ----
# Source build works because this is a CUDA *devel* base (nvcc present) but is slow.
# If the build is too slow, replace the line below with the matching prebuilt wheel, e.g.:
#   RUN pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
RUN pip install flash-attn==2.8.3 --no-build-isolation

WORKDIR /workspace
COPY . /workspace

# HF model/dataset cache (mount a host volume here to persist downloads).
ENV HF_HOME=/workspace/.hf_cache

# Enter the container already inside the opsd env; append your command after the image name.
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "opsd"]
