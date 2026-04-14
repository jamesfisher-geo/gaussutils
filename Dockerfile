FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and minimal runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip python3-dev python-is-python3 \
    libxcb1-dev libx11-dev libgl-dev libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    python -m pip install --upgrade pip

# Install fvdb-core + fvdb-reality-capture from pre-built wheels (PyTorch 2.8.0 + CUDA 12.8)
RUN pip install \
    fvdb-reality-capture \
    fvdb-core==0.4.0+pt210.cu130 \
    torch==2.10.0 \
    --extra-index-url="https://d36m13axqqhiit.cloudfront.net/simple" \
    --extra-index-url https://download.pytorch.org/whl/cu130

# Install gaussutils
COPY . /gaussutils
RUN pip install /gaussutils