FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

WORKDIR /workspace

# Copy requirements first to leverage Docker cache
COPY builder/requirements.txt /workspace/builder/requirements.txt

# Install Python packages
RUN pip install --upgrade pip && \
    pip install -r /workspace/builder/requirements.txt

# Install flash-attention with CUDA build skipped
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
RUN pip install packaging ninja && \
    pip install flash-attn==2.7.0.post2 --no-build-isolation

# Copy the rest of the application
COPY . /workspace

# Set up the entrypoint
CMD [ "python", "-u", "src/handler.py" ] 