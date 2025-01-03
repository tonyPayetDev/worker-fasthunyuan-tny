#!/bin/bash
set -e  # Exit on error

echo "Installing CUDA dependencies..."
pip install --no-cache-dir torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu121

echo "Installing core dependencies..."
pip install --no-cache-dir \
    transformers==4.46.1 \
    diffusers==0.32.0 \
    accelerate==1.0.1 \
    safetensors==0.4.5 \
    peft==0.13.2 \
    packaging==24.2

echo "Installing video processing dependencies..."
pip install --no-cache-dir \
    albumentations==1.4.20 \
    av==13.1.0 \
    decord==0.6.0 \
    einops==0.8.0 \
    imageio==2.36.0 \
    imageio-ffmpeg==0.5.1 \
    moviepy==1.0.3 \
    numpy==1.26.3 \
    opencv-python==4.10.0.84 \
    opencv-python-headless==4.10.0.84 \
    scikit-video==1.1.11

echo "Installing other required dependencies..."
pip install --no-cache-dir \
    huggingface_hub==0.26.1 \
    omegaconf==2.3.0 \
    loguru \
    bitsandbytes

echo "Installing flash-attn with CUDA build skipped..."
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
pip install --no-cache-dir flash-attn==2.7.0.post2 --no-build-isolation

echo "Installing FastVideo..."
pip install --no-cache-dir git+https://github.com/hao-ai-lab/FastVideo.git@dd75ee8509943f2872b1b196d245afe1961c629b

echo "Installation complete!" 