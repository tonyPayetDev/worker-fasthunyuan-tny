""" A template for a handler file. """

import os
import sys

# Add FastVideo to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "FastVideo"))

import torch
from transformers import BitsAndBytesConfig
from fastvideo.models.hunyuan.diffusion.pipelines import HunyuanVideoPipeline
from fastvideo.models.hunyuan.modules import HunyuanVideoTransformer3DModel
import runpod
from typing import Dict, Any


def export_to_video(frames, path, fps=24):
    """Export frames to video file"""
    import cv2
    import numpy as np

    os.makedirs(os.path.dirname(path), exist_ok=True)
    frames = [(frame * 255).astype(np.uint8) for frame in frames]

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def init_model():
    """Initialize the FastHunyuan model with NF4 quantization"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "/runpod/cache/model/NERDDISCO/FastHunyuan/main"

    # NF4 quantization config
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        llm_int8_skip_modules=["proj_out", "norm_out"],
    )

    # Load transformer with quantization
    transformer = HunyuanVideoTransformer3DModel.from_pretrained(
        model_id,
        subfolder="transformer/",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )

    # Initialize pipeline with CPU offload
    pipe = HunyuanVideoPipeline.from_pretrained(
        model_id, transformer=transformer, torch_dtype=torch.bfloat16
    )

    # Enable optimizations
    pipe.scheduler._shift = 17  # Default flow shift value
    pipe.vae.enable_tiling()
    pipe.enable_model_cpu_offload()

    return pipe


def handler(job: Dict[str, Any]):
    """Handler function for video generation with FastHunyuan"""
    job_input = job["input"]

    # Get parameters from input with defaults
    prompt = job_input.get("prompt", "A cinematic video of a beautiful landscape")
    height = job_input.get("height", 720)
    width = job_input.get("width", 1280)
    num_frames = job_input.get("num_frames", 45)
    num_inference_steps = job_input.get("num_inference_steps", 6)
    seed = job_input.get("seed", 1024)
    fps = job_input.get("fps", 24)

    # Initialize model
    pipe = init_model()

    # Set up prompt template
    prompt_template = {
        "template": (
            "<|start_header_cid|>system<|end_header_id|>\n\n"
            "Describe the video by detailing the following aspects: "
            "1. The main content and theme of the video."
            "2. The color, shape, size, texture, quantity, text, and spatial relationships of the contents, including objects, people, and anything else."
            "3. Actions, events, behaviors temporal relationships, physical movement changes of the contents."
            "4. Background environment, light, style, atmosphere, and qualities."
            "5. Camera angles, movements, and transitions used in the video."
            "6. Thematic and aesthetic concepts associated with the scene, i.e. realistic, futuristic, fairy tale, etc<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
        ),
        "crop_start": 95,
    }

    # Generate video
    generator = torch.Generator("cpu").manual_seed(seed)
    output = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_frames=num_frames,
        prompt_template=prompt_template,
        num_inference_steps=num_inference_steps,
        generator=generator,
    ).frames[0]

    # Save video
    output_path = f"/tmp/output_{seed}.mp4"
    export_to_video(output, output_path, fps=fps)

    # Return the video file path
    return {"video_path": output_path}


runpod.serverless.start({"handler": handler})
