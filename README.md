# FastHunyuan RunPod Worker

This is a RunPod worker implementation for FastHunyuan, a fast inference version of the Hunyuan text-to-video model.

## Development Guide

### Prerequisites

- RunPod account with API access
- NVIDIA GPU with CUDA support
- (Optional) VSCode for remote development

### Development Setup

1. Create a development pod on RunPod:

   - Go to [RunPod Console](https://www.runpod.io/console/pods)
   - Click "Deploy"
   - Select base image: `runpod/pytorch:2.1.2-py3.10-cuda12.1.0` (includes SSH and JupyterLab)
   - Choose your GPU type
   - Name your pod (e.g., "fasthunyuan-dev")
   - Deploy

2. Connect to your pod (choose one method):

   a. Using RunPod Web Terminal:

   - Click "Connect" on your pod in RunPod console
   - Select "SSH Terminal" or use the web terminal

   b. Using VSCode (Recommended for development):

   - Follow the [official RunPod guide for VSCode setup](https://docs.runpod.io/tutorials/pods/connect-to-vscode)
   - This allows you to develop directly on the pod with full IDE support

3. Install dependencies:

   ```bash
   cd /workspace
   chmod +x builder/install.sh
   ./builder/install.sh
   ```

4. Test the handler:
   ```bash
   python src/handler.py
   ```

### Development Tips

- The handler is located in `src/handler.py`
- Dependencies are managed in `builder/requirements.txt`
- System dependencies and installation steps are in `builder/install.sh`
- Using VSCode remote development provides a better development experience with:
  - Integrated terminal
  - Code completion
  - Debugging capabilities
  - Git integration

## Production Guide

### Building the Docker Image

1. Build the image:

   ```bash
   docker build -t your-registry/fasthunyuan-worker:version .
   ```

   Note: For production, we use a minimal base image (`nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04`) without development tools.

2. Push to your registry:
   ```bash
   docker push your-registry/fasthunyuan-worker:version
   ```

### Deploying on RunPod

1. Create a template on RunPod:

   - Base image: your-registry/fasthunyuan-worker:version
   - Container disk: at least 10GB
   - HTTP endpoints: enabled
   - Environment variables:
     ```
     HUGGING_FACE_HUB_TOKEN=your_token
     ```

2. Deploy serverless endpoints using the template

### API Usage

Example request:

```json
{
  "input": {
    "prompt": "A cinematic video of a beautiful landscape",
    "height": 720,
    "width": 1280,
    "num_frames": 45,
    "num_inference_steps": 6,
    "seed": 1024,
    "fps": 24
  }
}
```

Example response:

```json
{
  "video_path": "/tmp/output_1024.mp4"
}
```

## Project Structure

```
.
├── builder/
│   ├── install.sh        # Development installation script
│   └── requirements.txt  # Python dependencies
├── src/
│   └── handler.py        # RunPod handler implementation
├── Dockerfile           # Production container definition
└── README.md           # This file
```

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
