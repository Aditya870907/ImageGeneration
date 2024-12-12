# Enhanced Stable Diffusion Client/Server System
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-success.svg)](https://fastapi.tiangolo.com)
[![Gradio](https://img.shields.io/badge/Gradio-3.x-orange.svg)](https://gradio.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<div align="center">

![Stable Diffusion Logo](https://img.shields.io/badge/🎨%20Stable%20Diffusion-Client%20Server-brightgreen?style=for-the-badge)

**A modern, user-friendly interface for Stable Diffusion image generation with advanced features**

[Installation](#server-setup) •
[Features](#features) •
[Documentation](#client-usage) •
[Contributing](#development)

</div>

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/stable-diffusion-client
cd stable-diffusion-client

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# Start the server
python server.py

# In a new terminal, start the client
python client.py
```

Visit `http://localhost:7860` to access the web interface.

---

## Overview
A comprehensive Stable Diffusion interface system combining a FastAPI server with a Gradio web client. This system provides advanced image generation capabilities, model management, and image-to-video conversion features through an intuitive user interface.

## Features
- 🎨 **Interactive Web Interface**: Intuitive Gradio-based UI with organized tabs and controls
- 🤖 **Model Management**: 
  - Support for both local `.safetensors` models and online Hugging Face models
  - Model comparison capabilities
  - Automatic model scanning and loading
- 🖼️ **Image Generation**:
  - Batch processing support
  - Custom scheduler configurations
  - Adjustable generation parameters
  - Real-time status updates
- 🎥 **Image-to-Video Conversion**:
  - Multiple animation presets
  - Region-based animation (face, body, background)
  - Customizable motion types
  - Duration and frame control
- 📊 **Advanced Features**:
  - Side-by-side model comparison
  - Automatic memory optimization
  - Comprehensive metadata tracking
  - Progress monitoring
- 💾 **Project Management**:
  - Organized output structure
  - Metadata preservation
  - Prompt set management

## Server Setup

### Requirements
- Python 3.8+
- CUDA-capable GPU
- PyTorch with CUDA support
- FastAPI and dependencies

### Installation
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Starting the Server
```bash
python server.py
```

The server will:
1. Scan for local models in the `models` directory
2. Display available models (local and online)
3. Prompt for model selection
4. Initialize the selected model
5. Start the FastAPI server on port 8001

### Model Directory Structure
```
models/
├── model1.safetensors    # Local model file
├── model1.json          # Optional metadata
├── model2.safetensors
└── model2.json
```

### Model Metadata Format
```json
{
    "model_type": "SD",     // or "SDXL"
    "base_model": "SD 1.5",
    "description": "Model description",
    "merged_from": ["model1", "model2"]
}
```

## Client Usage

### Starting the Client
```bash
python client.py [--port PORT] [--share] [--debug]
```

### Interface Tabs

#### 1. 🔌 Connection
- Server URL configuration (default: http://localhost:8001)
- Model refresh functionality
- Connection status monitoring

#### 2. 📁 Project
- Prompt set management through YAML files
- Prompt set selection
- Configuration status display

#### 3. ⚙️ Settings
- Model selection (single or multiple for comparison)
- Image dimensions (256-1024 pixels)
- Generation steps (1-100)
- Guidance scale (1-20)
- Scheduler configuration
- Batch size control
- Output directory management

#### 4. ✏️ Prompt
- Main prompt input
- Negative prompt input
- Real-time validation

#### 5. 🖼️ Output
- Generation controls
- Progress monitoring
- Image gallery
- Status updates

#### 6. 🎥 Image to Video
- Animation presets:
  - Subtle: 20 frames, 2 seconds
  - Normal: 24 frames, 2 seconds
  - Slow: 40 frames, 8 seconds
  - Ultra slow: 40 frames, 12 seconds
- Region selection
- Motion type configuration
- Custom output settings

### Generation Parameters
```python
{
    "prompt": str,
    "negative_prompt": str = "",
    "width": int = 512,          # 384-2048
    "height": int = 512,         # 384-2048
    "num_steps": int = 30,       # 1-150
    "guidance_scale": float = 7.5, # 1.0-20.0
    "scheduler_type": str = "dpmsolver++",
    "karras_sigmas": bool = True,
    "enable_attention_slicing": bool = True,
    "enable_vae_slicing": bool = True,
    "enable_vae_tiling": bool = True
}
```

## API Endpoints

### Server API
- `POST /generate`: Generate images with specified parameters
- `GET /models`: List available models
- `GET /health`: Check server status and GPU information
- `POST /compare`: Generate images with multiple models

### Health Check Response
```json
{
    "status": "ok",
    "cuda_available": true,
    "model_loaded": true,
    "current_model": {
        "name": "Model Name",
        "type": "SD/SDXL",
        "base_model": "Base Model Info",
        "default_size": 512
    },
    "gpu_info": {
        "name": "GPU Name",
        "total_memory_gb": "16.00",
        "used_memory_gb": "4.00",
        "free_memory_gb": "12.00"
    }
}
```

## Output Structure
```
outputs/
├── images/
│   ├── prefix_model_timestamp.png
│   ├── prefix_model_timestamp_0.png
│   └── ...
└── metadata/
    ├── prefix_model_timestamp.yaml
    └── prefix_model_timestamp_0.yaml
```

## Best Practices

### Performance Optimization
1. Monitor GPU memory through the health endpoint
2. Use batch sizes appropriate for your GPU
3. Enable memory optimizations for large models
4. Consider model type when selecting resolution

### Generation Tips
1. Start with default settings
2. Use comparison mode to evaluate models
3. Save successful prompts
4. Monitor generation status
5. Select appropriate animation presets

### Error Handling
- Verify server connection before generation
- Monitor generation status
- Check error messages in status area
- Use appropriate model for desired resolution

## Development

### Contributing
1. Fork the repository
2. Create a feature branch
3. Implement changes
4. Submit a pull request

### Building From Source
```bash
git clone https://github.com/SikamikanikoBG/ImageGenerator
cd stable-diffusion-client
pip install -r requirements.txt
```

## License
[MIT License](LICENSE)

## Acknowledgments
- Stability AI for Stable Diffusion
- Hugging Face for model distribution
- Gradio team for the UI framework
- FastAPI team for the server framework