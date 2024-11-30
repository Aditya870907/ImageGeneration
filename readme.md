# Stable Diffusion Client/Server System

## Overview
This system consists of two main components:
1. A FastAPI server that handles the Stable Diffusion model loading and image generation
2. A Python client that manages project configurations and communicates with the server

## Server Setup

### Requirements
- Python 3.8+
- PyTorch with CUDA support
- FastAPI
- Diffusers
- All dependencies listed in requirements.txt

### Starting the Server
```bash
python server.py
```

The server will:
1. Scan for local models in the `models/` directory
2. Present a selection of available models (both local and online)
3. Initialize the selected model
4. Start listening on port 8001

### Model Directory Structure
```
models/
└── your_model.safetensors
└── your_model.json  # Optional model info file
```

The JSON file can contain additional model information:
```json
{
    "model_type": "SD" or "SDXL",
    "base_model": "SD 1.5",
    "description": "Model description",
    "merged_from": ["model1", "model2"]
}
```

## Client Usage

### Project Structure
```
projects/
└── your_project/
    ├── config.yaml
    └── prompts.yaml
```

### Configuration Files

#### config.yaml
```yaml
server_url: "http://192.168.1.101:8001"
name: "project_name"

# Default generation parameters
default_params:
  width: 512
  height: 512
  num_steps: 30
  guidance_scale: 7.5
  scheduler_type: "dpmsolver++"
  karras_sigmas: true
  scheduler_scale: 0.7
  clip_skip: 2
  enable_attention_slicing: true
  enable_vae_slicing: true
  enable_vae_tiling: false
  enable_model_cpu_offload: true
  enable_sequential_cpu_offload: false

# Ollama configuration for dynamic prompts
ollama:
  server_url: "http://localhost:11434"
  model: "llama2"
  system_message: "You are an expert at creating image generation prompts. Generate a detailed, creative prompt that will result in a visually stunning image. The prompt must be exactly 77 tokens long."
  prompt: "Create a detailed and creative image generation prompt that captures an interesting scene or concept."
  negative_prompt: "bad quality, low resolution, blurry"
  fallback_prompt: "A high quality, detailed masterpiece showing a beautiful landscape with mountains and trees."
```

#### prompts.yaml
```yaml
prompt_sets:
  - name: "landscape"
    prompt: "A beautiful mountain landscape with a lake and trees"
    negative_prompt: "ugly, blurry, low quality"
    params:
      width: 768
      height: 512
```

### Basic Commands

1. List available prompts in a project:
```bash
python client.py --input projects --project your_project --list
```

2. Generate an image using a predefined prompt:
```bash
python client.py --input projects --project your_project --prompt landscape
```

3. Generate multiple images (batch mode):
```bash
python client.py --input projects --project your_project --prompt landscape --batch 5
```

4. Use Ollama for dynamic prompt generation:
```bash
python client.py --input projects --project your_project --ollama
```

5. Generate multiple images with Ollama:
```bash
python client.py --input projects --project your_project --ollama --batch 5
```

### Output Structure
```
outputs/
└── project_name/
    └── prompt_name/
        └── timestamp_model/
            ├── images/
            │   └── image_timestamp.png
            └── metadata/
                └── metadata_timestamp.yaml
```

Each generation produces:
- The generated image(s)
- Metadata files containing:
  - Timestamp
  - Used prompt
  - Model settings
  - Generation parameters
  - Seed value

## Notes
- When using --ollama, the system will generate dynamic prompts using the LLM specified in config.yaml
- All prompts are automatically adjusted to be exactly 77 tokens
- The server automatically handles memory management and model loading
- Images and metadata are saved with timestamps for easy tracking
- The server provides a /health endpoint for monitoring system status

## Error Handling
- The client will validate all configuration files before running
- Missing or invalid configurations will result in clear error messages
- The server includes automatic memory management and error recovery
- Failed generations will be logged with detailed error information