# Stable Diffusion Client/Server System

## Overview
This system provides a robust client-server architecture for Stable Diffusion image generation:
1. A FastAPI server that handles model management and image generation
2. A Python client that manages project configurations and provides a user-friendly interface

## Key Features
- Support for both local and online Stable Diffusion models
- Dynamic prompt generation using Ollama
- Model comparison capabilities
- Batch processing
- Organized output management
- Automatic memory optimization
- Comprehensive error handling

## Server Setup

### Requirements
- Python 3.8+
- PyTorch with CUDA support
- FastAPI
- Diffusers
- Additional dependencies in requirements.txt

### Starting the Server
```bash
python server.py
```

The server will:
1. Scan the `models/` directory for local models
2. Present available models (both local and online)
3. Initialize the selected model
4. Start listening on port 8001

### Model Management

#### Directory Structure
```
models/
├── model1.safetensors
│   └── model1.json       # Optional model info
├── model2.safetensors
│   └── model2.json
└── ...
```

#### Model Information (JSON)
```json
{
    "model_type": "SD" or "SDXL",
    "base_model": "SD 1.5",
    "description": "Model description",
    "merged_from": ["model1", "model2"],
    "default_size": 512
}
```

## Client Usage

### Project Organization
```
projects/
└── your_project/
    ├── config.yaml      # Server and generation settings
    └── prompts.yaml     # Predefined prompts
```

### Configuration Files

#### config.yaml
```yaml
# Server configuration
server_url: "http://localhost:8001"
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
  
  # Performance optimizations
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
      guidance_scale: 8.0
```

### Command Reference

#### Basic Operations
```bash
# List available prompts
python client.py --input projects --project your_project --list

# Generate single image with predefined prompt
python client.py --input projects --project your_project --prompt landscape

# Generate multiple images (batch mode)
python client.py --input projects --project your_project --prompt landscape --batch 5

# Use Ollama for dynamic prompt generation
python client.py --input projects --project your_project --ollama

# Batch generation with Ollama
python client.py --input projects --project your_project --ollama --batch 5
```

#### Model Comparison
```bash
# Compare all models with single prompt
python client.py --input projects --project your_project --prompt landscape --compare

# Compare models with multiple iterations
python client.py --input projects --project your_project --prompt landscape --compare --batch 3
```

### Output Structure

```
outputs/
└── project_name/
    └── prompt_name/
        ├── timestamp_model/           # Single model output
        │   ├── images/
        │   │   └── image_timestamp.png
        │   └── metadata/
        │       └── metadata_timestamp.yaml
        │
        └── comparison_timestamp/      # Model comparison output
            ├── images/
            │   ├── model1_timestamp.png
            │   ├── model2_timestamp.png
            │   └── ...
            └── metadata/
                ├── model1_timestamp.yaml
                ├── model2_timestamp.yaml
                └── ...
```

#### Metadata Contents
Each generation produces a YAML file containing:
- Timestamp and seed value
- Used prompt and negative prompt
- Model information
- Generation parameters
- Performance settings

## Advanced Features

### Model Comparison
- Compare outputs across multiple models
- Use same prompt and parameters for fair comparison
- Generate side-by-side results
- Save comprehensive metadata for each model

### Dynamic Prompts
- Integration with Ollama for AI-generated prompts
- Automatic token length optimization (77 tokens)
- Quality-focused prompt enhancement
- Fallback handling for reliability

### Performance Optimization
- Automatic memory management
- VAE and attention optimizations
- Model offloading options
- Configurable batch processing

## Error Handling and Reliability

### Validation
- Configuration file validation
- Server health monitoring
- Model compatibility checks
- Parameter range verification

### Recovery
- Automatic retry mechanisms
- Graceful failure handling
- Detailed error logging
- Session recovery options

### Monitoring
- Server health endpoint
- Generation progress tracking
- Resource usage monitoring
- Batch progress indicators

## Best Practices
1. Start with recommended parameters in config.yaml
2. Use model comparison to find best model for your use case
3. Enable optimization flags based on your hardware
4. Monitor GPU memory usage for optimal batch sizes
5. Keep prompts consistent for valid comparisons
6. Regular server health checks during long runs