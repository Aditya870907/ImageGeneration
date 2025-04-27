# Stable Diffusion Client-Server Application

A Python-based application for generating images with Stable Diffusion models. This project includes a Flask-based server that manages the model loading and image generation, and a Gradio-based client interface for easy interaction.

## Features

- **Model Detection**: Automatically detects and catalogs Stable Diffusion models in specified directories
- **Client Interface**: User-friendly Gradio web interface for prompt input and image generation
- **Server Backend**: Flask server that handles model loading and image generation requests
- **Support for Multiple Models**: Ability to switch between different Stable Diffusion models
- **Customizable Generation Parameters**: Control over width, height, steps, guidance scale, and scheduler
- **Optimized for GPU**: Uses CUDA when available for faster generation

## Components

The application consists of three main Python files:

1. **server.py**: Flask server that loads models and handles image generation
2. **client.py**: Gradio-based user interface for generating images
3. **model_detection.py**: Utility for detecting and cataloging Stable Diffusion models

## Requirements

- Python 3.7+
- PyTorch with CUDA (for GPU acceleration)
- diffusers
- transformers
- Flask
- Gradio
- PIL

## Setup

1. Clone this repository
2. Install required dependencies: `pip install torch diffusers transformers flask gradio pillow`
3. Ensure your Stable Diffusion models are organized in appropriate directories
4. Update the `MODEL_BASE_DIR` in server.py to point to your models directory
5. Run the server: `python server.py`
6. Run the client: `python client.py`

## Usage

1. Start the server and client
2. Connect the client to the server (default: http://localhost:8001)
3. Select a model from the dropdown
4. Enter a prompt describing the image you want to generate
5. Optional: Add a negative prompt to specify what to avoid
6. Adjust generation parameters as needed
7. Click "Generate Image" button
8. The generated image will appear and be saved in the output directory

## Model Organization

The application looks for models in the following directories:
- Stable-diffusion, SD, stable-diffusion (for standard SD models)
- SDXL, SD-XL, stable-diffusion-xl (for SDXL models)

Models should be in .safetensors, .ckpt, or .pt format.

## License
MIT
## Acknowledgements

This application uses Hugging Face's diffusers library and Stable Diffusion models.
