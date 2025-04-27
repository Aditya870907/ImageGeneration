#!/bin/bash

echo "🔧 Setting up environment for Stable Diffusion..."

echo "📦 Installing only the essential dependencies..."
pip install --no-deps diffusers==0.19.3 safetensors==0.3.2

# Install FastAPI with minimal dependencies
pip install --no-deps fastapi uvicorn==0.23.2 python-multipart

# Install Gradio with minimal dependencies
pip install --no-deps gradio==3.40.1 pyyaml

echo "✅ Environment setup complete!"
echo "Note: Some dependency conflicts may appear but shouldn't affect functionality"