import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DDIMScheduler, EulerDiscreteScheduler, EulerAncestralDiscreteScheduler
from flask import Flask, request, jsonify
import base64
import io
import random
import os
import logging
from PIL import Image
from model_detection import ModelDetector
from pathlib import Path
import time

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
current_model = None
pipe = None
model_cache = {}
available_models = []

# Server configuration
MODEL_BASE_DIR = "/content/drive/MyDrive/ImageGenerator/models"

def initialize_server():
    """Initialize the server by detecting models"""
    global available_models
    
    logger.info("Initializing Stable Diffusion server...")
    
    # Ensure directories exist
    os.makedirs(MODEL_BASE_DIR, exist_ok=True)
    
    # Detect available models
    available_models = ModelDetector.scan_models_directory(MODEL_BASE_DIR)
    
    logger.info(f"Detected {len(available_models)} models: {[m['name'] for m in available_models]}")
    
    # Load the default model if available
    default_model_path = Path(MODEL_BASE_DIR) / "Stable-diffusion" / "v1-5-pruned-emaonly.safetensors"
    if default_model_path.exists():
        load_model(str(default_model_path))
    elif available_models:
        load_model(available_models[0]["id"])
    else:
        logger.warning("No models found for initialization")

def get_scheduler(scheduler_name, pipe):
    """Return the appropriate scheduler based on name"""
    if scheduler_name == "dpmsolver++":
        return DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "euler":
        return EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "euler_a":
        return EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name == "ddim":
        return DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        logger.warning(f"Unknown scheduler {scheduler_name}, using default")
        return DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

def load_model(model_id):
    """Load a Stable Diffusion model"""
    global pipe, current_model
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    try:
        # Find model info
        model_info = None
        for model in available_models:
            if model["id"] == model_id:
                model_info = model
                break
        
        if not model_info:
            logger.warning(f"Model {model_id} not found in available models")
            return False
            
        logger.info(f"Loading model: {model_info['name']}")
        start_time = time.time()
        
        # Load the model
        pipe = StableDiffusionPipeline.from_single_file(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            use_safetensors=True
        )
        
        # Move to device
        pipe = pipe.to(device)
        
        # Optimization for memory
        if device == "cuda":
            pipe.enable_attention_slicing()
        
        current_model = model_info
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        return True
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "current_model": current_model,
        "num_models": len(available_models)
    })

@app.route("/models", methods=["GET"])
def get_models():
    """Return available models"""
    return jsonify({"models": available_models})

@app.route("/generate", methods=["POST"])
def generate_image():
    """Generate an image based on input parameters"""
    global pipe, current_model
    
    if pipe is None:
        return jsonify({"error": "No model loaded"}), 500
    
    try:
        # Get parameters
        data = request.json
        prompt = data.get("prompt", "")
        negative_prompt = data.get("negative_prompt", "")
        width = data.get("width", 512)
        height = data.get("height", 512)
        num_steps = data.get("num_steps", 30)
        guidance_scale = data.get("guidance_scale", 7.5)
        seed = data.get("seed", random.randint(1, 2147483647))
        scheduler_type = data.get("scheduler_type", "dpmsolver++")
        
        # Check if we need to switch models
        model_id = data.get("model_id")
        if model_id and model_id != current_model["id"]:
            success = load_model(model_id)
            if not success:
                return jsonify({"error": "Failed to load model"}), 500
        
        # Set scheduler
        pipe.scheduler = get_scheduler(scheduler_type, pipe)
        
        # Set random seed
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        
        logger.info(f"Generating image with prompt: '{prompt[:50]}...' (size: {width}x{height}, steps: {num_steps})")
        start_time = time.time()
        
        # Generate image
        with autocast("cuda" if torch.cuda.is_available() else "cpu"):
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                generator=generator
            ).images[0]
        
        # Convert to base64
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        generation_time = time.time() - start_time
        logger.info(f"Image generated in {generation_time:.2f} seconds")
        
        return jsonify({
            "image_base64": image_base64,
            "seed": seed,
            "generation_time": generation_time
        })
    
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/switch_model", methods=["POST"])
def switch_model():
    """Switch to a different model"""
    data = request.json
    model_id = data.get("model_id")
    
    if not model_id:
        return jsonify({"error": "No model_id provided"}), 400
    
    success = load_model(model_id)
    if success:
        return jsonify({"status": "ok", "current_model": current_model})
    else:
        return jsonify({"error": "Failed to load model"}), 500

# Initialize and start the server
if __name__ == "__main__":
    initialize_server()
    app.run(host="0.0.0.0", port=8001)