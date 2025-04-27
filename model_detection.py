import os
from pathlib import Path
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelDetector:
    """Utility class to detect and catalog Stable Diffusion models"""
    
    @staticmethod
    def scan_models_directory(base_dir):
        """Scan for models in the specified directory and return model info"""
        models = []
        base_path = Path(base_dir)
        
        if not base_path.exists():
            logger.warning(f"Models directory {base_dir} does not exist")
            return models
            
        # Define model directories to scan
        sd_dirs = ["Stable-diffusion", "SD", "stable-diffusion"]
        sdxl_dirs = ["SDXL", "SD-XL", "stable-diffusion-xl"]
        
        # Check standard folders
        for sd_dir in sd_dirs:
            model_dir = base_path / sd_dir
            if model_dir.exists():
                models.extend(ModelDetector._process_model_directory(model_dir, "SD", 512))
        
        for sdxl_dir in sdxl_dirs:
            model_dir = base_path / sdxl_dir
            if model_dir.exists():
                models.extend(ModelDetector._process_model_directory(model_dir, "SDXL", 1024))
        
        # If no models found in standard folders, search the base directory
        if not models:
            models.extend(ModelDetector._process_model_directory(base_path, "SD", 512))
            
        return models
    
    @staticmethod
    def _process_model_directory(directory, model_type, default_size):
        """Process a directory to identify model files"""
        models = []
        supported_extensions = [".safetensors", ".ckpt", ".pt"]
        
        try:
            for model_file in directory.glob("*"):
                if model_file.suffix.lower() in supported_extensions:
                    model_id = str(model_file)
                    model_name = model_file.stem
                    
                    # Guess the base model from the filename
                    base_model = "SD 1.5"
                    if "xl" in model_name.lower():
                        base_model = "SDXL"
                        model_type = "SDXL"
                        default_size = 1024
                    elif "sd2" in model_name.lower() or "v2" in model_name.lower():
                        base_model = "SD 2.1"
                    
                    # Create model info
                    model_info = {
                        "id": model_id,
                        "name": model_name,
                        "type": model_type,
                        "base_model": base_model,
                        "default_size": default_size,
                        "description": f"{model_name} - {base_model} model"
                    }
                    
                    models.append(model_info)
                    logger.info(f"Found model: {model_name} ({base_model})")
        except Exception as e:
            logger.error(f"Error processing model directory {directory}: {e}")
        
        return models