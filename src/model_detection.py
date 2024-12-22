import os
from pathlib import Path
import json
import re
import logging
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)

class ModelDetector:
    # Common model naming patterns 
    MODEL_PATTERNS = {
        'pony': {
            'patterns': [r'pony', r'mlp', r'horse'],
            'base_model': 'Pony Diffusion',
            'type': 'SD'
        },
        'sdxl': {
            'patterns': [r'xl', r'sdxl', r'sd-xl'],
            'base_model': 'SDXL',
            'type': 'SDXL'
        },
        'anime': {
            'patterns': [r'anime', r'waifu', r'anything', r'nai'],
            'base_model': 'Anime Diffusion',
            'type': 'SD'
        },
        'inpainting': {
            'patterns': [r'inpaint', r'mask'],
            'base_model': 'Inpainting',
            'type': 'SD'
        }
    }

    # Model requirements configuration
    MODEL_REQUIREMENTS = {
        'stable_yogi': {
            'patterns': [r'stableYogi', r'yogis'],
            'required_embeddings': {
                'positive': 'Stable_Yogis_PDXL_Positives',
                'negative': 'Stable_Yogis_PDXL_Negatives-neg'
            },
            'prompt_info': '''
            Important Usage Tips:
            - Add Stable_Yogis_PDXL_Positives at the beginning of your prompt
            - Add Stable_Yogis_PDXL_Negatives-neg at the beginning of negative prompt
            '''
        }
        # Add other model-specific requirements here
    }

    @staticmethod
    def detect_model_requirements(model_name: str) -> Dict[str, Any]:
        """Detect any special requirements for the model based on its name"""
        model_name_lower = model_name.lower()
        requirements = {}

        for model_type, config in ModelDetector.MODEL_REQUIREMENTS.items():
            if any(re.search(pattern, model_name_lower) for pattern in config['patterns']):
                requirements.update({
                    k: v for k, v in config.items() 
                    if k not in ['patterns']  # Exclude internal patterns
                })

        return requirements

    @staticmethod
    def detect_model_type(model_path: Path) -> Dict:
        """
        Automatically detect model type and configuration from filename and metadata
        """
        model_info = {
            'name': model_path.stem,
            'id': str(model_path.absolute()),
            'type': 'SD',  # Default to SD
            'default_size': 512,  # Default size
            'is_local': True,
            'base_model': 'Unknown',
            'description': '',
            'requirements': {}  # Add requirements field
        }

        # Try to load existing metadata first
        json_path = model_path.with_suffix('.json')
        if json_path.exists():
            try:
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                    model_info.update(metadata)
                    logger.info(f"Loaded metadata from {json_path}")
                    return model_info
            except Exception as e:
                logger.warning(f"Failed to read model metadata from {json_path}: {e}")

        # Detect model type from filename
        model_name_lower = model_path.stem.lower()

        # Check for known model types
        for model_type, config in ModelDetector.MODEL_PATTERNS.items():
            if any(re.search(pattern, model_name_lower) for pattern in config['patterns']):
                model_info.update({
                    'base_model': config['base_model'],
                    'type': config['type']
                })
                if config['type'] == 'SDXL':
                    model_info['default_size'] = 1024
                break

        # Try to extract version information
        version_match = re.search(r'v(\d+(?:\.\d+)*)', model_name_lower)
        if version_match:
            model_info['version'] = version_match.group(1)

        # Detect any special requirements
        requirements = ModelDetector.detect_model_requirements(model_path.stem)
        if requirements:
            model_info['requirements'] = requirements

        # Generate description
        model_info['description'] = ModelDetector.generate_description(model_info)

        # Save detected metadata
        try:
            with open(json_path, 'w') as f:
                json.dump(model_info, f, indent=2)
            logger.info(f"Saved detected metadata to {json_path}")
        except Exception as e:
            logger.warning(f"Failed to save model metadata to {json_path}: {e}")

        return model_info

    @staticmethod
    def generate_description(model_info: Dict) -> str:
        """Generate a description based on detected model attributes"""
        desc_parts = []
        
        if model_info['base_model'] != 'Unknown':
            desc_parts.append(f"Based on {model_info['base_model']}")
        
        if model_info['type'] == 'SDXL':
            desc_parts.append("SDXL model")
        elif 'version' in model_info:
            desc_parts.append(f"Version {model_info['version']}")
            
        if model_info['default_size'] != 512:
            desc_parts.append(f"Default size {model_info['default_size']}x{model_info['default_size']}")

        # Add requirements info to description if present
        if 'requirements' in model_info and model_info['requirements']:
            if 'prompt_info' in model_info['requirements']:
                desc_parts.append(model_info['requirements']['prompt_info'])

        return " - ".join(desc_parts) if desc_parts else "No additional information available"

    @staticmethod
    def scan_models_directory(models_dir: str = "models") -> list:
        """Scan directory for models and automatically detect their configurations"""
        models = []
        models_path = Path(models_dir)

        if not models_path.exists():
            logger.warning(f"Models directory {models_dir} not found")
            return models

        # Scan for all supported model files
        model_files = list(models_path.glob("**/*.safetensors"))
        
        if not model_files:
            logger.warning(f"No .safetensors files found in {models_dir}")
            return models

        logger.info(f"Found {len(model_files)} model files")

        # Process each model file
        for model_file in model_files:
            try:
                model_info = ModelDetector.detect_model_type(model_file)
                models.append(model_info)
                logger.info(f"Detected model: {model_info['name']} ({model_info['base_model']})")
            except Exception as e:
                logger.error(f"Error processing model {model_file}: {e}")
                continue

        return models