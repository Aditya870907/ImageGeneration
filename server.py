import os
from pathlib import Path
# Set memory management environment variables
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import uvicorn
from pydantic import BaseModel, Field
import base64
from io import BytesIO
import logging
from typing import Optional
import gc
import inquirer
from contextlib import asynccontextmanager
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the pipeline and model info globally
pipe = None
current_model = None

def clear_memory():
    """Helper function to clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

class GenerationRequest(BaseModel):
    # Basic parameters
    prompt: str
    negative_prompt: str = ""
    
    # Image parameters
    width: int = Field(default=512, ge=384, le=2048)
    height: int = Field(default=512, ge=384, le=2048)
    
    # Generation parameters
    num_steps: int = Field(default=30, ge=1, le=150)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    
    # Advanced parameters
    num_images: int = Field(default=1, ge=1, le=4)
    seed: Optional[int] = None
    clip_skip: Optional[int] = Field(default=None, ge=1, le=4)
    
    # Scheduler parameters
    scheduler_type: str = Field(
        default="dpmsolver++",
        pattern="^(dpmsolver\+\+|euler_a|pndm|ddim|lms)$"
    )
    karras_sigmas: bool = True
    scheduler_scale: float = Field(default=0.7, ge=0.1, le=1.0)
    
    # Model behavior
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_model_cpu_offload: bool = True
    enable_sequential_cpu_offload: bool = False

class GenerationResponse(BaseModel):
    image_base64: str
    seed: int
    generation_settings: dict

def read_model_info(model_path: Path) -> dict:
    """Read model info from accompanying JSON file if it exists"""
    json_path = model_path.with_suffix('.json')
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read model info from {json_path}: {e}")
    return {}

def scan_local_models(models_dir="models"):
    """Scan for local .safetensors model files"""
    local_models = []
    models_path = Path(models_dir)
    
    if not models_path.exists():
        logger.warning(f"Models directory {models_dir} not found. Only using online models.")
        return local_models
        
    for model_file in models_path.glob("**/*.safetensors"):
        # Get model info from JSON if available
        model_info = read_model_info(model_file)
        file_size = model_file.stat().st_size / (1024 * 1024 * 1024)  # Convert to GB
        
        # Determine model type from info or name
        model_type = 'SD'  # Default to SD
        if model_info.get('model_type'):
            model_type = model_info['model_type']
        elif any(x in model_file.stem.lower() for x in ['xl', 'sdxl', 'sd-xl']):
            model_type = 'SDXL'
        
        # Set default size based on model type
        default_size = 1024 if model_type == 'SDXL' else 512
        
        logger.info(f"Found model {model_file.stem} - Type: {model_type}, Size: {file_size:.2f}GB")
        
        local_models.append({
            'name': f"Local: {model_file.stem}",
            'id': str(model_file.absolute()),
            'type': model_type,
            'default_size': default_size,
            'is_local': True,
            'base_model': model_info.get('base_model', 'Unknown'),
            'merged_models': model_info.get('merged_from', []),
            'description': model_info.get('description', '')
        })
        
    logger.info(f"Found {len(local_models)} local models in {models_dir}")
    return local_models

def select_model():
    """Interactive model selection"""
    # Online models
    online_models = [
        {
            'name': 'Stable Diffusion 1.4',
            'id': 'CompVis/stable-diffusion-v1-4',
            'type': 'SD',
            'default_size': 512,
            'is_local': False,
            'base_model': 'SD 1.4'
        },
        {
            'name': 'Hassan-SDXL',
            'id': 'hassanblend/Hassan-SDXL',
            'type': 'SDXL',
            'default_size': 1024,
            'is_local': False,
            'base_model': 'SDXL 1.0'
        },
        {
            'name': 'Stable Diffusion XL 1.0',
            'id': 'stabilityai/stable-diffusion-xl-base-1.0',
            'type': 'SDXL',
            'default_size': 1024,
            'is_local': False,
            'base_model': 'SDXL 1.0'
        }
    ]
    
    # Scan for local models
    local_models = scan_local_models()
    
    # Create selection list
    choices = []
    if local_models:
        choices.append('--- Local Models ---')
        for model in local_models:
            base_info = f" (Base: {model['base_model']})" if model['base_model'] != 'Unknown' else ""
            model['display_name'] = f"{model['name']}{base_info}"
            choices.append(model['display_name'])
    
    choices.append('--- Online Models ---')
    for model in online_models:
        model['display_name'] = f"{model['name']} (Base: {model['base_model']})"
        choices.append(model['display_name'])
    
    questions = [
        inquirer.List('model',
                     message="Select the model to load",
                     choices=choices)
    ]
    
    answers = inquirer.prompt(questions)
    selected_name = answers['model']
    
    # Skip separator entries
    if selected_name.startswith('---'):
        return None
        
    # Find selected model
    all_models = local_models + online_models
    selected = next(m for m in all_models if m.get('display_name', m['name']) == selected_name)
    
    # Log selected model info
    logger.info(f"Selected model: {selected['name']}")
    logger.info(f"Base model: {selected['base_model']}")
    logger.info(f"Type: {selected['type']}")
    if selected.get('description'):
        logger.info(f"Description: {selected['description']}")
    
    return selected

async def initialize_model(model_info):
    """Initialize the selected model"""
    global pipe, current_model
    
    try:
        clear_memory()
        logger.info(f"Initializing {model_info['name']}...")
        logger.info(f"Model type: {model_info['type']}, Base model: {model_info['base_model']}")
        
        # Log system info
        if torch.cuda.is_available():
            logger.info(f"CUDA available: True")
            logger.info(f"GPU count: {torch.cuda.device_count()}")
            logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Select pipeline based on model type
        PipelineClass = StableDiffusionXLPipeline if model_info['type'] == 'SDXL' else StableDiffusionPipeline
        
        logger.info(f"Loading model {model_info['id']}...")
        
        if model_info.get('is_local', False):
            if model_info['type'] == 'SDXL':
                pipe = PipelineClass.from_single_file(
                    model_info['id'],
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    requires_safety_checker=False
                )
            else:
                pipe = PipelineClass.from_single_file(
                    model_info['id'],
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    load_safety_checker=False
                )
        else:
            kwargs = {
                'torch_dtype': torch.float16,
                'use_safetensors': True,
                'variant': "fp16",
            }
            if model_info['type'] != 'SDXL':
                kwargs['safety_checker'] = None
            
            pipe = PipelineClass.from_pretrained(
                model_info['id'],
                **kwargs
            )
        
        # Ensure safety checker is disabled
        if hasattr(pipe, 'safety_checker'):
            pipe.safety_checker = None
        
        logger.info("Moving model to CUDA...")
        pipe = pipe.to("cuda")
        
        # Memory optimizations
        if hasattr(pipe, 'enable_attention_slicing'):
            pipe.enable_attention_slicing(slice_size=1)
        if hasattr(pipe, 'enable_vae_slicing'):
            pipe.enable_vae_slicing()
        if model_info['type'] == 'SDXL' and hasattr(pipe, 'enable_vae_tiling'):
            pipe.enable_vae_tiling()
        
        # Set up scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True
        )
        
        current_model = model_info
        clear_memory()
        logger.info("Pipeline initialization complete!")
        return True
        
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        raise RuntimeError(f"Failed to initialize pipeline: {str(e)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    while True:
        model_info = select_model()
        if model_info:
            try:
                await initialize_model(model_info)
                break
            except Exception as e:
                logger.error(f"Failed to initialize model: {e}")
                logger.info("Please select another model")
        else:
            logger.error("No valid model selected")
            continue
    
    yield
    # Shutdown
    clear_memory()

app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate_image(request: GenerationRequest):
    global pipe, current_model
    
    if pipe is None:
        logger.error("Pipeline is not initialized!")
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        logger.info(f"Configuring pipeline with request parameters...")
        clear_memory()
        
        # Adjust dimensions based on model type
        if request.width > current_model['default_size'] or request.height > current_model['default_size']:
            logger.warning(f"Adjusting dimensions to {current_model['default_size']}x{current_model['default_size']}")
            request.width = current_model['default_size']
            request.height = current_model['default_size']
        
        # Configure scheduler
        if request.scheduler_type == "dpmsolver++":
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=request.karras_sigmas
            )
        
        # Set random seed if provided
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device="cuda").manual_seed(request.seed)
        else:
            generator = torch.Generator(device="cuda").manual_seed(torch.randint(0, 2**32 - 1, (1,)).item())
        
        logger.info(f"Generating image with prompt: {request.prompt}")
        
        with torch.inference_mode():
            output = pipe(
                prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                num_inference_steps=request.num_steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
                num_images_per_prompt=1,
                generator=generator
            )
            
        if output.images is None or len(output.images) == 0:
            raise RuntimeError("No images were generated")
            
        # Get the seed that was used
        used_seed = generator.initial_seed()
            
        # Process and return first image
        image = output.images[0]
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Record generation settings
        generation_settings = {
            "model": current_model['name'],
            "base_model": current_model['base_model'],
            "seed": used_seed,
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "steps": request.num_steps,
            "guidance_scale": request.guidance_scale,
            "scheduler": request.scheduler_type,
            "dimensions": f"{request.width}x{request.height}",
            "scheduler_scale": request.scheduler_scale,
            "attention_slicing": request.enable_attention_slicing,
            "vae_slicing": request.enable_vae_slicing,
            "vae_tiling": request.enable_vae_tiling
        }
        
        logger.info("Image generated successfully")
        clear_memory()
        
        return GenerationResponse(
            image_base64=img_str,
            seed=used_seed,
            generation_settings=generation_settings
        )
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}", exc_info=True)
        clear_memory()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    if torch.cuda.is_available():
        memory_info = torch.cuda.mem_get_info()
        free_memory = memory_info[0] / 1024**3
        total_memory = memory_info[1] / 1024**3
        used_memory = total_memory - free_memory
    else:
        free_memory = total_memory = used_memory = 0
    
    return {
        "status": "ok", 
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": pipe is not None,
        "current_model": {
            "name": current_model['name'],
            "type": current_model['type'],
            "base_model": current_model['base_model'],
            "default_size": current_model['default_size'],
            "description": current_model.get('description', '')
        } if current_model else None,
        "gpu_info": {
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_memory_gb": f"{total_memory:.2f}",
            "used_memory_gb": f"{used_memory:.2f}",
            "free_memory_gb": f"{free_memory:.2f}"
        }
    }

if __name__ == "__main__":
    logger.info("Starting Stable Diffusion server...")
    uvicorn.run(app, host="0.0.0.0", port=8001)