import os
from pathlib import Path
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DPMSolverMultistepScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import uvicorn
from pydantic import BaseModel, Field
import base64
from io import BytesIO
import logging
from typing import Optional, List, Dict
import gc
from contextlib import asynccontextmanager
from src.model_detection import ModelDetector

# Configure memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables
pipe = None
current_model = None
available_models = []

class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = Field(default=512, ge=384, le=2048)
    height: int = Field(default=512, ge=384, le=2048)
    num_steps: int = Field(default=30, ge=1, le=150)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)
    num_images: int = Field(default=1, ge=1, le=4)
    seed: Optional[int] = None
    clip_skip: Optional[int] = Field(default=None, ge=1, le=4)
    model_id: Optional[str] = None
    scheduler_type: str = Field(default="dpmsolver++", pattern="^(dpmsolver\+\+|euler_a|euler|ddim)$")
    karras_sigmas: bool = True
    scheduler_scale: float = Field(default=0.7, ge=0.1, le=1.0)
    enable_attention_slicing: bool = True
    enable_vae_slicing: bool = True
    enable_vae_tiling: bool = True
    enable_model_cpu_offload: bool = False
    use_gpu: bool = True

class GenerationResponse(BaseModel):
    image_base64: str
    seed: int
    generation_settings: dict
    model_info: dict

def clear_memory():
    """Clear GPU memory and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()

def get_all_models():
    """Get list of available models"""
    return ModelDetector.scan_models_directory("models")

async def initialize_model(model_info: Dict) -> bool:
    """Initialize the model pipeline with proper CLIP handling"""
    global pipe, current_model

    try:
        clear_memory()
        logger.info(f"Initializing {model_info['name']}...")

        is_xl = model_info['type'] == 'SDXL' or 'xl' in model_info['name'].lower()
        model_path = model_info['id']

        # Initialize text encoder first if not XL
        if not is_xl:
            try:
                text_encoder = CLIPTextModel.from_pretrained(
                    "openai/clip-vit-large-patch14",
                    torch_dtype=torch.float16
                ).to("cuda")
                tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
            except Exception as e:
                logger.warning(f"Failed to load CLIP components: {e}")
                text_encoder = None
                tokenizer = None

        # Initialize pipeline with appropriate configuration
        if is_xl:
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16"
            )
        else:
            pipe_kwargs = {
                "torch_dtype": torch.float16,
                "use_safetensors": True,
                "variant": "fp16"
            }
            
            if text_encoder is not None and tokenizer is not None:
                pipe_kwargs.update({
                    "text_encoder": text_encoder,
                    "tokenizer": tokenizer
                })
                
            pipe = StableDiffusionPipeline.from_single_file(model_path, **pipe_kwargs)

        # Move to CUDA
        pipe = pipe.to("cuda")

        # Enable optimizations
        pipe.enable_attention_slicing(slice_size="max")
        pipe.enable_vae_slicing()
        if is_xl:
            pipe.enable_vae_tiling()

        # Set scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="dpmsolver++",
            use_karras_sigmas=True
        )

        current_model = model_info
        clear_memory()
        logger.info(f"Successfully initialized {model_info['name']}")
        return True

    except Exception as e:
        logger.error(f"Model initialization error: {str(e)}", exc_info=True)
        clear_memory()
        raise RuntimeError(f"Failed to initialize pipeline: {str(e)}")

async def generate_with_model(request: GenerationRequest, model_info: Dict) -> GenerationResponse:
    """Generate image with improved error handling"""
    global pipe, current_model
    
    try:
        if pipe is None or current_model is None or current_model['id'] != model_info['id']:
            await initialize_model(model_info)

        # Handle dimensions
        is_xl = model_info['type'] == 'SDXL' or 'xl' in model_info['name'].lower()
        if request.width > model_info['default_size'] or request.height > model_info['default_size']:
            request.width = model_info['default_size']
            request.height = model_info['default_size']

        # Prepare generator
        generator = torch.Generator(device="cuda")
        if request.seed is not None:
            generator.manual_seed(request.seed)
        else:
            generator.manual_seed(torch.randint(0, 2**32 - 1, (1,)).item())

        try:
            with torch.inference_mode():
                try:
                    # Prepare generation parameters
                    generation_params = {
                        "prompt": request.prompt,
                        "negative_prompt": request.negative_prompt,
                        "num_inference_steps": request.num_steps,
                        "guidance_scale": request.guidance_scale,
                        "width": request.width,
                        "height": request.height,
                        "generator": generator,
                        "output_type": "pil",
                        "num_images_per_prompt": 1
                    }
                    
                    # Remove num_images_per_prompt for XL models as they don't support it
                    if is_xl:
                        generation_params.pop("num_images_per_prompt", None)
                    
                    output = pipe(**generation_params)
                    
                    if not hasattr(output, 'images') or len(output.images) == 0:
                        raise RuntimeError("No images were generated in the output")

                except Exception as gen_error:
                    logger.error(f"First generation attempt failed: {str(gen_error)}")
                    # Try again with simplified parameters
                    fallback_params = {
                        "prompt": request.prompt,
                        "negative_prompt": "",
                        "num_inference_steps": 20,
                        "guidance_scale": 7.5,
                        "width": 512 if not is_xl else 1024,
                        "height": 512 if not is_xl else 1024,
                        "generator": generator,
                        "output_type": "pil"
                    }
                    
                    output = pipe(**fallback_params)

                    if not hasattr(output, 'images') or len(output.images) == 0:
                        raise RuntimeError("No images were generated in fallback attempt")

            # Process output
            image = output.images[0]
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return GenerationResponse(
                image_base64=img_str,
                seed=generator.initial_seed(),
                generation_settings={
                    "model": model_info['name'],
                    "base_model": model_info['base_model'],
                    "seed": generator.initial_seed(),
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "steps": request.num_steps,
                    "guidance_scale": request.guidance_scale,
                    "dimensions": f"{request.width}x{request.height}"
                },
                model_info=model_info
            )

        except Exception as e:
            logger.error(f"Generation error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    except Exception as e:
        logger.error(f"Generation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

    finally:
        if request.model_id and current_model and request.model_id != current_model['id']:
            clear_memory()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the server"""
    global available_models, pipe, current_model
    
    try:
        available_models = get_all_models()
        logger.info(f"Found {len(available_models)} models")
        
        if not available_models:
            logger.warning("No models found. Server will start but won't be able to generate images.")
        else:
            for model in available_models:
                logger.info(f"Available model: {model['name']} ({model['base_model']})")
        
        pipe = None
        current_model = None
        logger.info("Server initialized and ready")
        
    except Exception as e:
        logger.error(f"Initialization error: {e}")
        raise

    yield
    clear_memory()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/generate")
async def generate_image(request: GenerationRequest):
    """Generate image endpoint"""
    global pipe, current_model
    
    try:
        # Case 1: Specific model requested
        if request.model_id:
            model_info = next((m for m in available_models if m['id'] == request.model_id), None)
            if not model_info:
                raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
            logger.info(f"Using requested model: {model_info['name']}")
            return await generate_with_model(request, model_info)
        
        # Case 2: Use current model if available
        elif current_model and pipe:
            logger.info(f"Using current model: {current_model['name']}")
            return await generate_with_model(request, current_model)
        
        # Case 3: No model loaded, use first available
        elif available_models:
            model_info = available_models[0]
            logger.info(f"Using first available model: {model_info['name']}")
            return await generate_with_model(request, model_info)
        
        # Case 4: No models available
        else:
            raise HTTPException(status_code=500, detail="No models available for generation")
            
    except Exception as e:
        logger.error(f"Generation error: {str(e)}", exc_info=True)
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def list_models():
    """List all available models"""
    return {"models": available_models}

@app.get("/health")
async def health_check():
    """Check server health and status"""
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
        },
        "available_models": len(available_models),
        "ready_for_generation": len(available_models) > 0
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Stable Diffusion Server')
    parser.add_argument('--host', type=str, default="0.0.0.0", help='Host to bind to')
    parser.add_argument('--port', type=int, default=8001, help='Port to bind to')
    parser.add_argument('--log-level', type=str, default="info",
                       choices=['debug', 'info', 'warning', 'error', 'critical'])
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)