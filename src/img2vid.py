import argparse
import torch
from pathlib import Path
from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import numpy as np
import cv2  # Added missing import
from datetime import datetime
import json
import shutil
import warnings
import gc
import os

warnings.filterwarnings("ignore")

# Single consolidated region settings dictionary
REGION_SETTINGS = {
    'face': {
        'coordinates': [128, 128, 256, 256],
        'description': 'Center face area',
        'motions': ['subtle_movement', 'blink', 'expression'],
        'default_intensity': 0.3,
        'motion_scale': 10.0,
        'motion_types': {
            'smile': 'Gradual smile animation',
            'blink': 'Natural blinking motion',
            'look_left': 'Head turning slightly left',
            'look_right': 'Head turning slightly right',
            'nod': 'Slight nodding motion',
            'subtle': 'Very subtle facial movements',
        }
    },
    'tongue': {
        'coordinates': [128, 128, 256, 256],
        'description': 'Center face area',
        'motions': ['out', 'side', 'tip', 'subtle'],
        'default_intensity': 0.4,
        'motion_scale': 8.0,
        'motion_types': {
            'out': 'Tongue sticking out',
            'side': 'Tongue moving sideways',
            'tip': 'Tongue tip movement',
            'subtle': 'Subtle tongue motion',
        }
    },
    'face_large': {
        'coordinates': [64, 64, 384, 384],
        'description': 'Larger face and upper body area',
        'motions': ['subtle_movement', 'blink', 'expression'],
        'default_intensity': 0.3,
        'motion_scale': 10.0,
        'motion_types': {
            'smile': 'Gradual smile animation',
            'blink': 'Natural blinking motion',
            'look_around': 'Head moving slightly',
            'subtle': 'Very subtle movements',
        }
    },
    'upper_body': {
        'coordinates': [0, 128, 512, 256],
        'description': 'Upper third of image',
        'motions': ['wave', 'drift', 'subtle_movement'],
        'default_intensity': 0.5,
        'motion_scale': 12.0,
        'motion_types': {
            'wave': 'Gentle waving motion',
            'drift': 'Slow drifting movement',
            'subtle': 'Subtle natural movement',
        }
    },
    'lower_body': {
        'coordinates': [0, 384, 512, 128],
        'description': 'Lower third of image',
        'motions': ['wave', 'drift', 'subtle_movement'],
        'default_intensity': 0.5,
        'motion_scale': 12.0,
        'motion_types': {
            'wave': 'Gentle waving motion',
            'drift': 'Slow drifting movement',
            'subtle': 'Subtle natural movement',
        }
    },
    'full_body': {
        'coordinates': [64, 0, 384, 512],
        'description': 'Center vertical strip',
        'motions': ['wave', 'drift', 'subtle_movement'],
        'default_intensity': 0.5,
        'motion_scale': 12.0,
        'motion_types': {
            'wave': 'Gentle waving motion',
            'drift': 'Slow drifting movement',
            'subtle': 'Subtle natural movement',
        }
    },
    'custom': {
        'coordinates': None,
        'description': 'Enter custom coordinates',
        'motions': ['wave', 'drift', 'subtle_movement'],
        'default_intensity': 0.6,
        'motion_scale': 12.0,
        'motion_types': {
            'wave': 'Gentle waving motion',
            'drift': 'Slow drifting movement',
            'subtle': 'Subtle natural movement',
        }
    }
}


# Add to the animation presets at the top
ANIMATION_PRESETS = {
    'subtle': {
        'frames': 20,
        'duration': 100,
        'description': 'Subtle movement (2 seconds)'
    },
    'normal': {
        'frames': 24,
        'duration': 80,
        'description': 'Normal movement (2 seconds)'
    },
    'slow': {
        'frames': 40,  # More frames for smoother slow motion
        'duration': 200,  # Longer duration per frame
        'description': 'Slow motion (8 seconds)'
    },
    'ultra_slow': {
        'frames': 40,
        'duration': 300,  # Even longer duration
        'description': 'Ultra slow motion (12 seconds)'
    },
    'custom': {
        'frames': None,
        'duration': None,
        'description': 'Custom frames and duration'
    }
}






def get_region_coordinates(region_type):
    """Get region coordinates either from presets or user input."""
    if region_type == 'custom':
        try:
            print("\nEnter coordinates (or press Enter for defaults):")
            x = int(input("X position (0-512, default=128): ") or 128)
            y = int(input("Y position (0-512, default=128): ") or 128)
            w = int(input("Width (0-512, default=256): ") or 256)
            h = int(input("Height (0-512, default=256): ") or 256)
            return [x, y, w, h]
        except ValueError:
            print("Invalid input, using default values")
            return [128, 128, 256, 256]

    return REGION_SETTINGS[region_type]['coordinates']



class SessionManager:
    def __init__(self, base_output_dir):
        self.base_output_dir = Path(base_output_dir)
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_output_dir / f"session_{self.session_id}"
        self.metadata = {
            "session_id": self.session_id,
            "timestamp": datetime.now().isoformat(),
            "processed_images": []
        }
        self._setup_session_directories()

    def _setup_session_directories(self):
        """Create session directory structure"""
        self.session_dir.mkdir(parents=True, exist_ok=True)
        (self.session_dir / "gifs").mkdir(exist_ok=True)
        (self.session_dir / "frames").mkdir(exist_ok=True)
        (self.session_dir / "originals").mkdir(exist_ok=True)
        (self.session_dir / "masks").mkdir(exist_ok=True)

    def get_paths_for_image(self, image_name):
        """Generate paths for all outputs related to an image"""
        base_name = Path(image_name).stem
        return {
            "gif": self.session_dir / "gifs" / f"{base_name}_animated.gif",
            "frames_dir": self.session_dir / "frames" / base_name,
            "original": self.session_dir / "originals" / Path(image_name).name,
            "masks_dir": self.session_dir / "masks" / base_name
        }

    def save_metadata(self):
        """Save session metadata to JSON file"""
        metadata_path = self.session_dir / "session_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def add_processed_image(self, image_path, output_paths, settings):
        """Add processing record to metadata"""
        self.metadata["processed_images"].append({
            "original_image": str(image_path),
            "outputs": {k: str(v) for k, v in output_paths.items()},
            "settings": settings,
            "processing_time": datetime.now().isoformat()
        })
        self.save_metadata()


REGION_PRESETS = {
    'hair': {
        'motions': ['wave', 'flow', 'subtle_movement'],
        'default_intensity': 0.8,
        'description': 'Flowing or waving motion for hair',
        'motion_scale': 15.0
    },
    'face': {
        'motions': ['subtle_movement', 'blink', 'expression'],
        'default_intensity': 0.3,
        'description': 'Subtle facial movements and expressions',
        'motion_scale': 10.0
    },
    'clothing': {
        'motions': ['wave', 'flutter', 'subtle_movement'],
        'default_intensity': 0.5,
        'description': 'Fabric movement and flutter effects',
        'motion_scale': 12.0
    },
    'background': {
        'motions': ['drift', 'pan', 'subtle_movement'],
        'default_intensity': 0.4,
        'description': 'Background ambient motion',
        'motion_scale': 8.0
    },
    'custom': {
        'motions': ['wave', 'drift', 'subtle_movement'],
        'default_intensity': 0.6,
        'description': 'User-defined region with custom motion',
        'motion_scale': 12.0
    }
}


class MaskGenerator:
    def __init__(self, image_path):
        self.pil_image = Image.open(image_path)
        self.image = np.array(self.pil_image)
        self.height, self.width = self.image.shape[:2]
        print(f"Loaded image with size: {self.width}x{self.height}")

    def generate_region_mask(self, region_type, points=None):
        """Generate mask using menu-selected coordinates."""
        mask = np.zeros((self.height, self.width), dtype=np.uint8)

        # Pass the region_type directly to get_region_coordinates
        coords = get_region_coordinates(region_type)
        x, y, w, h = coords

        # Ensure coordinates are within bounds
        x = max(0, min(x, self.width))
        y = max(0, min(y, self.height))
        w = max(0, min(w, self.width - x))
        h = max(0, min(h, self.height - y))

        # Create mask
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        print(f"Created mask for region: x={x}, y={y}, width={w}, height={h}")

        return mask



class AnimationController:
    def __init__(self):
        self.regions = {}

    def add_region(self, name, mask, motion_type, intensity=1.0):
        self.regions[name] = {
            'mask': mask,
            'motion_type': motion_type,
            'intensity': intensity,
            'motion_scale': REGION_SETTINGS[name]['motion_scale']  # Updated to use REGION_SETTINGS
        }

    def combine_masks(self):
        """Combine all region masks with their respective intensities."""
        if not self.regions:
            return None

        first_mask = next(iter(self.regions.values()))['mask']
        combined_mask = np.zeros_like(first_mask, dtype=np.float32)

        for region in self.regions.values():
            mask = region['mask'].astype(np.float32) / 255.0
            combined_mask = np.maximum(combined_mask, mask * region['intensity'])

        return (combined_mask * 255).astype(np.uint8)


def clear_memory():
    """Clear CUDA memory and garbage collect."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    gc.collect()


def load_model():
    """Initialize the SVD pipeline with stable memory configuration."""
    clear_memory()

    # Set environment variables for better CUDA memory handling
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    torch.backends.cudnn.benchmark = True

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid",
        torch_dtype=torch.float16,
        variant="fp16"
    )

    # Use CPU offload without channels_last format
    pipe.enable_model_cpu_offload()

    # Set to eval mode
    pipe.unet.eval()
    pipe.vae.eval()

    # Optional: Enable attention slicing if memory is still an issue
    pipe.enable_attention_slicing()

    return pipe


def generate_animation(pipe, image_path, mask, controller, num_frames=20, slow_motion=False):
    """Generate animation frames with optional slow motion."""
    clear_memory()

    image = Image.open(image_path)
    target_h = 576
    target_w = int(target_h * (image.width / image.height))
    target_w = min(1024, target_w)
    target_size = (target_w, target_h)
    image = image.resize(target_size, Image.Resampling.LANCZOS)

    # Adjust settings for slow motion
    if slow_motion:
        num_inference_steps = 50  # Higher for smoother transitions
        noise_aug_strength = 0.02  # Lower for more detail preservation
    else:
        num_inference_steps = 30
        noise_aug_strength = 0.1

    try:
        with torch.inference_mode():
            frames = pipe(
                image,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames,
                motion_bucket_id=100,  # Lower for slower perceived motion
                noise_aug_strength=noise_aug_strength,
                generator=torch.manual_seed(42),
                decode_chunk_size=1
            ).frames[0]
    except RuntimeError as e:
        clear_memory()
        print("Falling back to standard settings...")
        frames = pipe(
            image,
            num_inference_steps=25,
            num_frames=num_frames,
            generator=torch.manual_seed(42),
            decode_chunk_size=1
        ).frames[0]

    clear_memory()
    return frames


def create_slow_motion_frames(frames_pil, slowdown_factor=2):
    """Create interpolated frames for slow motion effect."""
    slow_motion_frames = []

    for i in range(len(frames_pil) - 1):
        frame1 = np.array(frames_pil[i])
        frame2 = np.array(frames_pil[i + 1])

        # Create intermediate frames
        for j in range(slowdown_factor):
            alpha = j / slowdown_factor
            interpolated = cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)
            slow_motion_frames.append(Image.fromarray(interpolated))

    # Add the last frame
    slow_motion_frames.append(frames_pil[-1])
    return slow_motion_frames


def save_animation(frames, image_path, session_manager, regions_config, frame_duration=100, slowdown_factor=1):
    """Save animation with slow motion effect."""
    paths = session_manager.get_paths_for_image(image_path.name)

    paths["frames_dir"].mkdir(parents=True, exist_ok=True)
    paths["masks_dir"].mkdir(parents=True, exist_ok=True)

    # Convert to PIL images
    frames_pil = []
    for frame in frames:
        if isinstance(frame, np.ndarray):
            frames_pil.append(Image.fromarray(frame.astype(np.uint8)))
        elif isinstance(frame, torch.Tensor):
            frames_pil.append(Image.fromarray((frame.cpu().numpy() * 255).astype(np.uint8)))
        elif isinstance(frame, Image.Image):
            frames_pil.append(frame)

    # Resize to original size
    if frames_pil[0].size != (512, 512):
        frames_pil = [frame.resize((512, 512), Image.Resampling.LANCZOS) for frame in frames_pil]

    # Apply slow motion if requested
    if slowdown_factor > 1:
        frames_pil = create_slow_motion_frames(frames_pil, slowdown_factor)
        frame_duration = int(frame_duration * 0.7)  # Adjust duration for smooth playback

    # Save frames
    for idx, frame in enumerate(frames_pil):
        frame_path = paths["frames_dir"] / f"frame_{idx:03d}.png"
        frame.save(frame_path)

    # Save GIF
    frames_pil[0].save(
        paths["gif"],
        save_all=True,
        append_images=frames_pil[1:],
        duration=frame_duration,
        loop=0
    )

    shutil.copy2(image_path, paths["original"])

    print(f"\nSaved animation to: {paths['gif']}")
    print(f"Saved frames to: {paths['frames_dir']}")
    print(f"Animation length: {len(frames_pil)} frames at {frame_duration}ms per frame")
    print(f"Total duration: {(len(frames_pil) * frame_duration) / 1000:.1f} seconds")


def process_image_with_regions(pipe, image_path, session_manager, regions_config, num_frames=20, frame_duration=100):
    """Process image with region-specific animations."""
    try:
        clear_memory()

        image_path = Path(image_path)
        print(f"\nProcessing image: {image_path.absolute()}")

        mask_generator = MaskGenerator(image_path)
        animation_controller = AnimationController()

        # Use the motion type directly from the config
        for region_name, config in regions_config.items():
            print(f"\nConfiguring motion for {region_name}...")
            motion_type = config.get('motion_type')
            print(f"Using motion: {motion_type}")

            mask = mask_generator.generate_region_mask(
                region_name,
                config.get('points', None)
            )

            # Use the actual region name for adding to controller
            animation_controller.add_region(
                region_name,
                mask,
                motion_type,
                intensity=config.get('intensity', 1.0)
            )

        combined_mask = np.zeros((mask_generator.height, mask_generator.width), dtype=np.uint8)

        print("\nGenerating animation...")
        frames = generate_animation(pipe, image_path, combined_mask, animation_controller, num_frames=num_frames)
        save_animation(frames, image_path, session_manager, regions_config, frame_duration=frame_duration)

        clear_memory()

    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        clear_memory()
        raise


def main():
    parser = argparse.ArgumentParser(description="Generate animations using SVD")
    parser.add_argument("path", help="Path to image file")
    parser.add_argument("--output", default="outputs", help="Output directory path")
    args = parser.parse_args()

    clear_memory()

    try:
        # Get animation settings
        frames, duration = get_animation_settings()
        print(f"\nSelected settings: {frames} frames with {duration}ms duration per frame")

        # Initialize session and model
        session_manager = SessionManager(args.output)
        print(f"\nStarted new session: {session_manager.session_id}")

        pipe = load_model()

        # Get region selection first
        region_type = get_user_choice(REGION_SETTINGS, "Select region to animate:")

        # Create regions config with the selected region type
        regions_config = {
            region_type: {
                'name': region_type,
                'intensity': REGION_SETTINGS[region_type]['default_intensity']
            }
        }

        process_image_with_regions(
            pipe,
            args.path,
            session_manager,
            regions_config,
            num_frames=frames,
            frame_duration=duration
        )

    finally:
        clear_memory()
        if 'pipe' in locals():
            del pipe
        clear_memory()


if __name__ == "__main__":
    main()