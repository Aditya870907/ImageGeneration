import gradio as gr
import requests
import base64
from PIL import Image
import io
import yaml
from pathlib import Path
import re
import time
from datetime import datetime
import tiktoken
from typing import Dict, List, Optional, Tuple, Any
from src.img2vid import (
    REGION_SETTINGS,
    # Add this if not already imported
    SessionManager,
    load_model,
    process_image_with_regions,
    clear_memory
)

class I2VTab:
    def __init__(self, generator):
        self.generator = generator
        self.animation_presets = {
            'subtle': {'frames': 20, 'duration': 100, 'description': 'Subtle movement (2 seconds)'},
            'normal': {'frames': 24, 'duration': 80, 'description': 'Normal movement (2 seconds)'},
            'slow': {'frames': 40, 'duration': 200, 'description': 'Slow motion (8 seconds)'},
            'ultra_slow': {'frames': 40, 'duration': 300, 'description': 'Ultra slow motion (12 seconds)'}
        }
        self.region_settings = REGION_SETTINGS  # Import from client_i2v

    def create_tab(self) -> Dict:
        with gr.Tab("🎥 Image to Video", id="i2v"):
            with gr.Group():
                gr.Markdown("### Image to Video Animation")

                with gr.Row():
                    input_image = gr.Image(
                        label="Input Image",
                        type="filepath",
                        scale=2
                    )
                    preview_area = gr.Image(
                        label="Preview",
                        interactive=False,
                        scale=2
                    )

                with gr.Row():
                    animation_preset = gr.Dropdown(
                        label="Animation Style",
                        choices=list(self.animation_presets.keys()),
                        value="normal",
                        scale=2
                    )
                    region_type = gr.Dropdown(
                        label="Region to Animate",
                        choices=list(self.region_settings.keys()),
                        value="face",
                        scale=2
                    )

                with gr.Row():
                    motion_type = gr.Dropdown(
                        label="Motion Type",
                        choices=[],  # Will be updated based on region selection
                        interactive=True,
                        scale=2
                    )

                with gr.Row():
                    output_dir = gr.Textbox(
                        label="Output Directory",
                        value="outputs",
                        scale=2
                    )

                with gr.Row():
                    generate_btn = gr.Button("🎬 Generate Animation", variant="primary", scale=2)
                    cancel_btn = gr.Button("❌ Cancel", variant="stop", scale=1)

                generation_status = gr.HTML(
                    value='<div class="generating-status">Ready to generate animation</div>'
                )
                output_video = gr.Video(
                    label="Generated Animation",
                    format="mp4",
                    interactive=False
                )

        # Update motion types when region changes
        region_type.change(
            self._update_motion_types,
            inputs=[region_type],
            outputs=[motion_type]
        )

        # Generate animation when button is clicked
        generate_btn.click(
            self._generate_animation,
            inputs=[
                input_image,
                animation_preset,
                region_type,
                motion_type,
                output_dir
            ],
            outputs=[
                generation_status,
                output_video
            ]
        )

        # Cancel generation
        cancel_btn.click(
            self._cancel_generation,
            outputs=[
                generation_status,
                output_video
            ]
        )

        return {
            "input_image": input_image,
            "animation_preset": animation_preset,
            "region_type": region_type,
            "motion_type": motion_type,
            "output_dir": output_dir,
            "generate_btn": generate_btn,
            "cancel_btn": cancel_btn,
            "generation_status": generation_status,
            "output_video": output_video
        }

    def _update_motion_types(self, region: str) -> gr.update:
        """Update motion type choices based on selected region"""
        if region in self.region_settings:
            motion_types = list(self.region_settings[region]['motion_types'].keys())
            return gr.update(choices=motion_types, value=motion_types[0])
        return gr.update(choices=[], value=None)

    def _generate_animation(
            self,
            image_path: str,
            preset: str,
            region: str,
            motion: str,
            output_dir: str
    ) -> Tuple[str, str]:
        """Generate animation from input image"""
        try:
            if not image_path:
                return (
                    '<div class="generating-status error-status">Error: No input image provided</div>',
                    None
                )

            # Get preset settings
            preset_config = self.animation_presets[preset]
            frames = preset_config['frames']
            duration = preset_config['duration']

            # Initialize session manager
            session_manager = SessionManager(output_dir)

            # Configure region settings
            regions_config = {
                region: {
                    'name': region,
                    'intensity': self.region_settings[region]['default_intensity'],
                    'motion_type': motion
                }
            }

            # Load SVD model
            pipe = load_model()

            try:
                # Generate animation
                process_image_with_regions(
                    pipe,
                    image_path,
                    session_manager,
                    regions_config,
                    num_frames=frames,
                    frame_duration=duration
                )

                # Get output path
                paths = session_manager.get_paths_for_image(Path(image_path).name)
                gif_path = paths['gif']

                return (
                    '<div class="generating-status success-status">✅ Animation generated successfully</div>',
                    str(gif_path)
                )

            finally:
                # Clean up
                if 'pipe' in locals():
                    del pipe
                clear_memory()

        except Exception as e:
            return (
                f'<div class="generating-status error-status">❌ Error generating animation: {str(e)}</div>',
                None
            )

    def _cancel_generation(self) -> Tuple[str, None]:
        """Handle generation cancellation"""
        clear_memory()
        return (
            '<div class="generating-status">Generation cancelled</div>',
            None
        )

class ImageGenerator:
    def __init__(self):
        self.encoder = tiktoken.get_encoding("cl100k_base")

    def sanitize_filename(self, filename: str) -> str:
        valid_filename = re.sub(r'[<>:"/\\|?*]', '', filename)
        return valid_filename.replace(' ', '_')

    def check_server_health(self, server_url: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
        try:
            response = requests.get(f"{server_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                return health_data.get('available_models', []), None
            return None, f"Server returned status code {response.status_code}"
        except requests.exceptions.RequestException as e:
            return None, f"Server connection error: {str(e)}"

    def get_available_models(self, server_url: str) -> Tuple[Optional[List[Dict]], Optional[str]]:
        try:
            response = requests.get(f"{server_url}/models", timeout=10)
            response.raise_for_status()
            models = response.json()['models']
            return [m for m in models if m.get('status') != 'error'], None
        except Exception as e:
            return None, f"Error fetching models: {str(e)}"

    def load_prompts(self, file) -> Tuple[Optional[Dict], str]:
        """Load prompts from YAML file"""
        try:
            if file is None:
                return None, "No file uploaded"

            # Read the file content from the file path
            with open(file.name, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)

            if not isinstance(data, dict) or 'prompt_sets' not in data:
                return None, "Invalid YAML format. Must contain 'prompt_sets' key."

            prompts = {}
            for prompt_set in data['prompt_sets']:
                if 'name' in prompt_set and 'prompt' in prompt_set:
                    prompts[prompt_set['name']] = {
                        'prompt': prompt_set['prompt'],
                        'params': prompt_set.get('params', {}),
                        'negative_prompt': prompt_set.get('negative_prompt', '')
                    }

            if not prompts:
                return None, "No valid prompt sets found in YAML"

            return prompts, f"Successfully loaded {len(prompts)} prompt sets"
        except Exception as e:
            return None, f"Error loading YAML: {str(e)}"

    def save_image(self, image_data: bytes, path: Path):
        image = Image.open(io.BytesIO(image_data))
        image.save(path)
        return image

    def generate_images(self,
                        server_url: str,
                        selected_models: List[Dict],
                        compare_mode: bool,
                        params: Dict,
                        output_dir: str,
                        name_prefix: str,
                        batch_size: int = 1) -> Tuple[List[Dict], str]:
        """
        Modified to return both images and their metadata for proper labeling
        Returns Tuple[List[Dict[str, Any]], str] where dict contains 'image' and 'model_name'
        """
        output_path = Path(output_dir)
        images_dir = output_path / "images"
        metadata_dir = output_path / "metadata"
        images_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        generated_results = []  # List of dicts containing image and model info
        status_messages = []

        try:
            if compare_mode:
                # Override image dimensions for comparison mode
                params['width'] = min(params['width'], 512)  # Limit size for comparison
                params['height'] = min(params['height'], 512)

                for model in selected_models:
                    try:
                        params['model_id'] = model['id']
                        response = requests.post(f"{server_url}/generate", json=params, timeout=60)
                        response.raise_for_status()
                        result = response.json()

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_path = images_dir / f"{self.sanitize_filename(name_prefix)}_{model['name']}_{timestamp}.png"
                        metadata_path = metadata_dir / f"{self.sanitize_filename(name_prefix)}_{model['name']}_{timestamp}.yaml"

                        image = self.save_image(base64.b64decode(result['image_base64']), image_path)

                        # Store both image and model info
                        generated_results.append({
                            'image': image,
                            'model_name': model['name'],
                            'base_model': model.get('base_model', 'Unknown')
                        })

                        with open(metadata_path, 'w') as f:
                            yaml.safe_dump(result['generation_settings'], f)

                        status_messages.append(f"✅ Generated image with {model['name']}")
                    except Exception as e:
                        status_messages.append(f"❌ Failed with {model['name']}: {str(e)}")
            else:
                model = selected_models[0]
                params['model_id'] = model['id']

                for i in range(batch_size):
                    try:
                        response = requests.post(f"{server_url}/generate", json=params, timeout=60)
                        response.raise_for_status()
                        result = response.json()

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        image_path = images_dir / f"{self.sanitize_filename(name_prefix)}_{model['name']}_{timestamp}_{i}.png"
                        metadata_path = metadata_dir / f"{self.sanitize_filename(name_prefix)}_{model['name']}_{timestamp}_{i}.yaml"

                        image = self.save_image(base64.b64decode(result['image_base64']), image_path)
                        generated_results.append({
                            'image': image,
                            'model_name': model['name'],
                            'base_model': model.get('base_model', 'Unknown')
                        })

                        with open(metadata_path, 'w') as f:
                            yaml.safe_dump(result['generation_settings'], f)

                        status_messages.append(f"✅ Generated image {i + 1}/{batch_size}")
                        time.sleep(0.1)
                    except Exception as e:
                        status_messages.append(f"❌ Failed to generate image {i + 1}: {str(e)}")

            return generated_results, "\n".join(status_messages)
        except Exception as e:
            return [], f"❌ Critical error: {str(e)}"

class GradioInterface:
    def __init__(self):
        self.generator = ImageGenerator()
        self.default_css = """
                    #gallery {
                        min-height: 400px;
                        margin-top: 16px;
                        margin-bottom: 16px;
                    }
                    .generating {
                        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
                    }
                    @keyframes pulse {
                        0%, 100% { opacity: 1; }
                        50% { opacity: .5; }
                    }
                    .image-label {
                        text-align: center;
                        font-weight: bold;
                        margin-top: 8px;
                        margin-bottom: 16px;
                    }
                    .gallery-container {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                        gap: 20px;
                        padding: 20px;
                    }
                    .image-container {
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                        background: #f8f9fa;
                        padding: 15px;
                        border-radius: 12px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        transition: transform 0.2s;
                    }
                    .image-container:hover {
                        transform: translateY(-5px);
                        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                    }
                    .image-container img {
                        width: 100%;
                        height: auto;
                        border-radius: 8px;
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                    }
                    .generating-status {
                        text-align: center;
                        margin: 10px 0;
                        padding: 10px;
                        background: #e8f0fe;
                        border-radius: 8px;
                        font-weight: 500;
                    }
                    .error-status {
                        background: #fee8e8;
                        color: #d32f2f;
                    }
                    .success-status {
                        background: #e8fee8;
                        color: #2f7d32;
                    }
                """
        self.blocks = self.create_interface()

    def _create_header(self):
        """Create the header section of the interface"""
        gr.Markdown("""
        # Enhanced Stable Diffusion Image Generator
        Generate high-quality images with multiple models and advanced settings.
        """)

    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(title="Enhanced Stable Diffusion Image Generator", css=self.default_css) as app:
            self._create_header()

            # State management
            prompts_data = gr.State(None)
            models_state = gr.State([])
            is_generating = gr.State(False)

            with gr.Tabs() as tabs:
                # Combined Connection & Settings tab
                configuration_tab = self._create_configuration_tab()
                
                # Combined Project & Prompt tab with generation controls
                generation_tab = self._create_generation_tab()
                
                # Output tab (kept separate)
                output_tab = self._create_output_tab()

                # Image to Video tab (unchanged)
                i2v_tab = I2VTab(self.generator).create_tab()

            self._setup_event_handlers(
                configuration_tab,
                generation_tab,
                output_tab,
                prompts_data,
                models_state,
                is_generating
            )

        return app

    def _create_configuration_tab(self) -> Dict:
        with gr.Tab("⚙️ Configuration", id="configuration"):
            with gr.Group():
                # Server Configuration Section
                gr.Markdown("### Server Configuration")
                with gr.Row():
                    server_url = gr.Textbox(
                        label="Server URL",
                        placeholder="http://localhost:8001",
                        value="http://localhost:8001",
                        scale=4
                    )
                    refresh_btn = gr.Button("🔄 Refresh Models", scale=1)
                server_status = gr.Markdown("Server status: Not connected")

                # Model Settings Section
                gr.Markdown("### Model Settings")
                with gr.Row():
                    model_selection = gr.Dropdown(
                        label="Select Model(s)",
                        choices=[],
                        multiselect=True,
                        interactive=True,
                        scale=3
                    )
                    compare_mode = gr.Checkbox(
                        label="Compare Models",
                        value=False,
                        scale=1
                    )

                # Generation Settings Section
                gr.Markdown("### Generation Settings")
                with gr.Row():
                    with gr.Column(scale=1):
                        steps = gr.Slider(
                            label="Steps",
                            minimum=1,
                            maximum=100,
                            value=30,
                            step=1
                        )
                        guidance_scale = gr.Slider(
                            label="Guidance Scale",
                            minimum=1,
                            maximum=20,
                            value=7.5,
                            step=0.1
                        )
                    with gr.Column(scale=1):
                        width = gr.Slider(
                            label="Width",
                            minimum=256,
                            maximum=1024,
                            value=512,
                            step=64
                        )
                        height = gr.Slider(
                            label="Height",
                            minimum=256,
                            maximum=1024,
                            value=512,
                            step=64
                        )

                with gr.Row():
                    scheduler_type = gr.Dropdown(
                        label="Scheduler",
                        choices=["dpmsolver++", "euler_a", "euler", "ddim"],
                        value="dpmsolver++",
                        scale=2
                    )
                    batch_size = gr.Slider(
                        label="Batch Size (ignored in compare mode)",
                        minimum=1,
                        maximum=100,
                        value=1,
                        step=1,
                        scale=2
                    )

            return {
                "server_url": server_url,
                "refresh_btn": refresh_btn,
                "server_status": server_status,
                "model_selection": model_selection,
                "compare_mode": compare_mode,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "width": width,
                "height": height,
                "scheduler_type": scheduler_type,
                "batch_size": batch_size
            }

    def _create_generation_tab(self) -> Dict:
        with gr.Tab("🎨 Generation", id="generation"):
            with gr.Group():
                # Project Settings Section
                gr.Markdown("### Project Settings")
                with gr.Row():
                    prompts_file = gr.File(
                        label="Upload prompts.yaml",
                        file_types=[".yaml", ".yml"],
                        scale=4
                    )
                prompts_status = gr.Markdown("No prompts loaded")
                prompt_set = gr.Dropdown(
                    label="Select Prompt Set",
                    choices=[],
                    interactive=True
                )

                # Prompt Configuration Section
                gr.Markdown("### Prompt Configuration")
                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="Enter your image generation prompt",
                    lines=3
                )
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    placeholder="Enter negative prompt (optional)",
                    lines=3
                )

                # Output Configuration
                gr.Markdown("### Output Configuration")
                with gr.Row():
                    name_prefix = gr.Textbox(
                        label="Output Name Prefix",
                        value="generated",
                        scale=2
                    )
                    output_dir = gr.Textbox(
                        label="Output Directory",
                        value="outputs",
                        scale=2
                    )

                # Generation Controls
                with gr.Row():
                    generate_btn = gr.Button(
                        "🚀 Generate Images",
                        variant="primary",
                        scale=2
                    )
                    cancel_btn = gr.Button(
                        "❌ Cancel",
                        variant="stop",
                        scale=1
                    )

            return {
                "prompts_file": prompts_file,
                "prompts_status": prompts_status,
                "prompt_set": prompt_set,
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "name_prefix": name_prefix,
                "output_dir": output_dir,
                "generate_btn": generate_btn,
                "cancel_btn": cancel_btn
            }

    def _create_output_tab(self) -> Dict:
        with gr.Tab("🖼️ Output", id="output"):
            with gr.Group():
                gr.Markdown("### Generation Output")
                generation_status = gr.HTML(
                    value='<div class="generating-status">Ready to generate</div>'
                )
                gallery_output = gr.HTML(
                    label="Generated Images",
                    elem_id="gallery"
                )
                status_output = gr.Markdown("Status: Ready")

            return {
                "generation_status": generation_status,
                "gallery_output": gallery_output,
                "status_output": status_output
            }

    def _setup_event_handlers(
            self,
            configuration_tab: Dict,
            generation_tab: Dict,
            output_tab: Dict,
            prompts_data: gr.State,
            models_state: gr.State,
            is_generating: gr.State
    ):
        # Refresh models event
        configuration_tab["refresh_btn"].click(
            self._refresh_models,
            inputs=[configuration_tab["server_url"]],
            outputs=[
                configuration_tab["server_status"],
                models_state,
                configuration_tab["model_selection"],
                models_state
            ]
        )

        # Load prompts event
        generation_tab["prompts_file"].upload(
            self._load_prompts,
            inputs=[generation_tab["prompts_file"]],
            outputs=[
                prompts_data,
                generation_tab["prompt_set"],
                generation_tab["prompts_status"]
            ]
        )

        # Update from prompt set event
        generation_tab["prompt_set"].change(
            self._update_from_prompt_set,
            inputs=[generation_tab["prompt_set"], prompts_data],
            outputs=[
                generation_tab["prompt"],
                generation_tab["negative_prompt"],
                configuration_tab["steps"],
                configuration_tab["guidance_scale"],
                configuration_tab["width"],
                configuration_tab["height"],
                configuration_tab["scheduler_type"]
            ]
        )

        # Compare mode update event
        configuration_tab["compare_mode"].change(
            self._update_compare_mode,
            inputs=[configuration_tab["compare_mode"]],
            outputs=[
                configuration_tab["model_selection"],
                configuration_tab["batch_size"],
                output_tab["generation_status"]
            ]
        )

        # Generate images event
        gen_event = generation_tab["generate_btn"].click(
            self._generate,
            inputs=[
                configuration_tab["server_url"],
                configuration_tab["model_selection"],
                configuration_tab["compare_mode"],
                generation_tab["prompt"],
                generation_tab["negative_prompt"],
                configuration_tab["steps"],
                configuration_tab["guidance_scale"],
                configuration_tab["width"],
                configuration_tab["height"],
                configuration_tab["scheduler_type"],
                configuration_tab["batch_size"],
                generation_tab["output_dir"],
                generation_tab["name_prefix"],
                models_state
            ],
            outputs=[
                output_tab["generation_status"],
                output_tab["gallery_output"],
                output_tab["status_output"]
            ]
        )

        # Cancel generation event
        generation_tab["cancel_btn"].click(
            self._cancel_generation,
            outputs=[
                output_tab["generation_status"],
                output_tab["gallery_output"],
                output_tab["status_output"]
            ],
            cancels=[gen_event]
        )

        # Generation status updates
        generation_tab["generate_btn"].click(
            lambda: True,
            outputs=is_generating,
            queue=False
        )

    def _refresh_models(self, url: str) -> Tuple[str, List, gr.update, List]:
        """Refresh available models from the server"""
        models, error = self.generator.get_available_models(url)
        if error:
            return (
                f'<div class="generating-status error-status">❌ Server Error: {error}</div>',
                [],
                gr.update(choices=[], value=None),
                []
            )

        model_names = [f"{m['name']} ({m['id']})" for m in models]
        return (
            f'<div class="generating-status success-status">✅ Connected: {len(models)} models available</div>',
            models,
            gr.update(choices=model_names, value=None),
            models
        )

    def _load_prompts(self, file) -> Tuple[Optional[Dict], gr.update, str]:
        """Load prompts from YAML file"""
        prompts, message = self.generator.load_prompts(file)
        if prompts:
            choices = list(prompts.keys())
            return (
                prompts,
                gr.update(choices=choices, value=choices[0] if choices else None),
                f'<div class="generating-status success-status">✅ {message}</div>'
            )
        return (
            None,
            gr.update(choices=[], value=None),
            f'<div class="generating-status error-status">❌ {message}</div>'
        )

    def _update_from_prompt_set(self, prompt_set_name: str, prompts: Dict) -> List[Any]:
        """Update interface with selected prompt set values"""
        if not prompts or prompt_set_name not in prompts:
            return [gr.update()] * 7

        prompt_data = prompts[prompt_set_name]
        params = prompt_data.get('params', {})

        return [
            prompt_data['prompt'],
            prompt_data.get('negative_prompt', ''),
            params.get('num_steps', 30),
            params.get('guidance_scale', 7.5),
            params.get('width', 512),
            params.get('height', 512),
            params.get('scheduler_type', 'dpmsolver++')
        ]

    def _update_compare_mode(self, compare_enabled: bool) -> Tuple[gr.update, gr.update, str]:
        """Update interface based on compare mode selection"""
        status_message = (
            'Compare mode: Using all models with 512x512 resolution'
            if compare_enabled
            else 'Single model mode: Using selected model and settings'
        )
        return [
            gr.update(interactive=not compare_enabled, value=[] if compare_enabled else None),
            gr.update(value=1 if compare_enabled else None, interactive=not compare_enabled),
            f'<div class="generating-status">{status_message}</div>'
        ]

    def _cancel_generation(self) -> Tuple[str, str, str]:
        """Handle generation cancellation"""
        return (
            '<div class="generating-status">Generation cancelled</div>',
            "",
            "Generation cancelled by user"
        )

    def _create_gallery_html(self, results: List[Dict]) -> str:
        """Create HTML for the image gallery"""
        gallery_html = '<div class="gallery-container">'

        for result in results:
            try:
                buffered = io.BytesIO()
                result['image'].save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                model_name = result.get('model_name', 'Unknown Model')
                base_model = result.get('base_model', 'Unknown Base')

                gallery_html += f'''
                <div class="image-container">
                    <img src="data:image/png;base64,{img_str}" 
                         title="{model_name} ({base_model})"
                         alt="{model_name}"
                    />
                    <div class="image-label">
                        <div style="font-weight:bold;color:#1a73e8;">{model_name}</div>
                        <div style="font-size:0.9em;color:#666;">({base_model})</div>
                    </div>
                </div>
                '''
            except Exception as e:
                print(f"Error processing gallery image: {str(e)}")
                continue

        gallery_html += '</div>'
        return gallery_html

    def _generate(
            self,
            server_url: str,
            model_selection: List[str],
            compare_mode: bool,
            prompt: str,
            negative_prompt: str,
            steps: int,
            guidance_scale: float,
            width: int,
            height: int,
            scheduler_type: str,
            batch_size: int,
            output_dir: str,
            name_prefix: str,
            models_list: List[Dict]
    ) -> Tuple[str, str, str]:
        """Handle image generation process"""
        try:
            # Input validation
            if not prompt:
                return (
                    '<div class="generating-status error-status">Error: No prompt provided</div>',
                    "",
                    "❌ Please enter a prompt"
                )

            if not server_url:
                return (
                    '<div class="generating-status error-status">Error: No server URL</div>',
                    "",
                    "❌ Please enter server URL"
                )

            # Model selection logic
            if compare_mode:
                if not models_list:
                    return (
                        '<div class="generating-status error-status">Error: No models available</div>',
                        "",
                        "❌ No models available. Please check server connection."
                    )
                selected_models = models_list
                width = min(width, 512)
                height = min(height, 512)
                batch_size = 1
            else:
                if not model_selection:
                    return (
                        '<div class="generating-status error-status">Error: No model selected</div>',
                        "",
                        "❌ Please select at least one model"
                    )
                selected_models = [
                    m for m in models_list
                    if f"{m['name']} ({m['id']})" in model_selection
                ]
                if not selected_models:
                    return (
                        '<div class="generating-status error-status">Error: Invalid model selection</div>',
                        "",
                        "❌ Selected models not found in available models list"
                    )

            # Prepare parameters
            params = {
                "prompt": prompt,
                "negative_prompt": negative_prompt or "",
                "num_steps": int(steps),
                "guidance_scale": float(guidance_scale),
                "width": int(width),
                "height": int(height),
                "scheduler_type": scheduler_type,
                "karras_sigmas": True,
                "scheduler_scale": 0.7,
                "enable_attention_slicing": True,
                "enable_vae_slicing": True,
                "enable_vae_tiling": True,
                "enable_model_cpu_offload": True,
                "enable_sequential_cpu_offload": False,
                "num_images": 1,
                "clip_skip": 2
            }

            # Generate images
            results, status = self.generator.generate_images(
                server_url=server_url,
                selected_models=selected_models,
                compare_mode=compare_mode,
                params=params,
                output_dir=output_dir,
                name_prefix=name_prefix,
                batch_size=batch_size
            )

            if not results:
                return (
                    '<div class="generating-status error-status">Generation failed</div>',
                    "",
                    f"❌ No images were generated\n\nDetails:\n{status}"
                )

            # Create gallery HTML
            gallery_html = self._create_gallery_html(results)

            # Generate status message
            if compare_mode:
                status_msg = f"✅ Generated comparison images with {len(results)} models"
            else:
                status_msg = f"✅ Generated {len(results)} images with selected model"

            if status:
                status_msg += f"\n\nDetails:\n{status}"

            return (
                '<div class="generating-status success-status">✅ Generation complete</div>',
                gallery_html,
                status_msg
            )

        except Exception as e:
            error_msg = f"❌ Unexpected error: {str(e)}"
            print(f"Generate function error: {str(e)}")
            return (
                '<div class="generating-status error-status">❌ Generation failed</div>',
                "",
                error_msg
            )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Launch Stable Diffusion Client Interface')
    parser.add_argument('--port', type=int, default=7860, help='Port to run the interface on')
    parser.add_argument('--share', action='store_true', help='Create a public URL')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    # Create the interface and launch directly from the blocks object
    interface = GradioInterface()
    interface.blocks.launch(
        server_port=args.port,
        share=args.share,
        debug=args.debug
    )