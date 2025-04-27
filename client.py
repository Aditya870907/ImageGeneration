import gradio as gr
import requests
import os
import base64
import io
from PIL import Image
import time
import json
from pathlib import Path
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default server URL
DEFAULT_SERVER_URL = "http://localhost:8001"

# Default output directory
DEFAULT_OUTPUT_DIR = "output"

def create_interface():
    # Global variables
    server_url = DEFAULT_SERVER_URL
    connected = False
    models = []
    selected_model = None
    
    # Create output directory
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    
    # Helper functions
    def connect_to_server(url):
        nonlocal server_url, connected, models, selected_model
        server_url = url
        
        try:
            # Check server health
            response = requests.get(f"{server_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"Server connected: {health_data}")
                connected = True
                
                # Fetch models
                response = requests.get(f"{server_url}/models", timeout=10)
                if response.status_code == 200:
                    models_data = response.json()
                    models = models_data.get("models", [])
                    
                    # Set selected model
                    if health_data.get("current_model"):
                        selected_model = health_data["current_model"]["name"]
                    elif models:
                        selected_model = models[0]["name"]
                    
                    model_names = [model["name"] for model in models]
                    return "Connected", f"Connected to server. Found {len(models)} models.", gr.update(choices=model_names, value=selected_model)
                else:
                    return "Disconnected", f"Failed to fetch models: {response.status_code}", gr.update(choices=[])
            else:
                connected = False
                return "Disconnected", f"Failed to connect to server: {response.status_code}", gr.update(choices=[])
        except Exception as e:
            connected = False
            logger.error(f"Connection error: {e}")
            return "Disconnected", f"Connection error: {str(e)}", gr.update(choices=[])
    
    def generate_image(prompt, negative_prompt, width, height, steps, guidance, model_name, scheduler):
        nonlocal connected, models, server_url
        
        if not connected:
            return None, "Not connected to server. Please connect first."
        
        try:
            # Find model_id from name
            model_id = None
            for m in models:
                if m["name"] == model_name:
                    model_id = m["id"]
                    break
            
            # Prepare the request
            request_data = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": width,
                "height": height,
                "num_steps": steps,
                "guidance_scale": guidance,
                "scheduler_type": scheduler,
                "karras_sigmas": True
            }
            
            if model_id:
                request_data["model_id"] = model_id
            
            # Send the request
            start_time = time.time()
            response = requests.post(f"{server_url}/generate", 
                                    json=request_data, 
                                    timeout=180)
            
            if response.status_code == 200:
                result = response.json()
                image_data = base64.b64decode(result["image_base64"])
                image = Image.open(io.BytesIO(image_data))
                
                # Save the image
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                filename = f"{DEFAULT_OUTPUT_DIR}/generation_{timestamp}.png"
                image.save(filename)
                
                generation_time = time.time() - start_time
                return image, f"Generated in {generation_time:.2f}s - Seed: {result['seed']}"
            else:
                return None, f"Generation error: {response.status_code} {response.text}"
        except Exception as e:
            return None, f"Generation error: {str(e)}"
    
    # Create the Gradio interface
    with gr.Blocks(title="Stable Diffusion Client") as interface:
        gr.Markdown("# 🖼️ Stable Diffusion Client")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## 🔌 Connection")
                server_url_input = gr.Textbox(value=DEFAULT_SERVER_URL, label="Server URL")
                with gr.Row():
                    connect_btn = gr.Button("Connect")
                    status = gr.Textbox(value="Disconnected", label="Status")
                
                gr.Markdown("## ⚙️ Settings")
                model_dropdown = gr.Dropdown(choices=[], label="Model")
                
                with gr.Row():
                    width = gr.Slider(minimum=384, maximum=1024, step=64, value=512, label="Width")
                    height = gr.Slider(minimum=384, maximum=1024, step=64, value=512, label="Height")
                
                with gr.Row():
                    steps = gr.Slider(minimum=1, maximum=100, step=1, value=30, label="Steps")
                    guidance = gr.Slider(minimum=1, maximum=20, step=0.1, value=7.5, label="Guidance Scale")
                
                scheduler = gr.Dropdown(
                    choices=["dpmsolver++", "euler_a", "euler", "ddim"], 
                    value="dpmsolver++",
                    label="Scheduler"
                )
            
            with gr.Column(scale=2):
                gr.Markdown("## ✏️ Prompt")
                prompt = gr.Textbox(
                    lines=5, 
                    placeholder="Describe what you want to see...",
                    label="Prompt"
                )
                negative_prompt = gr.Textbox(
                    lines=3, 
                    placeholder="Describe what you want to avoid...",
                    label="Negative Prompt"
                )
                
                gr.Markdown("## 🖼️ Output")
                generate_btn = gr.Button("Generate Image", variant="primary")
                image_output = gr.Image(type="pil", label="Generated Image")
                output_message = gr.Textbox(label="Status")
        
        # Connect event handlers
        connect_btn.click(
            connect_to_server,
            inputs=[server_url_input],
            outputs=[status, output_message, model_dropdown]
        )
        
        generate_btn.click(
            generate_image,
            inputs=[prompt, negative_prompt, width, height, steps, guidance, model_dropdown, scheduler],
            outputs=[image_output, output_message]
        )
    
    return interface

# Launch the interface
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)