import requests
import base64
from PIL import Image
import io
import argparse
from datetime import datetime
import os
import yaml
from pathlib import Path
import re
import time
from tqdm import tqdm
import tiktoken


def count_tokens(text):
    """Count tokens in text using tiktoken"""
    encoder = tiktoken.get_encoding("cl100k_base")
    return len(encoder.encode(text))


def safe_generate_prompt_with_ollama(ollama_config, attempt=1, max_attempts=3):
    """Safely generate prompt with Ollama with retries"""
    if 'server_url' not in ollama_config:
        raise ValueError("Ollama configuration missing required 'server_url' field")

    server_url = ollama_config['server_url']

    for i in range(max_attempts):
        try:
            model = ollama_config.get('model', 'llama2')

            try:
                requests.get(f"{server_url}/api/version", timeout=2)
            except requests.RequestException as e:
                raise RuntimeError(f"Cannot connect to Ollama server at {server_url}: {str(e)}")

            system_message = ollama_config.get('system_message', 'Generate an image generation prompt.')
            base_prompt = ollama_config.get('prompt', 'Generate a creative image.')

            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": base_prompt}
                ],
                "stream": False
            }

            print(f"\nAttempting to generate prompt (attempt {i + 1}/{max_attempts})...")

            response = requests.post(f"{server_url}/api/chat", json=payload, timeout=10)
            response.raise_for_status()

            generated_prompt = response.json()['message']['content'].strip()

            try:
                token_count = count_tokens(generated_prompt)
            except Exception as e:
                print(f"Token counting failed: {e}")
                token_count = 0

            start_time = time.time()
            while token_count != 77 and (time.time() - start_time) < 5:
                if token_count > 77:
                    words = generated_prompt.split()
                    generated_prompt = " ".join(words[:-1])
                else:
                    quality_terms = ["detailed", "high quality", "masterpiece", "8k", "beautiful"]
                    for term in quality_terms:
                        if count_tokens(generated_prompt + f", {term}") <= 77:
                            generated_prompt += f", {term}"
                        if count_tokens(generated_prompt) == 77:
                            break
                token_count = count_tokens(generated_prompt)

            print(f"Generated prompt ({token_count} tokens):")
            print(generated_prompt)
            return generated_prompt

        except requests.Timeout:
            print(f"Ollama request timed out on attempt {i + 1}")
            if i == max_attempts - 1:
                print("All attempts failed, using fallback prompt")
                return ollama_config.get('fallback_prompt', 'A basic image.')
            time.sleep(1)

        except Exception as e:
            print(f"Error on attempt {i + 1}: {e}")
            if i == max_attempts - 1:
                print("All attempts failed, using fallback prompt")
                return ollama_config.get('fallback_prompt', 'A basic image.')
            time.sleep(1)

    return ollama_config.get('fallback_prompt', 'A basic image.')


def check_server_health(server_url, timeout=5):
    """Check server health with timeout"""
    try:
        response = requests.get(f"{server_url}/health", timeout=timeout)
        if response.status_code == 200:
            health_data = response.json()
            return health_data.get('available_models', []), True
        return None, False
    except requests.Timeout:
        print(f"Health check timeout after {timeout} seconds")
        return None, False
    except Exception as e:
        print(f"Health check error: {e}")
        return None, False


def sanitize_filename(filename):
    """Convert a string into a valid filename by removing invalid characters"""
    valid_filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    valid_filename = valid_filename.replace(' ', '_')
    return valid_filename


def load_project_config(input_folder, project_name):
    """Load configuration files from the project folder"""
    project_path = Path(input_folder) / project_name

    if not project_path.exists():
        raise ValueError(f"Project folder {project_path} does not exist")

    config_path = project_path / "config.yaml"
    prompts_path = project_path / "prompts.yaml"

    if not config_path.exists():
        raise ValueError(f"Config file not found in {project_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if prompts_path.exists():
        with open(prompts_path) as f:
            prompts = yaml.safe_load(f)
    else:
        prompts = {"prompt_sets": []}

    return config, prompts


def get_prompt_config(prompts, prompt_name):
    """Get specific prompt configuration by name"""
    for prompt_set in prompts.get('prompt_sets', []):
        if prompt_set['name'] == prompt_name:
            return prompt_set
    raise ValueError(f"Prompt set '{prompt_name}' not found")


def create_output_structure(base_dir, project_name, name_prefix, model_name=None, is_comparison=False):
    """Create and return paths for organized output structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    project_dir = Path(base_dir) / sanitize_filename(project_name)
    project_dir.mkdir(parents=True, exist_ok=True)

    if is_comparison:
        batch_name = f"comparison_{timestamp}"
    else:
        batch_name = f"{timestamp}"
        if model_name:
            batch_name = f"{timestamp}_{sanitize_filename(model_name)}"

    prompt_dir = project_dir / sanitize_filename(name_prefix) / batch_name
    prompt_dir.mkdir(parents=True, exist_ok=True)

    images_dir = prompt_dir / "images"
    metadata_dir = prompt_dir / "metadata"
    images_dir.mkdir(exist_ok=True)
    metadata_dir.mkdir(exist_ok=True)

    return images_dir, metadata_dir


def save_generation_result(result, images_dir, metadata_dir, name_prefix):
    """Save a single generation result (image and metadata)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = sanitize_filename(result['generation_settings']['model'])

    # Save image
    image_filename = f"{sanitize_filename(name_prefix)}_{model_name}_{timestamp}.png"
    image_path = images_dir / image_filename

    image_data = base64.b64decode(result['image_base64'])
    image = Image.open(io.BytesIO(image_data))
    image.save(image_path)

    # Save metadata
    metadata_filename = f"{sanitize_filename(name_prefix)}_{model_name}_{timestamp}.yaml"
    metadata_path = metadata_dir / metadata_filename

    with open(metadata_path, 'w') as f:
        yaml.safe_dump(result['generation_settings'], f, default_flow_style=False)

    return image_path, metadata_path


def safe_generate_single_image(server_url, final_prompt, negative_prompt, params, model_id=None, timeout=30):
    """Generate a single image with timeout and error handling"""
    try:
        payload = {
            "prompt": final_prompt,
            "negative_prompt": negative_prompt,
            "num_steps": params.get('num_steps', 30),
            "guidance_scale": params.get('guidance_scale', 7.5),
            "width": params.get('width', 512),
            "height": params.get('height', 512),
            "scheduler_type": params.get('scheduler_type', 'dpmsolver++'),
            "karras_sigmas": params.get('karras_sigmas', True),
            "scheduler_scale": params.get('scheduler_scale', 0.7),
            "enable_attention_slicing": params.get('enable_attention_slicing', True),
            "enable_vae_slicing": params.get('enable_vae_slicing', True),
            "enable_vae_tiling": params.get('enable_vae_tiling', False),
            "enable_model_cpu_offload": params.get('enable_model_cpu_offload', True),
            "enable_sequential_cpu_offload": params.get('enable_sequential_cpu_offload', False),
            "num_images": params.get('num_images', 1),
            "clip_skip": params.get('clip_skip', 2)
        }

        if model_id:
            payload["model_id"] = model_id

        if 'seed' in params:
            payload['seed'] = params['seed']

        response = requests.post(f"{server_url}/generate", json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json()

    except requests.Timeout:
        print(f"Server request timed out after {timeout} seconds")
        return None
    except Exception as e:
        print(f"Error in image generation: {e}")
        return None


def compare_models(server_url, final_prompt, negative_prompt, params, output_dirs=None, name_prefix=None,
                   show_image=False, timeout=120):
    """Generate images using all available models for comparison with immediate saving"""
    try:
        payload = {
            "prompt": final_prompt,
            "negative_prompt": negative_prompt,
            "num_steps": params.get('num_steps', 30),
            "guidance_scale": params.get('guidance_scale', 7.5),
            "width": params.get('width', 512),
            "height": params.get('height', 512),
            "scheduler_type": params.get('scheduler_type', 'dpmsolver++'),
            "karras_sigmas": params.get('karras_sigmas', True),
            "scheduler_scale": params.get('scheduler_scale', 0.7),
            "enable_attention_slicing": params.get('enable_attention_slicing', True),
            "enable_vae_slicing": params.get('enable_vae_slicing', True),
            "enable_vae_tiling": params.get('enable_vae_tiling', False),
            "enable_model_cpu_offload": params.get('enable_model_cpu_offload', True),
            "enable_sequential_cpu_offload": params.get('enable_sequential_cpu_offload', False),
            "num_images": params.get('num_images', 1),
            "clip_skip": params.get('clip_skip', 2)
        }

        if 'seed' in params:
            payload['seed'] = params['seed']

        # Get list of available models first
        models_response = requests.get(f"{server_url}/models", timeout=10)
        models_response.raise_for_status()
        available_models = models_response.json()['models']

        print(f"\nFound {len(available_models)} available models")
        valid_models = [m for m in available_models if m.get('status') != 'error']
        print(f"Using {len(valid_models)} valid models for comparison")

        results = []
        with tqdm(valid_models, desc="Generating comparisons") as pbar:
            for model in pbar:
                try:
                    model_payload = payload.copy()
                    model_payload["model_id"] = model['id']

                    pbar.set_description(f"Generating with {model['name']}")
                    response = requests.post(f"{server_url}/generate", json=model_payload, timeout=timeout)
                    response.raise_for_status()
                    result = response.json()

                    # Save result immediately if output directories are provided
                    if output_dirs and name_prefix:
                        image_path, metadata_path = save_generation_result(
                            result,
                            output_dirs[0],  # images_dir
                            output_dirs[1],  # metadata_dir
                            name_prefix
                        )
                        print(f"\nSaved result for {model['name']}:")
                        print(f"Image: {image_path}")
                        print(f"Metadata: {metadata_path}")
                        if show_image:
                            Image.open(image_path).show()

                    results.append(result)
                    pbar.set_description(f"Completed {model['name']}")
                except Exception as e:
                    print(f"\nFailed to generate with {model['name']}: {e}")
                    continue

        if not results:
            raise RuntimeError("No successful generations")

        return results

    except requests.Timeout:
        print(f"Server request timed out after {timeout} seconds")
        return None
    except Exception as e:
        print(f"Error in comparison generation: {e}")
        return None


def generate_image(project_config, output_dir, use_ollama=False, prompt_config=None, batch_size=None, show_image=True,
                   compare=False):
    """Generate image(s) using project configurations"""
    server_url = project_config.get('server_url')
    if not server_url:
        raise ValueError("Server URL not found in config.yaml")

    # Check server health and get available models
    available_models, is_healthy = check_server_health(server_url)
    if not is_healthy:
        raise RuntimeError(f"Server is not healthy or unreachable at {server_url}")

    current_model = available_models[0] if available_models else None
    name_prefix = 'ollama_generated' if use_ollama else prompt_config['name']

    # Create output directories
    images_dir, metadata_dir = create_output_structure(
        output_dir,
        project_config.get('name', 'default_project'),
        name_prefix,
        current_model.get('name') if current_model else None,
        is_comparison=compare
    )

    # Get parameters
    if use_ollama:
        params = project_config.get('default_params', {})
        negative_prompt = project_config['ollama'].get('negative_prompt', '')
    else:
        params = {**project_config.get('default_params', {}), **prompt_config.get('params', {})}
        negative_prompt = prompt_config.get('negative_prompt', '')

    print("\nGeneration Details:")
    print(f"Server URL: {server_url}")
    print(f"Parameters: steps={params.get('num_steps', 30)}, "
          f"guidance={params.get('guidance_scale', 7.5)}, "
          f"size={params.get('width', 512)}x{params.get('height', 512)}")

    try:
        if compare:
            total_iterations = batch_size if batch_size else 1
            print(f"\nGenerating comparison across {len(available_models)} models, "
                  f"{total_iterations} iteration(s) each...")

            for iteration in range(total_iterations):
                if use_ollama:
                    final_prompt = safe_generate_prompt_with_ollama(project_config['ollama'])
                else:
                    final_prompt = prompt_config['prompt']
                    if not any(x in final_prompt.lower() for x in ['masterpiece:', 'best quality', 'photorealistic:']):
                        final_prompt = "(masterpiece:1.2), (photorealistic:1.4), (best quality), " + final_prompt

                print(f"\nIteration {iteration + 1}/{total_iterations}")

                # Pass output directories to compare_models
                results = compare_models(
                    server_url,
                    final_prompt,
                    negative_prompt,
                    params,
                    output_dirs=(images_dir, metadata_dir),
                    name_prefix=f"{name_prefix}_iter{iteration + 1}",
                    show_image=show_image
                )

                # Add delay between iterations if needed
                if results and iteration < total_iterations - 1:
                    time.sleep(1)

        elif batch_size:
            print(f"\nGenerating {batch_size} images...")
            successful_generations = 0
            failed_generations = 0

            with tqdm(total=batch_size, desc="Generating images") as pbar:
                for i in range(batch_size):
                    try:
                        if use_ollama:
                            final_prompt = safe_generate_prompt_with_ollama(project_config['ollama'])
                        else:
                            final_prompt = prompt_config['prompt']
                            if not any(x in final_prompt.lower() for x in
                                       ['masterpiece:', 'best quality', 'photorealistic:']):
                                final_prompt = "(masterpiece:1.2), (photorealistic:1.4), (best quality), " + final_prompt

                        result = safe_generate_single_image(
                            server_url, final_prompt, negative_prompt, params
                        )

                        if result:
                            image_path, metadata_path = save_generation_result(
                                result, images_dir, metadata_dir, name_prefix
                            )
                            successful_generations += 1
                        else:
                            failed_generations += 1

                    except Exception as e:
                        print(f"\nFailed to generate image {i + 1}: {str(e)}")
                        failed_generations += 1

                    pbar.update(1)

                    # Check server health periodically
                    if i > 0 and i % 5 == 0:
                        _, is_healthy = check_server_health(server_url)
                        if not is_healthy:
                            print("\nServer health check failed, stopping batch")
                            break

                    # Add a small delay between requests
                    if i < batch_size - 1:
                        time.sleep(0.1)

            print(f"\nBatch generation summary:")
            print(f"Total requested: {batch_size}")
            print(f"Successfully generated: {successful_generations}")
            print(f"Failed generations: {failed_generations}")
            print(f"Images saved in: {images_dir}")
            print(f"Metadata saved in: {metadata_dir}")
        else:
            # Single image generation
            if use_ollama:
                final_prompt = safe_generate_prompt_with_ollama(project_config['ollama'])
            else:
                final_prompt = prompt_config['prompt']
                if not any(x in final_prompt.lower() for x in ['masterpiece:', 'best quality', 'photorealistic:']):
                    final_prompt = "(masterpiece:1.2), (photorealistic:1.4), (best quality), " + final_prompt

            result = safe_generate_single_image(
                server_url, final_prompt, negative_prompt, params
            )

            if result:
                image_path, metadata_path = save_generation_result(
                    result, images_dir, metadata_dir, name_prefix
                )
                print("\nOutput Details:")
                print(f"Image saved: {image_path}")
                print(f"Metadata saved: {metadata_path}")
                print("\nGeneration Settings:")
                for key, value in result['generation_settings'].items():
                    print(f"{key}: {value}")
                if show_image:
                    Image.open(image_path).show()
            else:
                raise RuntimeError("Failed to generate image")

    except Exception as e:
        print(f"Error during generation process: {e}")
        raise

def list_prompts(prompts_config):
    """List all available prompts in the config"""
    print("\nAvailable prompt sets:")
    for prompt_set in prompts_config.get('prompt_sets', []):
        print(f"- {prompt_set['name']}")
        print(f"  Prompt: {prompt_set['prompt']}")
        if 'params' in prompt_set:
            print("  Custom parameters:", prompt_set['params'])
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images using project configurations')
    parser.add_argument('--input', required=True, help='Input folder containing project folders')
    parser.add_argument('--project', required=True, help='Project name (folder name in input directory)')
    parser.add_argument('--prompt', help='Specific prompt set to use (not required with --ollama)')
    parser.add_argument('--output', default='outputs', help='Output directory for generated images')
    parser.add_argument('--list', action='store_true', help='List available prompts in the project')
    parser.add_argument('--batch', type=int, help='Number of images to generate in batch mode')
    parser.add_argument('--ollama', action='store_true', help='Use Ollama for dynamic prompt generation')
    parser.add_argument('--compare', action='store_true',
                        help='Generate images using all available models for comparison')

    args = parser.parse_args()

    try:
        config, prompts = load_project_config(args.input, args.project)

        if 'name' not in config:
            config['name'] = args.project

        if args.ollama:
            if 'ollama' not in config:
                raise ValueError("Ollama configuration not found in config.yaml. Please add 'ollama' section.")

            required_fields = ['server_url', 'model', 'system_message', 'prompt']
            missing_fields = [field for field in required_fields if field not in config['ollama']]
            if missing_fields:
                raise ValueError(
                    f"Missing required Ollama configuration fields in config.yaml: {', '.join(missing_fields)}\n"
                    f"Please ensure all required fields are included in the 'ollama' section."
                )

            if not config['ollama']['server_url']:
                raise ValueError(
                    "Ollama server_url cannot be empty in config.yaml.\n"
                    "Please specify the full URL (e.g., 'http://localhost:11434')"
                )

        if not config.get('server_url'):
            raise ValueError(
                "Stable Diffusion server_url not found in config.yaml.\n"
                "Please specify the full URL (e.g., 'http://192.168.1.101:8001')"
            )

        if args.list and not args.ollama:
            list_prompts(prompts)
        elif args.ollama:
            generate_image(config, args.output,
                           use_ollama=True,
                           batch_size=args.batch,
                           show_image=not bool(args.batch),
                           compare=args.compare)
        elif args.prompt:
            prompt_config = get_prompt_config(prompts, args.prompt)
            generate_image(config, args.output,
                           use_ollama=False,
                           prompt_config=prompt_config,
                           batch_size=args.batch,
                           show_image=not bool(args.batch),
                           compare=args.compare)
        else:
            print("Please specify either --ollama, --list to see available prompts, or --prompt to generate an image")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)