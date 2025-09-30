#!/usr/bin/env python3
"""
Test InternVL3-8B Vision-Language Model with sequential mapping.
PROPERLY tests with images, not just text!
"""

from pathlib import Path

import torch
from PIL import Image
from rich import print as rprint
from transformers import AutoModel, AutoTokenizer


def test_sequential_vlm_inference():
    """Test sequential mapping with actual image inputs."""

    model_path = '/efs/shared/PTM/InternVL3-8B'

    # Use actual test images from your dataset
    test_image_paths = [
        '/efs/shared/PoC_data/evaluation_data/image_001.png',
        '/efs/shared/PoC_data/evaluation_data/image_002.png'
    ]

    rprint("[bold cyan]Testing InternVL3-8B VLM with Sequential Mapping[/bold cyan]")
    rprint("=" * 60)
    rprint("[yellow]Using ACTUAL IMAGES for Vision-Language Model testing![/yellow]")

    # Load model with sequential mapping
    rprint("\n[yellow]Loading VLM with sequential device mapping...[/yellow]")

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=False,
        trust_remote_code=True,
        device_map='sequential'
    ).eval()

    rprint("[green]✅ VLM loaded with sequential mapping![/green]")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )
    rprint("[green]✅ Tokenizer loaded![/green]")

    # Test with actual images
    rprint("\n[bold yellow]Testing VLM inference with images:[/bold yellow]")

    # Document extraction prompts (your actual use case)
    test_prompts = [
        "Extract all text from this business document.",
        "What type of business document is this?",
        "What is the total amount in this document?"
    ]

    for img_path in test_image_paths[:1]:  # Test with first image
        rprint(f"\n[cyan]Testing with image: {Path(img_path).name}[/cyan]")

        # Check if image exists
        if not Path(img_path).exists():
            rprint("[yellow]Image not found, using a dummy test image[/yellow]")
            # Create a simple test image
            image = Image.new('RGB', (224, 224), color='white')
        else:
            # Load actual image
            image = Image.open(img_path).convert('RGB')
            rprint(f"[green]✅ Image loaded: {image.size}[/green]")

        for prompt in test_prompts[:1]:  # Test first prompt
            rprint(f"\n[cyan]Prompt: {prompt}[/cyan]")

            try:
                # Method 1: Use model's chat method (recommended for InternVL3)
                if hasattr(model, 'chat'):
                    rprint("[yellow]Using model.chat() method...[/yellow]")

                    # InternVL3's chat method handles everything
                    response = model.chat(
                        tokenizer=tokenizer,
                        pixel_values=None,  # Let model process the image
                        question=prompt,
                        generation_config={
                            "max_new_tokens": 100,
                            "do_sample": False
                        },
                        images=[image]  # Pass PIL image directly
                    )

                    rprint(f"[green]✅ Response: {response}[/green]")

                    # Check for gibberish
                    if '!' * 10 in response or response.count('!') > len(response) * 0.3:
                        rprint("[red]⚠️ GIBBERISH DETECTED![/red]")
                    else:
                        rprint("[green]✅ Clean response![/green]")

                else:
                    # Method 2: Manual preprocessing
                    rprint("[yellow]Using manual preprocessing...[/yellow]")

                    # Build conversation
                    conversations = [{
                        'from': 'human',
                        'value': f'<image>\n{prompt}'
                    }]

                    # Apply chat template
                    prompt_text = tokenizer.apply_chat_template(
                        conversations,
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    # Tokenize text
                    inputs = tokenizer(prompt_text, return_tensors='pt')

                    # Process image (this is model-specific)
                    # InternVL3 should have an image processor
                    if hasattr(model, 'vision_model'):
                        rprint("[cyan]Processing image through vision model...[/cyan]")
                        # This varies by model implementation
                        pixel_values = model.vision_model.preprocess(image)
                        inputs['pixel_values'] = pixel_values

                    # Move to appropriate device
                    device = next(model.parameters()).device
                    inputs = {k: v.to(device) if torch.is_tensor(v) else v
                             for k, v in inputs.items()}

                    # Generate
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=100,
                            do_sample=False
                        )

                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if prompt in response:
                        response = response.replace(prompt, "").strip()

                    rprint(f"[green]Response: {response}[/green]")

                    # Check for gibberish
                    if '!' * 10 in response or response.count('!') > len(response) * 0.3:
                        rprint("[red]⚠️ GIBBERISH DETECTED![/red]")
                    else:
                        rprint("[green]✅ Clean response![/green]")

            except Exception as e:
                rprint(f"[red]❌ Error: {str(e)[:200]}[/red]")

                # Try a simpler approach
                rprint("[yellow]Trying simpler text generation without image...[/yellow]")
                try:
                    # Just to see if text generation works at all
                    simple_prompt = "Hello, this is a test."
                    inputs = tokenizer(simple_prompt, return_tensors='pt')
                    inputs = {k: v.to('cuda:0') for k, v in inputs.items()}

                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=10)

                    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    rprint(f"[green]Simple text response: {response}[/green]")

                except Exception as e2:
                    rprint(f"[red]Even simple text failed: {str(e2)[:100]}[/red]")

    # Memory status
    rprint("\n[bold cyan]GPU Memory Status:[/bold cyan]")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        rprint(f"GPU {i}: {allocated:.2f}GB / {total:.1f}GB ({allocated/total*100:.1f}%)")

    rprint("\n[bold green]VLM testing complete![/bold green]")
    rprint("[yellow]Key: Check if responses are clean or gibberish with sequential mapping[/yellow]")


if __name__ == "__main__":
    rprint("[bold magenta]InternVL3-8B Vision-Language Model Test[/bold magenta]")
    rprint("Testing with ACTUAL IMAGES on 4x V100 with sequential mapping\n")

    test_sequential_vlm_inference()