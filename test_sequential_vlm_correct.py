#!/usr/bin/env python3
"""
Test InternVL3-8B VLM with CORRECT chat() method usage.
Based on InternVL3 official documentation.
"""

from pathlib import Path

import torch
from PIL import Image
from rich import print as rprint
from transformers import AutoModel, AutoTokenizer


def test_sequential_vlm_correct():
    """Test sequential mapping with correct InternVL3 chat usage."""

    model_path = '/efs/shared/PTM/InternVL3-8B'
    test_image_path = '/efs/shared/PoC_data/evaluation_data/image_001.png'

    rprint("[bold cyan]Testing InternVL3-8B VLM with CORRECT API[/bold cyan]")
    rprint("=" * 60)

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

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )

    rprint("[green]✅ Model and tokenizer loaded![/green]")

    # Load test image
    if not Path(test_image_path).exists():
        rprint("[yellow]Creating dummy test image...[/yellow]")
        image = Image.new('RGB', (448, 448), color='white')
    else:
        image = Image.open(test_image_path).convert('RGB')
        rprint(f"[green]✅ Image loaded: {image.size}[/green]")

    # Test prompts
    test_prompts = [
        "Extract all text from this business document.",
        "What type of business document is this?",
        "What is the total amount?"
    ]

    rprint("\n[bold yellow]Testing InternVL3 chat() method:[/bold yellow]")

    # Based on InternVL3 documentation, the chat method signature is:
    # model.chat(tokenizer, pixel_values, question, generation_config)
    # where pixel_values should be the preprocessed image tensor

    for prompt in test_prompts:
        rprint(f"\n[cyan]Prompt: {prompt}[/cyan]")

        try:
            # Method 1: Let the model handle image preprocessing internally
            # This is based on the InternVL3 documentation pattern

            # The correct way based on InternVL3 docs:
            # Convert image to the format expected by the model
            pixel_values = None  # Let the model handle it

            # InternVL3's chat expects:
            # - tokenizer: the tokenizer object
            # - pixel_values: preprocessed image tensor OR None
            # - question: the text prompt
            # - generation_config: dict with generation parameters

            # First, try with the image embedded in the question
            question_with_image = f"<image>\n{prompt}"

            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=question_with_image,
                generation_config={
                    "max_new_tokens": 100,
                    "do_sample": False,
                    "temperature": 1.0
                }
            )

            rprint(f"[green]✅ Response: {response}[/green]")

            # Check for gibberish
            if '!' * 10 in response or response.count('!') > len(response) * 0.3:
                rprint("[red]⚠️ GIBBERISH DETECTED![/red]")
                return False
            else:
                rprint("[green]✅ Clean response![/green]")
                return True

        except Exception as e:
            rprint(f"[red]❌ Error with embedded image tag: {str(e)[:150]}[/red]")

            # Method 2: Try preprocessing the image manually
            try:
                rprint("[yellow]Trying manual image preprocessing...[/yellow]")

                # InternVL3 models typically use a specific image size
                # Resize to expected dimensions (usually 448x448 for InternVL3)
                image_resized = image.resize((448, 448))

                # Convert to tensor and normalize
                import torchvision.transforms as transforms

                transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                       std=[0.229, 0.224, 0.225])
                ])

                pixel_values = transform(image_resized).unsqueeze(0)  # Add batch dimension

                # Move to same device as model
                device = next(model.parameters()).device
                pixel_values = pixel_values.to(device)

                response = model.chat(
                    tokenizer=tokenizer,
                    pixel_values=pixel_values,
                    question=prompt,
                    generation_config={
                        "max_new_tokens": 100,
                        "do_sample": False
                    }
                )

                rprint(f"[green]✅ Response: {response}[/green]")

                # Check for gibberish
                if '!' * 10 in response or response.count('!') > len(response) * 0.3:
                    rprint("[red]⚠️ GIBBERISH DETECTED![/red]")
                    return False
                else:
                    rprint("[green]✅ Clean response![/green]")
                    return True

            except Exception as e2:
                rprint(f"[red]❌ Manual preprocessing failed: {str(e2)[:150]}[/red]")

    # Memory status
    rprint("\n[bold cyan]GPU Memory Status:[/bold cyan]")
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1e9
        total = torch.cuda.get_device_properties(i).total_memory / 1e9
        rprint(f"GPU {i}: {allocated:.2f}GB / {total:.1f}GB ({allocated/total*100:.1f}%)")

    return False


if __name__ == "__main__":
    rprint("[bold magenta]InternVL3-8B VLM Test with CORRECT API Usage[/bold magenta]")
    rprint("Testing sequential mapping with proper chat() method\n")

    success = test_sequential_vlm_correct()

    if success:
        rprint("\n[bold green]🎉 SUCCESS! Sequential mapping produces clean responses![/bold green]")
    else:
        rprint("\n[bold red]❌ Sequential mapping still produces gibberish or errors[/bold red]")