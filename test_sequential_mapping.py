#!/usr/bin/env python3
"""
Test sequential device mapping for InternVL3-8B on 4x V100 setup.
This tests if sequential mapping can avoid the gibberish issue.
"""

import torch
from rich import print as rprint
from transformers import AutoModel, AutoTokenizer


def test_sequential_mapping():
    """Test if sequential device mapping works for InternVL3-8B."""

    model_path = '/efs/shared/PTM/InternVL3-8B'

    rprint("[bold cyan]Testing Sequential Device Mapping for InternVL3-8B[/bold cyan]")
    rprint("=" * 60)

    # Check GPU configuration
    world_size = torch.cuda.device_count()
    rprint(f"[cyan]GPUs detected: {world_size}[/cyan]")

    for i in range(world_size):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        rprint(f"  GPU {i}: {props.name} - {memory_gb:.1f}GB")

    rprint("\n[yellow]Loading model with sequential device mapping...[/yellow]")

    try:
        # Load with sequential mapping
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=False,  # V100 doesn't support Flash Attention
            trust_remote_code=True,
            device_map='sequential'
        ).eval()

        rprint("[green]✅ Model loaded successfully with sequential mapping![/green]")

        # Check device placement
        try:
            first_param_device = next(model.parameters()).device
            rprint(f"[cyan]First parameter device: {first_param_device}[/cyan]")
        except StopIteration:
            rprint("[yellow]Could not determine parameter device[/yellow]")

        # Load tokenizer
        rprint("\n[yellow]Loading tokenizer...[/yellow]")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        rprint("[green]✅ Tokenizer loaded successfully![/green]")

        # Test simple inference
        rprint("\n[bold yellow]Testing inference quality:[/bold yellow]")
        test_prompts = [
            "What is 2+2?",
            "Name three colors.",
            "What is the capital of France?"
        ]

        for prompt in test_prompts:
            rprint(f"\n[cyan]Prompt: {prompt}[/cyan]")

            try:
                # Prepare inputs
                inputs = tokenizer(prompt, return_tensors="pt")

                # Move inputs to appropriate device
                if torch.cuda.is_available():
                    # For sequential mapping, inputs should go to cuda:0
                    inputs = {k: v.to('cuda:0') for k, v in inputs.items()}

                # Generate response
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=30,
                        do_sample=False,  # Deterministic
                        temperature=1.0
                    )

                # Decode response
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Remove the prompt from response if present
                if prompt in response:
                    response = response.replace(prompt, "").strip()

                rprint(f"[green]Response: {response}[/green]")

                # Check for gibberish
                gibberish_chars = response.count('!') + response.count('?')
                if gibberish_chars > len(response) * 0.5:
                    rprint("[red]⚠️ Warning: Possible gibberish detected![/red]")
                else:
                    rprint("[green]✅ Clean response![/green]")

            except Exception as e:
                rprint(f"[red]❌ Inference error: {e}[/red]")

        # Memory status after loading
        rprint("\n[bold cyan]GPU Memory Status:[/bold cyan]")
        for i in range(world_size):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            rprint(f"GPU {i}: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved / {total:.1f}GB total")

        rprint("\n[bold green]✅ Sequential mapping test complete![/bold green]")
        return True

    except Exception as e:
        rprint("\n[red]❌ Failed to load model with sequential mapping:[/red]")
        rprint(f"[red]Error: {e}[/red]")

        # Try alternative mappings
        rprint("\n[yellow]Alternative device mappings to try:[/yellow]")
        rprint("1. 'balanced' - Balanced distribution")
        rprint("2. 'balanced_low_0' - Minimize GPU 0 usage")
        rprint("3. Custom mapping with vision on GPU 1")

        return False


if __name__ == "__main__":
    rprint("[bold magenta]InternVL3-8B Sequential Device Mapping Test[/bold magenta]")
    rprint("Testing if sequential mapping avoids gibberish on 4x V100\n")

    success = test_sequential_mapping()

    if success:
        rprint("\n[bold green]🎉 Sequential mapping appears to work![/bold green]")
        rprint("This could be the solution for 4x V100 setups!")
    else:
        rprint("\n[bold red]Sequential mapping failed or produced gibberish.[/bold red]")
        rprint("Need to explore other device mapping strategies.")