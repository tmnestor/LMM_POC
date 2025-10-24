"""
InternVL3-8B V100 Multi-GPU Fix
Addresses gibberish responses on multi-GPU V100 setups.
"""

import torch
from rich import print as rprint
from transformers import AutoModel, AutoTokenizer


def load_internvl3_8b_for_v100_multi_gpu(
    model_path: str, verbose: bool = True
) -> tuple:
    """
    Special loading pattern for 4x V100 16GB GPUs.

    Strategy: Keep model on single GPU but use other GPUs for:
    - Intermediate activations
    - Gradient storage (if training)
    - Buffer overflow
    """

    world_size = torch.cuda.device_count()

    if verbose:
        rprint(f"[cyan]🔍 V100 Multi-GPU Fix: {world_size} GPUs detected[/cyan]")

    # Strategy 1: Try to fit entire model on 2 GPUs (32GB)
    # Vision + first half of language model on GPU 0
    # Second half of language model on GPU 1
    # GPUs 2-3 reserved for activations

    device_map = {
        "vision_model": 0,
        "mlp1": 0,
        "language_model.model.embed_tokens": 0,
        "language_model.model.tok_embeddings": 0,
    }

    # Try loading with this simplified device map
    if verbose:
        rprint("[yellow]🔧 Attempting V100-optimized device mapping...[/yellow]")
        rprint("[yellow]Strategy: Minimize cross-device communication[/yellow]")

    try:
        # First attempt: Sequential device map (less splitting)
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,  # V100-compatible (changed from bfloat16)
            low_cpu_mem_usage=True,
            use_flash_attn=False,  # V100 doesn't support Flash Attention
            trust_remote_code=True,
            device_map="sequential",  # Try sequential instead of auto
        ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=False
        )

        if verbose:
            rprint("[green]✅ Model loaded with sequential device mapping[/green]")

        return model, tokenizer

    except Exception as e:
        if verbose:
            rprint(f"[red]Sequential mapping failed: {e}[/red]")
            rprint("[yellow]Trying alternative approach...[/yellow]")

        # Fallback: Use balanced_low_0 to minimize GPU 0 usage
        try:
            model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.float16,  # V100-compatible (changed from bfloat16)
                low_cpu_mem_usage=True,
                use_flash_attn=False,
                trust_remote_code=True,
                device_map="balanced_low_0",  # Minimize GPU 0 load
            ).eval()

            tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, use_fast=False
            )

            if verbose:
                rprint("[green]✅ Model loaded with balanced_low_0 mapping[/green]")

            return model, tokenizer

        except Exception as e2:
            if verbose:
                rprint(f"[red]All device mappings failed: {e2}[/red]")
            raise


def test_inference_quality(model, tokenizer, verbose=True):
    """Quick test to check if responses are gibberish."""

    test_prompts = ["What is 2+2?", "Name three colors.", "Is water wet?"]

    clean_responses = 0

    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")

        # Ensure inputs are on same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,  # Deterministic for testing
                temperature=1.0,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Check for gibberish (excessive exclamation marks)
        gibberish_ratio = response.count("!") / max(len(response), 1)

        if verbose:
            rprint(f"Q: {prompt}")
            rprint(f"A: {response}")

        if gibberish_ratio < 0.3:  # Less than 30% exclamation marks
            clean_responses += 1
            if verbose:
                rprint("[green]✅ Clean response[/green]")
        else:
            if verbose:
                rprint(
                    f"[red]❌ Gibberish detected (ratio: {gibberish_ratio:.2f})[/red]"
                )

    success_rate = clean_responses / len(test_prompts)

    if verbose:
        rprint(f"\n[bold]Success rate: {success_rate:.1%}[/bold]")

    return success_rate > 0.5  # At least half should be clean


if __name__ == "__main__":
    # Test the fix
    rprint("[bold magenta]Testing InternVL3-8B V100 Multi-GPU Fix[/bold magenta]")

    model_path = "/home/jovyan/nfs_share/models/InternVL3-8B"

    try:
        model, tokenizer = load_internvl3_8b_for_v100_multi_gpu(model_path)

        rprint("\n[cyan]Testing response quality...[/cyan]")
        is_clean = test_inference_quality(model, tokenizer)

        if is_clean:
            rprint("\n[bold green]🎉 SUCCESS! Clean responses achieved![/bold green]")
        else:
            rprint("\n[bold red]❌ Still producing gibberish[/bold red]")

    except Exception as e:
        rprint(f"[red]Failed to load model: {e}[/red]")
