#!/usr/bin/env python3
"""
Diagnostic script to test InternVL3-8B multi-GPU device mapping
and identify why it produces gibberish responses.
"""

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer
from rich import print as rprint
import math


def diagnose_multi_gpu_setup():
    """Diagnose the multi-GPU setup and device mapping."""

    world_size = torch.cuda.device_count()
    rprint(f"[bold cyan]🔍 GPU Configuration:[/bold cyan]")
    rprint(f"Number of GPUs: {world_size}")

    total_memory = 0
    for i in range(world_size):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1e9
        total_memory += memory_gb
        rprint(f"  GPU {i}: {props.name} - {memory_gb:.1f}GB")

    rprint(f"Total GPU Memory: {total_memory:.1f}GB")
    return world_size, total_memory


def test_device_mappings(model_path="/home/jovyan/nfs_share/models/InternVL3-8B"):
    """Test different device mapping strategies."""

    world_size, total_memory = diagnose_multi_gpu_setup()

    # Test 1: Direct single GPU (baseline - will OOM but shows if model works)
    rprint("\n[bold yellow]Test 1: Single GPU Direct Loading (baseline)[/bold yellow]")
    rprint("[yellow]Expected: OOM during inference but clean responses if it works[/yellow]")

    # Test 2: Official split_model approach
    rprint("\n[bold yellow]Test 2: Official split_model() Device Map[/bold yellow]")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    rprint(f"Model has {num_layers} layers to distribute")

    # Official device map
    device_map = {}
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)

    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for _ in range(num_layer):
            if layer_cnt < num_layers:
                device_map[f'language_model.model.layers.{layer_cnt}'] = i
                layer_cnt += 1

    # Critical components on GPU 0
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    rprint("[cyan]Device distribution:[/cyan]")
    for gpu_id in range(world_size):
        components = sum(1 for v in device_map.values() if v == gpu_id)
        rprint(f"  GPU {gpu_id}: {components} components")

    # Test 3: Alternative - All on GPU 0 except vision on GPU 1
    rprint("\n[bold yellow]Test 3: Alternative Split (Vision on GPU 1, Rest on GPU 0)[/bold yellow]")
    alt_device_map = {
        'vision_model': 1,  # Move vision to GPU 1
        'mlp1': 0,
        'language_model': 0  # Everything else on GPU 0
    }
    rprint("Alternative mapping: Vision encoder isolated on GPU 1")

    # Test 4: Balanced split
    rprint("\n[bold yellow]Test 4: Balanced Split Across All GPUs[/bold yellow]")
    balanced_map = "balanced"  # Let HF figure it out

    return {
        "official": device_map,
        "alternative": alt_device_map,
        "balanced": balanced_map,
        "auto": "auto"
    }


def test_simple_inference(model, tokenizer, device_map_name="unknown"):
    """Test a simple inference to check for gibberish."""

    rprint(f"\n[bold green]Testing inference with {device_map_name} mapping:[/bold green]")

    # Simple test prompt
    prompt = "What is 2+2?"

    try:
        inputs = tokenizer(prompt, return_tensors="pt")
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        rprint(f"Response: {response}")

        # Check for gibberish
        if response.count('!') > len(response) * 0.5:
            rprint("[red]❌ GIBBERISH DETECTED![/red]")
            return False
        else:
            rprint("[green]✅ Clean response![/green]")
            return True

    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        return None


if __name__ == "__main__":
    rprint("[bold magenta]InternVL3-8B Multi-GPU Diagnostic[/bold magenta]")
    rprint("=" * 60)

    # Get device mappings
    device_maps = test_device_mappings()

    rprint("\n[bold cyan]Recommended next steps:[/bold cyan]")
    rprint("1. Try each device mapping strategy")
    rprint("2. Test with simple prompts to detect gibberish")
    rprint("3. Monitor GPU memory distribution with nvidia-smi")
    rprint("4. Check if model components are properly synchronized")

    rprint("\n[yellow]Key insight:[/yellow]")
    rprint("The gibberish likely comes from tensor synchronization issues")
    rprint("when model components are split across devices.")