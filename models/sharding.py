"""InternVL model sharding utilities.

Extracted from registry.py — purely mechanical move.

Provides device_map generation for splitting InternVL models across GPUs.
"""


def split_internvl_model(model_path: str) -> dict[str, int]:
    """Build a device_map that shards an InternVL model across all GPUs.

    GPU 0 hosts the vision encoder, MLP projector, embeddings, norm, and
    output head — so it gets fewer LLM layers.  The remaining layers are
    distributed evenly across all GPUs.  The final LLM layer is pinned back
    to GPU 0 to avoid cross-device tensor mismatches during generation.

    Reference: OpenGVLab/InternVL official multi-GPU example.
    """
    import math

    import torch
    from transformers import AutoConfig

    world_size = torch.cuda.device_count()
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = cfg.llm_config.num_hidden_layers

    # GPU 0 counts as "half" because it also hosts the vision model
    layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    per_gpu = [layers_per_gpu] * world_size
    per_gpu[0] = math.ceil(per_gpu[0] * 0.5)

    device_map: dict[str, int] = {}
    layer_idx = 0
    for gpu_id, n_layers in enumerate(per_gpu):
        for _ in range(n_layers):
            if layer_idx >= num_layers:
                break
            device_map[f"language_model.model.layers.{layer_idx}"] = gpu_id
            layer_idx += 1

    # Fixed components on GPU 0
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    # Pin last layer to GPU 0 to prevent cross-device errors in generation
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map
