# Standard Attention vs FlashAttention2

## Overview

FlashAttention2 fundamentally changes **where** and **how** attention is computed on the GPU.
The algorithm produces identical results — the difference is purely in memory access patterns and hardware utilisation.

---

## GPU Memory Hierarchy

Understanding the hardware is key to understanding why FlashAttention2 is faster.

```mermaid
flowchart LR
    SRAM["<b>SRAM (On-Chip)</b><br/>≈20 MB capacity<br/>≈19 TB/s bandwidth<br/>Ultra-fast, tiny"]
    SRAM <--->|"≈30x faster<br/>bandwidth gap"| HBM
    HBM["<b>HBM (VRAM)</b><br/>24 GB capacity (A10G)<br/>≈600 GB/s bandwidth<br/>Slow, large"]

    style SRAM fill:#4CAF50,color:#fff,stroke:#2E7D32,stroke-width:3px
    style HBM fill:#FF9800,color:#fff,stroke:#E65100,stroke-width:3px
```

> **The bottleneck is not compute — it's memory bandwidth.**
> The GPU can compute far faster than it can move data between HBM and SRAM.

---

## Standard Attention — Step by Step

Standard attention materialises the full N×N attention matrix in HBM (VRAM),
requiring **multiple round trips** between slow HBM and fast SRAM.

```mermaid
flowchart LR
    QKV["<b>Q, K, V</b><br/>in HBM"]
    QKV -->|"① Read Q, K"| C1["<b>SRAM</b><br/>S = QKᵀ / √d"]
    C1 -->|"② Write S back"| S_HBM["<b>S (N×N)</b><br/>in HBM<br/>⚠️ O(N²)"]
    S_HBM -->|"③ Read S"| C2["<b>SRAM</b><br/>P = softmax(S)"]
    C2 -->|"④ Write P back"| P_HBM["<b>P (N×N)</b><br/>in HBM<br/>⚠️ O(N²)"]
    P_HBM -->|"⑤ Read P, V"| C3["<b>SRAM</b><br/>O = P × V"]
    C3 -->|"⑥ Write output"| Output["<b>Output O</b><br/>in HBM"]

    style S_HBM fill:#FF5252,color:#fff,stroke:#B71C1C,stroke-width:3px
    style P_HBM fill:#FF5252,color:#fff,stroke:#B71C1C,stroke-width:3px
    style QKV fill:#FF9800,color:#fff,stroke:#E65100,stroke-width:2px
    style Output fill:#4CAF50,color:#fff,stroke:#2E7D32,stroke-width:2px
    style C1 fill:#BBDEFB,stroke:#1565C0,stroke-width:2px
    style C2 fill:#BBDEFB,stroke:#1565C0,stroke-width:2px
    style C3 fill:#BBDEFB,stroke:#1565C0,stroke-width:2px
```

**6 HBM transfers** — each one is slow. Two massive N×N matrices stored in VRAM.

---

## FlashAttention2 — Step by Step

FlashAttention2 **tiles** the computation so the N×N matrix is never fully materialised.
Everything stays in fast SRAM using an online softmax algorithm.

```mermaid
flowchart LR
    QKV["<b>Q, K, V</b><br/>in HBM"]
    QKV -->|"① Load tiles"| Tile["<b>SRAM</b><br/>Load Q, K, V tiles"]

    subgraph FUSED["Fused SRAM Kernel (on-chip)"]
        direction LR
        S1["S_tile =<br/>Q × Kᵀ / √d"]
        S2["Online<br/>softmax"]
        S3["O_tile =<br/>P × V"]
        S4["Accumulate<br/>output"]
        S1 --> S2 --> S3 --> S4
    end

    Tile --> S1
    S4 -.->|"loop tiles"| Tile
    S4 -->|"② Write final output"| Out["<b>Output O</b><br/>in HBM"]

    style QKV fill:#FF9800,color:#fff,stroke:#E65100,stroke-width:2px
    style Out fill:#4CAF50,color:#fff,stroke:#2E7D32,stroke-width:2px
    style Tile fill:#BBDEFB,stroke:#1565C0,stroke-width:2px
    style FUSED fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px
    style S1 fill:#C8E6C9,stroke:#2E7D32
    style S2 fill:#C8E6C9,stroke:#2E7D32
    style S3 fill:#C8E6C9,stroke:#2E7D32
    style S4 fill:#C8E6C9,stroke:#2E7D32
```

**2 HBM transfers** — read once, write once. The N×N matrix **never exists** in VRAM.

---

## Side-by-Side Comparison

```mermaid
flowchart LR
    subgraph Standard["⚠️ Standard Attention"]
        direction TB
        S_Q["Q, K, V in HBM"] --> S_attn["Compute QKᵀ"]
        S_attn --> S_NxN["N×N matrix in HBM"]
        S_NxN --> S_soft["Softmax"]
        S_soft --> S_NxN2["N×N matrix in HBM"]
        S_NxN2 --> S_mul["Multiply by V"]
        S_mul --> S_out["Output"]
    end

    Standard ~~~|"vs"| Flash

    subgraph Flash["✅ FlashAttention2"]
        direction TB
        F_Q["Q, K, V in HBM"] --> F_tile["Load tiles into SRAM"]
        F_tile --> F_fused["Fused: QKᵀ + softmax + V"]
        F_fused -.->|"loop tiles"| F_tile
        F_fused --> F_out["Output"]
    end

    style S_NxN fill:#FF5252,color:#fff,stroke:#B71C1C,stroke-width:2px
    style S_NxN2 fill:#FF5252,color:#fff,stroke:#B71C1C,stroke-width:2px
    style S_out fill:#4CAF50,color:#fff,stroke:#2E7D32,stroke-width:2px
    style F_out fill:#4CAF50,color:#fff,stroke:#2E7D32,stroke-width:2px
    style F_fused fill:#C8E6C9,stroke:#2E7D32,stroke-width:2px
    style Standard fill:#FFF3E0,stroke:#E65100,stroke-width:2px
    style Flash fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px
```

---

## Memory Usage — InternVL3.5-8B on A10G (24 GB)

How the available VRAM budget changes based on attention implementation:

```mermaid
flowchart LR
    subgraph STD["Standard Attention"]
        direction LR
        SM_model["<b>Model Weights</b><br/>≈17 GB"]
        SM_attn["<b>N×N Attention</b><br/><b>Matrices</b><br/>≈1-2 GB"]
        SM_kv["<b>KV Cache +</b><br/><b>Activations</b><br/>≈4 GB"]
        SM_free["<b>Free</b><br/>≈1-2 GB"]
        SM_model ~~~ SM_attn ~~~ SM_kv ~~~ SM_free
    end

    STD ~~~|"vs"| FA2

    subgraph FA2["FlashAttention2"]
        direction LR
        FM_model["<b>Model Weights</b><br/>≈17 GB"]
        FM_kv["<b>KV Cache +</b><br/><b>Activations</b><br/>≈2 GB"]
        FM_free["<b>Free</b><br/>≈5-6 GB"]
        FM_model ~~~ FM_kv ~~~ FM_free
    end

    style SM_attn fill:#FF5252,color:#fff,stroke:#B71C1C,stroke-width:2px
    style SM_free fill:#FFF9C4,stroke:#F57F17,stroke-width:2px
    style FM_free fill:#C8E6C9,stroke:#2E7D32,stroke-width:3px
    style STD fill:#FFF3E0,stroke:#E65100,stroke-width:2px
    style FA2 fill:#E8F5E9,stroke:#2E7D32,stroke-width:2px
    style SM_model fill:#90CAF9,stroke:#1565C0
    style FM_model fill:#90CAF9,stroke:#1565C0
    style SM_kv fill:#CE93D8,stroke:#6A1B9A
    style FM_kv fill:#CE93D8,stroke:#6A1B9A
```

---

## Impact on Micro-Batch Size

| Metric | Standard Attention | FlashAttention2 |
|--------|-------------------|-----------------|
| Attention memory | O(N²) — full N×N in VRAM | O(N) — tiles in SRAM only |
| HBM round trips | 6 per attention layer | 2 per attention layer |
| Speed | Memory-bandwidth bound | Compute bound (ideal) |
| A10G batch size (InternVL3.5) | **~2 images** | **~3–4 images** |
| A10G throughput | Baseline | **~2–3x faster** |

---

## The Online Softmax Trick

The key algorithmic insight enabling FlashAttention2 — computing exact softmax without seeing all values at once:

```mermaid
flowchart LR
    subgraph T1["Tile 1 — Process first block"]
        direction TB
        T1a["Scores: 2.1, 0.5, 1.3"]
        T1b["max = 2.1"]
        T1c["sum = exp(2.1-2.1) + exp(0.5-2.1) + exp(1.3-2.1)"]
        T1a --> T1b --> T1c
    end

    T1 -->|"pass running<br/>max + sum"| T2

    subgraph T2["Tile 2 — Update with new block"]
        direction TB
        T2a["Scores: 3.0, 0.8, 1.7"]
        T2b["new max = max(2.1, 3.0) = 3.0"]
        T2c["Rescale: old_sum × exp(2.1 - 3.0)"]
        T2d["new_sum = rescaled + exp(3.0-3.0) + ..."]
        T2a --> T2b --> T2c --> T2d
    end

    T2 --> R["<b>Exact softmax</b><br/>Mathematically identical<br/>to full computation"]

    style T1 fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    style T2 fill:#E3F2FD,stroke:#1565C0,stroke-width:2px
    style R fill:#C8E6C9,stroke:#2E7D32,stroke-width:3px
```

Each tile updates a **running maximum** and **running sum**, rescaling previous results.
The final output is **mathematically identical** to computing softmax over the full row.

---

## Summary

| | Standard | FlashAttention2 |
|--|---------|-----------------|
| N×N matrix in VRAM | Yes | Never |
| Memory scaling | O(N²) | O(N) |
| Computation location | HBM ↔ SRAM ping-pong | Fused in SRAM |
| Bottleneck | Memory bandwidth | Compute (ideal) |
| Result | Identical | Identical |
| Implementation | PyTorch default | `pip install flash-attn` |

> FlashAttention2 doesn't change **what** is computed — it changes **where** it's computed.
> By keeping work in fast SRAM and eliminating HBM round trips, it unlocks both speed and memory savings.
