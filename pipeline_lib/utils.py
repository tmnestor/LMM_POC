"""
Utility functions for pipeline notebooks

Generic helper functions that can be used across different pipeline implementations.
"""

import pickle

import torch
from rich import print as rprint


def show_pipeline_memory(df, stage_name):
    """
    Show DataFrame and GPU memory usage with auto-scaling units.

    Args:
        df: DataFrame to measure
        stage_name: Name of the pipeline stage (for display)

    Example:
        >>> show_pipeline_memory(df, "After Stage 2")
        💾 After Stage 2: DataFrame=2.3MB, GPU=9.8GB (reserved 9.8GB)
    """
    # Get accurate DataFrame size via pickle
    df_bytes = len(pickle.dumps(df, protocol=pickle.HIGHEST_PROTOCOL))

    # Auto-scale to appropriate unit
    if df_bytes < 1024:
        df_size = f"{df_bytes}B"
    elif df_bytes < 1024**2:
        df_size = f"{df_bytes/1024:.1f}KB"
    elif df_bytes < 1024**3:
        df_size = f"{df_bytes/1024**2:.1f}MB"
    else:
        df_size = f"{df_bytes/1024**3:.2f}GB"

    # Get GPU memory if available
    if torch.cuda.is_available():
        gpu_allocated = torch.cuda.memory_allocated() / 1e9
        gpu_reserved = torch.cuda.memory_reserved() / 1e9
        rprint(f"[dim]   💾 {stage_name}: DataFrame={df_size}, GPU={gpu_allocated:.1f}GB (reserved {gpu_reserved:.1f}GB)[/dim]")
    else:
        rprint(f"[dim]   💾 {stage_name}: DataFrame={df_size}[/dim]")
