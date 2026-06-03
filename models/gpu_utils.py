"""GPU status display and model loading utilities.

Extracted from registry.py — purely mechanical move.
"""

from contextlib import contextmanager


@contextmanager
def quiet_loading():
    """Suppress tqdm progress bars during model weight loading."""
    from transformers.utils import logging as hf_logging

    hf_logging.disable_progress_bar()
    try:
        yield
    finally:
        hf_logging.enable_progress_bar()


def print_gpu_status(console) -> None:
    """Print GPU memory usage table after model loading."""
    import torch

    if not torch.cuda.is_available():
        return

    from rich.table import Table

    gpu_table = Table(
        title="GPU Status",
        show_header=True,
        header_style="bold cyan",
    )
    gpu_table.add_column("GPU", style="white")
    gpu_table.add_column("Total", justify="right", style="dim")
    gpu_table.add_column("Allocated", justify="right")
    gpu_table.add_column("Reserved", justify="right")
    gpu_table.add_column("Utilization", justify="right")

    for gpu_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(gpu_id)
        vram = props.total_memory / (1024**3)
        alloc = torch.cuda.memory_allocated(gpu_id) / (1024**3)
        resv = torch.cuda.memory_reserved(gpu_id) / (1024**3)
        util = (resv / vram) * 100 if vram > 0 else 0
        color = "green" if util < 50 else ("yellow" if util < 80 else "red")
        gpu_table.add_row(
            f"{gpu_id}: {props.name}",
            f"{vram:.1f} GB",
            f"{alloc:.2f} GB",
            f"{resv:.2f} GB",
            f"[{color}]{util:.1f}%[/{color}]",
        )
    console.print(gpu_table)
