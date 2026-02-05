#!/usr/bin/env python3
"""
Robust GPU Memory Detection Module

Provides reliable GPU memory detection across different hardware configurations,
including multi-GPU setups, with comprehensive error handling and diagnostics.
"""

import torch


def get_total_gpu_memory() -> float:
    """
    Get total GPU memory across all available devices.

    Returns:
        float: Total GPU memory in GB, or 0.0 if no GPU available
    """
    if not torch.cuda.is_available():
        return 0.0

    try:
        total_memory = 0.0
        device_count = torch.cuda.device_count()

        for device_idx in range(device_count):
            props = torch.cuda.get_device_properties(device_idx)
            total_memory += props.total_memory / (1024**3)  # Convert to GB

        return total_memory
    except Exception:
        return 0.0


def get_total_available_gpu_memory() -> float:
    """
    Get total available (free) GPU memory across all devices.

    Returns:
        float: Available GPU memory in GB, or 0.0 if no GPU available
    """
    if not torch.cuda.is_available():
        return 0.0

    try:
        available_memory = 0.0
        device_count = torch.cuda.device_count()

        for device_idx in range(device_count):
            props = torch.cuda.get_device_properties(device_idx)
            total = props.total_memory
            allocated = torch.cuda.memory_allocated(device_idx)
            reserved = torch.cuda.memory_reserved(device_idx)

            # Available = Total - Reserved (reserved includes allocated + cached)
            available = total - reserved
            available_memory += available / (1024**3)  # Convert to GB

        return available_memory
    except Exception:
        return 0.0


def get_device_memory_info(device_idx: int = 0) -> dict:
    """
    Get detailed memory information for a specific GPU device.

    Args:
        device_idx: GPU device index

    Returns:
        dict: Memory information including total, allocated, reserved, available
    """
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    try:
        props = torch.cuda.get_device_properties(device_idx)
        total = props.total_memory
        allocated = torch.cuda.memory_allocated(device_idx)
        reserved = torch.cuda.memory_reserved(device_idx)
        available = total - reserved

        return {
            "device_idx": device_idx,
            "device_name": props.name,
            "total_gb": total / (1024**3),
            "allocated_gb": allocated / (1024**3),
            "reserved_gb": reserved / (1024**3),
            "available_gb": available / (1024**3),
            "utilization_percent": (reserved / total) * 100 if total > 0 else 0,
        }
    except Exception as e:
        return {"error": str(e), "device_idx": device_idx}


def diagnose_gpu_memory(verbose: bool = True) -> dict:
    """
    Comprehensive GPU memory diagnostics.

    Args:
        verbose: Enable detailed output to console

    Returns:
        dict: Complete diagnostic information for all GPUs
    """
    diagnostics = {
        "detection_successful": False,
        "cuda_available": torch.cuda.is_available(),
        "device_count": 0,
        "devices": [],
        "total_memory_gb": 0.0,
        "total_available_gb": 0.0,
        "recommendations": [],
    }

    if not torch.cuda.is_available():
        diagnostics["error"] = "CUDA not available"
        diagnostics["recommendations"].append("Install CUDA-enabled PyTorch")
        if verbose:
            print("❌ CUDA not available")
        return diagnostics

    try:
        device_count = torch.cuda.device_count()
        diagnostics["device_count"] = device_count
        diagnostics["cuda_version"] = torch.version.cuda

        if verbose:
            print(f"🔍 Found {device_count} GPU(s), CUDA {torch.version.cuda}")

        total_memory = 0.0
        total_available = 0.0

        for device_idx in range(device_count):
            device_info = get_device_memory_info(device_idx)
            diagnostics["devices"].append(device_info)

            if "error" not in device_info:
                total_memory += device_info["total_gb"]
                total_available += device_info["available_gb"]

                if verbose:
                    print(
                        f"  GPU {device_idx}: {device_info['device_name']}"
                        f" - {device_info['total_gb']:.1f}GB total,"
                        f" {device_info['available_gb']:.1f}GB available"
                        f" ({device_info['utilization_percent']:.1f}% used)"
                    )

        diagnostics["total_memory_gb"] = total_memory
        diagnostics["total_available_gb"] = total_available
        diagnostics["detection_successful"] = True

        # Generate recommendations based on memory state
        if total_available < 2.0:
            diagnostics["recommendations"].append(
                "Low GPU memory available - consider clearing cache"
            )
        if device_count > 1:
            diagnostics["recommendations"].append(
                f"Multi-GPU setup detected ({device_count} devices)"
            )

        if verbose and diagnostics["recommendations"]:
            for rec in diagnostics["recommendations"]:
                print(f"  💡 {rec}")

    except Exception as e:
        diagnostics["error"] = str(e)
        diagnostics["recommendations"].append(f"Diagnostic error: {e}")
        if verbose:
            print(f"❌ Diagnostic error: {e}")

    return diagnostics
