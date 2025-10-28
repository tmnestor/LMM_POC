"""Reproducibility utilities for ensuring deterministic model behavior."""

import random

import numpy as np
import torch


def set_seed(seed: int = 42, verbose: bool = True) -> None:
    """
    Set random seeds for reproducibility across all libraries.

    This function sets seeds for:
    - Python's random module
    - NumPy's random generator
    - PyTorch (CPU and CUDA)

    Args:
        seed: Random seed value (default: 42)
        verbose: Print confirmation message (default: True)

    Example:
        >>> set_seed(42)
        ✅ Random seed set to 42 for reproducibility

        >>> set_seed(123, verbose=False)
        # Silent execution
    """
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seed (CPU)
    torch.manual_seed(seed)

    # Set PyTorch random seed (CUDA/GPU) if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if verbose:
        print(f"✅ Random seed set to {seed} for reproducibility")


def get_random_state() -> dict:
    """
    Get current random state for all libraries.

    Useful for debugging or verifying seed settings.

    Returns:
        Dictionary containing random states for Python, NumPy, and PyTorch

    Example:
        >>> set_seed(42)
        >>> state = get_random_state()
        >>> state['seed_set']
        True
    """
    return {
        'python_state': random.getstate(),
        'numpy_state': np.random.get_state(),
        'torch_cpu_state': torch.get_rng_state(),
        'torch_cuda_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        'seed_set': True
    }


def configure_deterministic_mode(enabled: bool = True, verbose: bool = True) -> None:
    """
    Configure PyTorch for deterministic operations.

    This enables additional PyTorch settings for maximum reproducibility,
    though it may impact performance.

    Args:
        enabled: Enable deterministic mode (default: True)
        verbose: Print configuration messages (default: True)

    Note:
        This may reduce performance. Use only when exact reproducibility
        is critical (e.g., debugging, paper experiments).

    Example:
        >>> configure_deterministic_mode(True)
        ✅ PyTorch deterministic mode enabled
        ⚠️ Note: May reduce performance
    """
    if enabled:
        # Use deterministic algorithms when possible
        torch.use_deterministic_algorithms(True, warn_only=True)

        # Disable CUDA benchmark mode (non-deterministic)
        torch.backends.cudnn.benchmark = False

        # Enable CUDA deterministic mode
        torch.backends.cudnn.deterministic = True

        if verbose:
            print("✅ PyTorch deterministic mode enabled")
            print("⚠️  Note: May reduce performance")
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        if verbose:
            print("✅ PyTorch deterministic mode disabled (performance optimized)")


def verify_seed_setting(seed: int = 42) -> bool:
    """
    Verify that seed setting produces reproducible results.

    Runs a simple test to check if random number generation is reproducible.

    Args:
        seed: Seed to test with (default: 42)

    Returns:
        True if reproducible, False otherwise

    Example:
        >>> set_seed(42)
        >>> verify_seed_setting(42)
        ✅ Seed verification passed - reproducibility confirmed
        True
    """
    # First run
    set_seed(seed, verbose=False)
    values1 = [random.random(), np.random.random(), torch.rand(1).item()]

    # Second run with same seed
    set_seed(seed, verbose=False)
    values2 = [random.random(), np.random.random(), torch.rand(1).item()]

    # Check if values match
    matches = all(abs(v1 - v2) < 1e-10 for v1, v2 in zip(values1, values2, strict=False))

    if matches:
        print("✅ Seed verification passed - reproducibility confirmed")
    else:
        print("❌ Seed verification failed - results not reproducible")

    return matches
