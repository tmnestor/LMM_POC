"""
Dynamic GPU Configuration System

Eliminates hardcoded GPU specifications and provides dynamic detection
and configuration for H200, L40S, V100, and other GPU types.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class GPUConfig:
    """Dynamic GPU configuration based on detected hardware."""
    name: str
    memory_gb: float
    compute_capability: tuple
    architecture: str
    memory_buffer_gb: float
    optimal_batch_size: int
    max_tiles: int
    supports_flash_attention: bool
    quantization_threshold_gb: float
    is_high_memory: bool

class GPUDetector:
    """Dynamically detect and configure GPU settings without hardcoding."""

    def __init__(self):
        self.gpu_configs = {}
        self._detect_gpus()

    def _detect_gpus(self):
        """Detect all available GPUs and their configurations."""
        if not torch.cuda.is_available():
            return

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            config = self._create_gpu_config(props, i)
            self.gpu_configs[i] = config

    def _create_gpu_config(self, props, device_id: int) -> GPUConfig:
        """Create dynamic GPU configuration based on detected properties."""
        name = props.name
        memory_gb = props.total_memory / 1e9
        compute_capability = (props.major, props.minor)

        # Dynamic architecture detection
        architecture = self._detect_architecture(name, compute_capability)

        # Dynamic memory thresholds (no hardcoding!)
        memory_buffer_gb = self._calculate_memory_buffer(memory_gb, architecture)
        quantization_threshold_gb = self._calculate_quantization_threshold(memory_gb)

        # Dynamic performance settings
        optimal_batch_size = self._calculate_optimal_batch_size(memory_gb)
        max_tiles = self._calculate_max_tiles(memory_gb)

        # Feature detection
        supports_flash_attention = compute_capability >= (7, 5)  # SM 7.5+
        is_high_memory = memory_gb >= 40  # Dynamic threshold

        return GPUConfig(
            name=name,
            memory_gb=memory_gb,
            compute_capability=compute_capability,
            architecture=architecture,
            memory_buffer_gb=memory_buffer_gb,
            optimal_batch_size=optimal_batch_size,
            max_tiles=max_tiles,
            supports_flash_attention=supports_flash_attention,
            quantization_threshold_gb=quantization_threshold_gb,
            is_high_memory=is_high_memory
        )

    def _detect_architecture(self, name: str, compute_capability: tuple) -> str:
        """Dynamically detect GPU architecture without hardcoding."""
        name_upper = name.upper()

        # High-memory datacenter GPUs
        if any(gpu in name_upper for gpu in ['H200', 'H100', 'A100']):
            return 'datacenter_high_memory'

        # Professional workstation GPUs
        elif any(gpu in name_upper for gpu in ['L40S', 'L40', 'RTX A6000']):
            return 'workstation_high_memory'

        # Cloud/inference GPUs (A10, A10G, L4 - 24GB GDDR6)
        # Note: L4 has 24GB nominal but PyTorch reports ~22GB usable
        elif any(gpu in name_upper for gpu in ['A10G', 'A10', 'NVIDIA L4']):
            return 'cloud_inference'

        # Legacy datacenter GPUs
        elif any(gpu in name_upper for gpu in ['V100', 'P100']):
            return 'legacy_datacenter'

        # Consumer high-end GPUs
        elif any(gpu in name_upper for gpu in ['RTX 4090', 'RTX 3090']):
            return 'consumer_high_end'

        # Detect by compute capability as fallback
        elif compute_capability >= (8, 0):
            return 'modern_high_performance'
        elif compute_capability >= (7, 0):
            return 'modern_standard'
        else:
            return 'legacy'

    def _calculate_memory_buffer(self, memory_gb: float, architecture: str) -> float:
        """Calculate appropriate memory buffer based on GPU characteristics."""
        if architecture == 'datacenter_high_memory':
            return min(20.0, memory_gb * 0.15)  # 15% buffer, max 20GB
        elif architecture in ['workstation_high_memory', 'consumer_high_end']:
            return min(12.0, memory_gb * 0.20)  # 20% buffer, max 12GB
        elif architecture == 'cloud_inference':
            return min(4.0, memory_gb * 0.15)   # 15% buffer, max 4GB (A10/A10G optimized for inference)
        elif architecture == 'legacy_datacenter':
            return min(4.0, memory_gb * 0.25)   # 25% buffer, max 4GB
        else:
            return min(8.0, memory_gb * 0.20)   # 20% buffer, max 8GB

    def _calculate_quantization_threshold(self, memory_gb: float) -> float:
        """Calculate when quantization becomes necessary."""
        # Dynamic threshold: use quantization if model would use >70% of memory
        return memory_gb * 0.70

    def _calculate_optimal_batch_size(self, memory_gb: float) -> int:
        """Calculate optimal batch size based on available memory."""
        if memory_gb >= 80:   # H200, H100-80GB
            return 4
        elif memory_gb >= 40: # L40S, A100-40GB
            return 2
        elif memory_gb >= 20: # RTX 4090, RTX 3090
            return 1
        else:                 # V100, RTX 3080, etc.
            return 1

    def _calculate_max_tiles(self, memory_gb: float) -> int:
        """Calculate maximum tiles for vision processing."""
        # Scale tiles with available memory
        if memory_gb >= 80:
            return 32
        elif memory_gb >= 40:
            return 24
        elif memory_gb >= 20:
            return 16
        else:
            return 12  # Conservative for limited memory

    def get_primary_gpu_config(self) -> Optional[GPUConfig]:
        """Get configuration for primary GPU (device 0)."""
        return self.gpu_configs.get(0)

    def get_total_memory_gb(self) -> float:
        """Get total memory across all GPUs."""
        return sum(config.memory_gb for config in self.gpu_configs.values())

    def get_gpu_count(self) -> int:
        """Get number of available GPUs."""
        return len(self.gpu_configs)

    def should_use_quantization(self, model_size_gb: float) -> bool:
        """Dynamically determine if quantization is needed."""
        if not self.gpu_configs:
            return True  # Default to quantization if no GPU

        total_memory = self.get_total_memory_gb()
        primary_config = self.get_primary_gpu_config()

        # Use quantization if model would exceed threshold on any single GPU
        return model_size_gb > primary_config.quantization_threshold_gb

    def get_memory_strategy(self, model_size_gb: float) -> Dict[str, Any]:
        """Get comprehensive memory strategy for model loading."""
        config = self.get_primary_gpu_config()
        if not config:
            return {"strategy": "cpu_fallback"}

        use_quantization = self.should_use_quantization(model_size_gb)

        return {
            "strategy": "multi_gpu" if self.get_gpu_count() > 1 else "single_gpu",
            "use_quantization": use_quantization,
            "device_map": "auto" if self.get_gpu_count() > 1 else "cuda:0",
            "memory_buffer_gb": config.memory_buffer_gb,
            "batch_size": config.optimal_batch_size,
            "max_tiles": config.max_tiles,
            "supports_flash_attention": config.supports_flash_attention,
            "architecture": config.architecture,
            "total_memory_gb": self.get_total_memory_gb(),
            "gpu_count": self.get_gpu_count()
        }


# Global instance for easy access
_gpu_detector = None

def get_gpu_detector() -> GPUDetector:
    """Get global GPU detector instance."""
    global _gpu_detector
    if _gpu_detector is None:
        _gpu_detector = GPUDetector()
    return _gpu_detector

def get_gpu_config() -> Optional[GPUConfig]:
    """Get primary GPU configuration."""
    return get_gpu_detector().get_primary_gpu_config()

def get_memory_strategy(model_size_gb: float) -> Dict[str, Any]:
    """Get memory strategy for model of given size."""
    return get_gpu_detector().get_memory_strategy(model_size_gb)


def main():
    """Test dynamic GPU detection."""
    detector = GPUDetector()

    print("🔍 Dynamic GPU Detection Results:")
    print("=" * 50)

    if detector.get_gpu_count() == 0:
        print("❌ No CUDA GPUs detected")
        return

    for i, config in detector.gpu_configs.items():
        print(f"\n🎮 GPU {i}: {config.name}")
        print(f"   💾 Memory: {config.memory_gb:.1f}GB")
        print(f"   🏗️ Architecture: {config.architecture}")
        print(f"   🔧 Compute: {config.compute_capability}")
        print(f"   📦 Batch Size: {config.optimal_batch_size}")
        print(f"   🎯 Max Tiles: {config.max_tiles}")
        print(f"   ⚡ Flash Attention: {'✅' if config.supports_flash_attention else '❌'}")
        print(f"   🧠 High Memory: {'✅' if config.is_high_memory else '❌'}")

    print(f"\n📊 Total Memory: {detector.get_total_memory_gb():.1f}GB")

    # Test memory strategies
    print("\n🎯 Memory Strategies:")
    for model_size in [4, 8, 16]:
        strategy = detector.get_memory_strategy(model_size)
        print(f"   {model_size}GB model: {strategy['strategy']}, quantization: {strategy['use_quantization']}")


if __name__ == "__main__":
    main()