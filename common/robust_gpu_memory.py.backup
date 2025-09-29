"""
Robust GPU Memory Detection System

A comprehensive, bulletproof memory detection system designed for production
environments where interactive debugging is not available. Specifically
optimized for V100 multi-GPU setups and other challenging scenarios.

Key Features:
- Total available VRAM calculation across all GPUs
- Extensive error handling and fallback strategies
- Mixed GPU type support
- Comprehensive diagnostics and logging
- V100-specific optimizations
- Self-validation and consistency checks
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch


@dataclass
class GPUMemoryInfo:
    """Comprehensive GPU memory information."""
    device_id: int
    name: str
    total_memory_gb: float
    allocated_memory_gb: float
    reserved_memory_gb: float
    available_memory_gb: float
    fragmentation_gb: float
    is_available: bool
    error_message: Optional[str] = None


@dataclass
class MemoryDetectionResult:
    """Complete memory detection results."""
    total_gpus: int
    working_gpus: int
    total_memory_gb: float
    total_available_gb: float
    total_allocated_gb: float
    total_reserved_gb: float
    total_fragmentation_gb: float
    per_gpu_info: List[GPUMemoryInfo]
    detection_success: bool
    warnings: List[str]
    is_multi_gpu: bool
    memory_distribution_balanced: bool


class RobustGPUMemoryDetector:
    """
    Bulletproof GPU memory detection system.

    Designed to work reliably without interactive debugging support,
    with extensive error handling and fallback strategies.
    """

    def __init__(self, verbose: bool = True):
        """
        Initialize the robust memory detector.

        Args:
            verbose: Enable detailed logging and diagnostics
        """
        self.verbose = verbose
        self.warnings = []
        self._memory_conversion_factor = 1024**3  # Use consistent 1024^3 for GB conversion

    def detect_gpu_memory(self) -> MemoryDetectionResult:
        """
        Perform comprehensive GPU memory detection with extensive error handling.

        Returns:
            MemoryDetectionResult: Complete memory detection results
        """
        if self.verbose:
            print("🔍 Starting robust GPU memory detection...")

        # Check if CUDA is available
        if not torch.cuda.is_available():
            return self._create_cpu_fallback_result()

        try:
            device_count = torch.cuda.device_count()
            if device_count == 0:
                return self._create_cpu_fallback_result()

            if self.verbose:
                print(f"📊 Detected {device_count} GPU(s), analyzing each device...")

            per_gpu_info = []
            total_memory = 0.0
            total_available = 0.0
            total_allocated = 0.0
            total_reserved = 0.0
            total_fragmentation = 0.0
            working_gpus = 0

            # Analyze each GPU individually with error isolation
            for device_id in range(device_count):
                gpu_info = self._analyze_single_gpu(device_id)
                per_gpu_info.append(gpu_info)

                if gpu_info.is_available:
                    working_gpus += 1
                    total_memory += gpu_info.total_memory_gb
                    total_available += gpu_info.available_memory_gb
                    total_allocated += gpu_info.allocated_memory_gb
                    total_reserved += gpu_info.reserved_memory_gb
                    total_fragmentation += gpu_info.fragmentation_gb

            # Validate results and create comprehensive report
            result = MemoryDetectionResult(
                total_gpus=device_count,
                working_gpus=working_gpus,
                total_memory_gb=total_memory,
                total_available_gb=total_available,
                total_allocated_gb=total_allocated,
                total_reserved_gb=total_reserved,
                total_fragmentation_gb=total_fragmentation,
                per_gpu_info=per_gpu_info,
                detection_success=working_gpus > 0,
                warnings=self.warnings.copy(),
                is_multi_gpu=working_gpus > 1,
                memory_distribution_balanced=self._check_memory_balance(per_gpu_info)
            )

            # Perform validation and add warnings if needed
            self._validate_detection_results(result)

            if self.verbose:
                self._print_comprehensive_report(result)

            return result

        except Exception as e:
            self._add_warning(f"Critical error in GPU detection: {e}")
            if self.verbose:
                print(f"❌ GPU detection failed: {e}")
            return self._create_error_fallback_result(str(e))

    def _analyze_single_gpu(self, device_id: int) -> GPUMemoryInfo:
        """
        Analyze a single GPU with comprehensive error handling.

        Args:
            device_id: GPU device ID to analyze

        Returns:
            GPUMemoryInfo: Complete information about the GPU
        """
        try:
            # Get basic device properties
            props = torch.cuda.get_device_properties(device_id)
            gpu_name = props.name
            total_memory_bytes = props.total_memory
            total_memory_gb = total_memory_bytes / self._memory_conversion_factor

            # Get current memory usage with error handling
            try:
                allocated_bytes = torch.cuda.memory_allocated(device_id)
                reserved_bytes = torch.cuda.memory_reserved(device_id)
                allocated_gb = allocated_bytes / self._memory_conversion_factor
                reserved_gb = reserved_bytes / self._memory_conversion_factor
                available_gb = total_memory_gb - reserved_gb
                fragmentation_gb = reserved_gb - allocated_gb

                # Validate memory values
                if available_gb < 0:
                    self._add_warning(f"GPU {device_id}: Negative available memory detected")
                    available_gb = 0.0

                if fragmentation_gb < 0:
                    fragmentation_gb = 0.0

            except Exception as e:
                self._add_warning(f"GPU {device_id}: Failed to get memory usage: {e}")
                allocated_gb = 0.0
                reserved_gb = 0.0
                available_gb = total_memory_gb  # Assume all memory is available
                fragmentation_gb = 0.0

            if self.verbose:
                print(f"   GPU {device_id} ({gpu_name}): {total_memory_gb:.1f}GB total, {available_gb:.1f}GB available")

            return GPUMemoryInfo(
                device_id=device_id,
                name=gpu_name,
                total_memory_gb=total_memory_gb,
                allocated_memory_gb=allocated_gb,
                reserved_memory_gb=reserved_gb,
                available_memory_gb=available_gb,
                fragmentation_gb=fragmentation_gb,
                is_available=True
            )

        except Exception as e:
            error_msg = f"Failed to analyze GPU {device_id}: {e}"
            self._add_warning(error_msg)
            if self.verbose:
                print(f"   ❌ GPU {device_id}: {error_msg}")

            return GPUMemoryInfo(
                device_id=device_id,
                name="Unknown",
                total_memory_gb=0.0,
                allocated_memory_gb=0.0,
                reserved_memory_gb=0.0,
                available_memory_gb=0.0,
                fragmentation_gb=0.0,
                is_available=False,
                error_message=error_msg
            )

    def _check_memory_balance(self, gpu_infos: List[GPUMemoryInfo]) -> bool:
        """
        Check if memory is reasonably balanced across GPUs.

        Args:
            gpu_infos: List of GPU memory information

        Returns:
            bool: True if memory distribution is balanced
        """
        working_gpus = [gpu for gpu in gpu_infos if gpu.is_available]
        if len(working_gpus) <= 1:
            return True

        # Check if total memory varies by more than 20%
        memory_values = [gpu.total_memory_gb for gpu in working_gpus]
        min_memory = min(memory_values)
        max_memory = max(memory_values)

        if min_memory == 0:
            return False

        variation = (max_memory - min_memory) / min_memory
        return variation <= 0.20  # 20% tolerance

    def _validate_detection_results(self, result: MemoryDetectionResult) -> None:
        """
        Validate detection results and add warnings for suspicious values.

        Args:
            result: Detection results to validate
        """
        # Check for reasonable total memory values
        if result.detection_success:
            if result.total_memory_gb < 1.0:
                self._add_warning("Detected total GPU memory is suspiciously low")
            elif result.total_memory_gb > 1000.0:
                self._add_warning("Detected total GPU memory is suspiciously high")

            # Check for excessive fragmentation
            if result.total_fragmentation_gb > result.total_memory_gb * 0.3:
                self._add_warning("High memory fragmentation detected (>30%)")

            # Check for mixed GPU types in multi-GPU setup
            if result.is_multi_gpu and not result.memory_distribution_balanced:
                self._add_warning("Mixed GPU types detected - memory distribution unbalanced")

        # V100-specific checks
        self._validate_v100_configuration(result)

    def _validate_v100_configuration(self, result: MemoryDetectionResult) -> None:
        """
        V100-specific validation and optimization suggestions.

        Args:
            result: Detection results to validate for V100 compatibility
        """
        v100_gpus = [gpu for gpu in result.per_gpu_info
                     if gpu.is_available and "V100" in gpu.name.upper()]

        if v100_gpus:
            if self.verbose:
                print(f"🎯 V100 Configuration detected: {len(v100_gpus)} V100 GPU(s)")

            # Check V100 memory expectations
            for gpu in v100_gpus:
                # V100s should have 16GB or 32GB typically
                if gpu.total_memory_gb < 15.0 or gpu.total_memory_gb > 35.0:
                    self._add_warning(f"V100 GPU {gpu.device_id}: Unexpected memory size {gpu.total_memory_gb:.1f}GB")

            # V100 optimization analysis for any number of V100s
            total_v100_memory = sum(gpu.total_memory_gb for gpu in v100_gpus)
            if self.verbose:
                if len(v100_gpus) == 1:
                    print(f"💡 Single V100 setup: {total_v100_memory:.1f}GB V100 memory")
                else:
                    print(f"💡 Multi-V100 setup: {len(v100_gpus)}x V100 = {total_v100_memory:.1f}GB total V100 memory")

            # Dynamic V100 memory assessment for InternVL3-8B
            # InternVL3-8B needs ~16GB + buffer. V100 assessment:
            # 1x V100 (16GB): Tight but should work with quantization
            # 2x V100 (32GB): Comfortable for full precision
            # 3x V100 (48GB): Excellent for full precision
            # 4x V100 (64GB): Excellent for full precision

            model_memory_needed = 16.0  # InternVL3-8B estimate
            memory_buffer = 4.0  # Safety buffer
            total_needed = model_memory_needed + memory_buffer

            if total_v100_memory >= total_needed:
                if self.verbose:
                    print(f"✅ V100 memory sufficient for full precision InternVL3-8B: {total_v100_memory:.1f}GB >= {total_needed:.1f}GB needed")
            else:
                if self.verbose:
                    print(f"⚠️ V100 memory tight for full precision InternVL3-8B: {total_v100_memory:.1f}GB < {total_needed:.1f}GB needed")
                    print("💡 Consider quantization for 1x V100, or use 2+ V100s for full precision")
                self._add_warning(f"V100 memory may need quantization: {total_v100_memory:.1f}GB available, {total_needed:.1f}GB needed")

    def _create_cpu_fallback_result(self) -> MemoryDetectionResult:
        """Create fallback result for CPU-only systems."""
        if self.verbose:
            print("💻 No CUDA GPUs available, using CPU fallback")

        return MemoryDetectionResult(
            total_gpus=0,
            working_gpus=0,
            total_memory_gb=0.0,
            total_available_gb=0.0,
            total_allocated_gb=0.0,
            total_reserved_gb=0.0,
            total_fragmentation_gb=0.0,
            per_gpu_info=[],
            detection_success=False,
            warnings=["No CUDA GPUs available"],
            is_multi_gpu=False,
            memory_distribution_balanced=True
        )

    def _create_error_fallback_result(self, error_message: str) -> MemoryDetectionResult:
        """Create fallback result for critical errors."""
        return MemoryDetectionResult(
            total_gpus=0,
            working_gpus=0,
            total_memory_gb=0.0,
            total_available_gb=0.0,
            total_allocated_gb=0.0,
            total_reserved_gb=0.0,
            total_fragmentation_gb=0.0,
            per_gpu_info=[],
            detection_success=False,
            warnings=[f"Critical detection error: {error_message}"],
            is_multi_gpu=False,
            memory_distribution_balanced=False
        )

    def _add_warning(self, message: str) -> None:
        """Add a warning message to the detection results."""
        self.warnings.append(message)

    def _print_comprehensive_report(self, result: MemoryDetectionResult) -> None:
        """
        Print a comprehensive memory detection report.

        Args:
            result: Detection results to report
        """
        print("\n" + "="*70)
        print("🔍 ROBUST GPU MEMORY DETECTION REPORT")
        print("="*70)

        if result.detection_success:
            print(f"✅ Success: {result.working_gpus}/{result.total_gpus} GPUs detected")
            print(f"📊 Total Memory: {result.total_memory_gb:.2f}GB")
            print(f"💾 Available Memory: {result.total_available_gb:.2f}GB")
            print(f"⚡ Allocated Memory: {result.total_allocated_gb:.2f}GB")
            print(f"🔄 Reserved Memory: {result.total_reserved_gb:.2f}GB")
            print(f"📦 Fragmentation: {result.total_fragmentation_gb:.2f}GB")
            print(f"🖥️  Multi-GPU: {'Yes' if result.is_multi_gpu else 'No'}")
            print(f"⚖️  Balanced Distribution: {'Yes' if result.memory_distribution_balanced else 'No'}")

            print("\n📋 Per-GPU Breakdown:")
            for gpu in result.per_gpu_info:
                if gpu.is_available:
                    utilization = (gpu.allocated_memory_gb / gpu.total_memory_gb * 100) if gpu.total_memory_gb > 0 else 0
                    print(f"   GPU {gpu.device_id} ({gpu.name}): {gpu.total_memory_gb:.1f}GB total, "
                          f"{gpu.available_memory_gb:.1f}GB available ({utilization:.1f}% used)")
                else:
                    print(f"   GPU {gpu.device_id}: ❌ Failed - {gpu.error_message}")

        else:
            print("❌ Detection Failed")

        if result.warnings:
            print(f"\n⚠️  Warnings ({len(result.warnings)}):")
            for warning in result.warnings:
                print(f"   • {warning}")

        print("="*70)

    def get_total_available_memory(self) -> float:
        """
        Get total available VRAM across all GPUs - the main function you need!

        Returns:
            float: Total available VRAM in GB across all working GPUs
        """
        result = self.detect_gpu_memory()
        return result.total_available_gb

    def validate_memory_for_model(self, model_size_gb: float, buffer_gb: float = 4.0) -> Tuple[bool, str]:
        """
        Validate if there's sufficient memory for a specific model.

        Args:
            model_size_gb: Estimated model size in GB
            buffer_gb: Additional buffer memory needed

        Returns:
            Tuple[bool, str]: (is_sufficient, diagnostic_message)
        """
        result = self.detect_gpu_memory()

        if not result.detection_success:
            return False, "GPU detection failed"

        total_needed = model_size_gb + buffer_gb

        if result.total_available_gb >= total_needed:
            return True, f"Sufficient memory: {result.total_available_gb:.1f}GB available, {total_needed:.1f}GB needed"
        else:
            return False, f"Insufficient memory: {result.total_available_gb:.1f}GB available, {total_needed:.1f}GB needed"

    def diagnose_memory_issues(self) -> Dict[str, any]:
        """
        Comprehensive memory diagnostics for troubleshooting.

        Returns:
            Dict with detailed diagnostic information
        """
        result = self.detect_gpu_memory()

        diagnostics = {
            "detection_successful": result.detection_success,
            "total_gpus_detected": result.total_gpus,
            "working_gpus": result.working_gpus,
            "total_memory_gb": result.total_memory_gb,
            "total_available_gb": result.total_available_gb,
            "memory_utilization_percent": (result.total_allocated_gb / result.total_memory_gb * 100) if result.total_memory_gb > 0 else 0,
            "fragmentation_percent": (result.total_fragmentation_gb / result.total_memory_gb * 100) if result.total_memory_gb > 0 else 0,
            "is_multi_gpu": result.is_multi_gpu,
            "memory_balanced": result.memory_distribution_balanced,
            "warnings_count": len(result.warnings),
            "warnings": result.warnings,
            "per_gpu_details": [
                {
                    "device_id": gpu.device_id,
                    "name": gpu.name,
                    "total_gb": gpu.total_memory_gb,
                    "available_gb": gpu.available_memory_gb,
                    "working": gpu.is_available,
                    "error": gpu.error_message
                }
                for gpu in result.per_gpu_info
            ],
            "recommendations": self._generate_recommendations(result)
        }

        return diagnostics

    def _generate_recommendations(self, result: MemoryDetectionResult) -> List[str]:
        """Generate recommendations based on detection results."""
        recommendations = []

        if not result.detection_success:
            recommendations.append("Check CUDA installation and GPU drivers")

        if result.total_fragmentation_gb > 2.0:
            recommendations.append("Consider clearing GPU cache: torch.cuda.empty_cache()")

        if result.is_multi_gpu and not result.memory_distribution_balanced:
            recommendations.append("Mixed GPU setup detected - consider using device_map='auto'")

        # V100-specific recommendations
        v100_gpus = [gpu for gpu in result.per_gpu_info if gpu.is_available and "V100" in gpu.name.upper()]
        if v100_gpus:
            v100_count = len(v100_gpus)
            v100_total_memory = sum(gpu.total_memory_gb for gpu in v100_gpus)

            if v100_count == 1:
                if v100_total_memory >= 20.0:
                    recommendations.append("Single V100 has sufficient memory for full precision InternVL3-8B")
                else:
                    recommendations.append("Single V100 may need quantization for InternVL3-8B (consider 8-bit)")
            else:
                if v100_total_memory >= 20.0:
                    recommendations.append(f"{v100_count}x V100 setup has excellent memory for full precision models")
                else:
                    recommendations.append(f"{v100_count}x V100 setup may need quantization for large models")

        return recommendations


# Convenience functions for easy integration
def get_total_gpu_memory() -> float:
    """Get total GPU memory across all devices in GB."""
    detector = RobustGPUMemoryDetector(verbose=False)
    result = detector.detect_gpu_memory()
    return result.total_memory_gb


def get_total_available_gpu_memory() -> float:
    """Get total available GPU memory across all devices in GB."""
    detector = RobustGPUMemoryDetector(verbose=False)
    return detector.get_total_available_memory()


def validate_model_memory_requirements(model_size_gb: float, buffer_gb: float = 4.0, verbose: bool = True) -> bool:
    """
    Validate if there's sufficient GPU memory for a model.

    Args:
        model_size_gb: Estimated model size in GB
        buffer_gb: Additional buffer memory needed
        verbose: Enable detailed output

    Returns:
        bool: True if sufficient memory is available
    """
    detector = RobustGPUMemoryDetector(verbose=verbose)
    is_sufficient, message = detector.validate_memory_for_model(model_size_gb, buffer_gb)

    if verbose:
        print(f"💾 Memory Validation: {message}")

    return is_sufficient


def diagnose_gpu_memory(verbose: bool = True) -> Dict[str, any]:
    """
    Comprehensive GPU memory diagnostics.

    Args:
        verbose: Enable detailed output

    Returns:
        Dict with diagnostic information
    """
    detector = RobustGPUMemoryDetector(verbose=verbose)
    return detector.diagnose_memory_issues()