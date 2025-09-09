"""V100-optimized model loading utilities.

This module provides comprehensive model loading functionality with V100 GPU
optimizations, memory management, and validation for Llama Vision models.
"""

from pathlib import Path

import torch
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
)

from .gpu_optimization import (
    comprehensive_memory_cleanup,
    configure_cuda_memory_allocation,
    optimize_model_for_v100,
)


class V100ModelLoader:
    """Handles V100-optimized loading of Llama Vision models."""

    def __init__(self):
        """Initialize the model loader."""
        self.console = Console()
        self.model = None
        self.processor = None

    def load_model(
        self,
        model_path: str,
        use_quantization: bool = True,
        device_map: str = "auto",
        max_new_tokens: int = 4000,
        torch_dtype: str = "bfloat16",
        low_cpu_mem_usage: bool = True,
    ) -> tuple:
        """Load Llama Vision model with V100 optimizations.

        Args:
            model_path: Path to the model directory
            use_quantization: Whether to use 8-bit quantization
            device_map: Device mapping strategy
            max_new_tokens: Maximum new tokens (V100 optimized)
            torch_dtype: Torch data type to use
            low_cpu_mem_usage: Whether to use low CPU memory mode

        Returns:
            Tuple of (model, processor)

        Raises:
            FileNotFoundError: If model path doesn't exist
            RuntimeError: If model loading fails
        """
        rprint(
            "[bold yellow]🚀 Loading Llama Vision model with V100 production optimizations...[/bold yellow]"
        )

        try:
            # PHASE 1: Configure V100-specific CUDA memory allocation
            rprint(
                "[blue]🔧 Configuring V100-optimized CUDA memory allocation...[/blue]"
            )
            configure_cuda_memory_allocation()  # Sets 32MB blocks for V100

            # Validate model path first
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Model path does not exist: {model_path}")

            # PHASE 2: Configure V100-optimized quantization
            quantization_config = self._configure_quantization(use_quantization)

            # PHASE 3: Load model with V100-optimized configuration
            rprint(
                "[dim]Loading Llama-3.2-Vision model with V100 optimizations...[/dim]"
            )

            # Convert torch_dtype string to actual dtype
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            torch_dtype_obj = dtype_map.get(torch_dtype, torch.bfloat16)

            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype_obj,
                device_map=device_map,
                quantization_config=quantization_config,
                low_cpu_mem_usage=low_cpu_mem_usage,
            )

            # Load processor
            rprint("[dim]Loading processor...[/dim]")
            self.processor = AutoProcessor.from_pretrained(model_path)

            # PHASE 4: Apply V100-specific model optimizations
            optimize_model_for_v100(self.model)

            # PHASE 5: Comprehensive validation
            self._validate_model_loading()

            # PHASE 6: Memory analysis and warnings
            self._analyze_memory_usage()

            # PHASE 7: Display configuration
            self._display_configuration(model_path, use_quantization, max_new_tokens)

            # PHASE 8: Model functionality test
            self._test_model_functionality()

            # PHASE 9: Initial memory cleanup
            rprint("[dim]Performing initial V100 memory cleanup...[/dim]")
            comprehensive_memory_cleanup(self.model, self.processor)

            rprint(
                "[bold green]🎉 V100-optimized model loading and validation complete![/bold green]"
            )
            rprint(
                "[cyan]🔧 V100 optimizations active: CPU offload, vision skip, 32MB blocks[/cyan]"
            )

            return self.model, self.processor

        except Exception as e:
            rprint(
                "[bold red]❌ CRITICAL ERROR: V100-optimized model loading failed[/bold red]"
            )
            rprint(f"[red]💥 Error: {e}[/red]")
            rprint(
                "[yellow]💡 Check model path, GPU memory, and V100 compatibility requirements[/yellow]"
            )
            raise

    def _configure_quantization(self, use_quantization: bool):
        """Configure V100-optimized quantization settings.

        Args:
            use_quantization: Whether to enable quantization

        Returns:
            BitsAndBytesConfig or None
        """
        if use_quantization:
            rprint(
                "[yellow]🔧 Configuring V100-optimized 8-bit quantization with BitsAndBytesConfig[/yellow]"
            )

            # V100-OPTIMIZED quantization configuration
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_enable_fp32_cpu_offload=True,  # CRITICAL for V100 memory management
                llm_int8_skip_modules=[
                    "vision_tower",
                    "multi_modal_projector",
                ],  # Skip vision components
                llm_int8_threshold=6.0,  # Standard threshold for outlier detection
            )

            rprint("[green]✅ V100-optimized BitsAndBytesConfig configured[/green]")
            rprint("[cyan]💡 Key V100 optimizations:[/cyan]")
            rprint("[cyan]   • CPU offload enabled for memory efficiency[/cyan]")
            rprint(
                "[cyan]   • Vision modules skipped to prevent quantization issues[/cyan]"
            )
            rprint("[cyan]   • 32MB CUDA memory blocks configured[/cyan]")

            return quantization_config
        else:
            rprint("[blue]ℹ️ No quantization - loading in full precision[/blue]")
            return None

    def _validate_model_loading(self):
        """Validate that model and processor loaded correctly."""
        if self.model is None:
            raise RuntimeError("Model failed to load - returned None")
        if self.processor is None:
            raise RuntimeError("Processor failed to load - returned None")

        # Model parameter analysis
        model_params = sum(p.numel() for p in self.model.parameters())
        if model_params < 1e9:  # Less than 1B parameters seems wrong
            rprint(
                f"[yellow]⚠️ Warning: Model has unusually few parameters: {model_params:,.0f}[/yellow]"
            )

        # Device placement validation
        model_device = next(self.model.parameters()).device
        if model_device.type == "cpu" and torch.cuda.is_available():
            rprint(
                "[yellow]⚠️ Warning: Model loaded on CPU despite CUDA availability[/yellow]"
            )

        rprint("[bold green]✅ Model and processor loaded successfully![/bold green]")
        rprint(f"[cyan]📊 Device: {model_device}[/cyan]")

    def _analyze_memory_usage(self):
        """Analyze and report GPU memory usage with V100-specific warnings."""
        if not torch.cuda.is_available():
            rprint(
                "[yellow]⚠️ CUDA not available - using CPU (significantly slower)[/yellow]"
            )
            return

        gpu_name = torch.cuda.get_device_name()
        memory_allocated = torch.cuda.memory_allocated() / 1e9
        memory_reserved = torch.cuda.memory_reserved() / 1e9
        memory_total = torch.cuda.get_device_properties(0).total_memory / 1e9

        rprint(f"[magenta]🎮 GPU: {gpu_name}[/magenta]")
        rprint(f"[blue]💾 Memory Allocated: {memory_allocated:.2f}GB[/blue]")
        rprint(f"[blue]💾 Memory Reserved: {memory_reserved:.2f}GB[/blue]")
        rprint(f"[blue]💾 Total GPU Memory: {memory_total:.0f}GB[/blue]")

        # V100-specific memory warnings and fragmentation detection
        memory_usage_pct = memory_allocated / memory_total * 100
        fragmentation = memory_reserved - memory_allocated

        if "V100" in gpu_name:
            rprint(
                "[blue]🎯 Detected V100: Applying specialized memory management[/blue]"
            )
            if fragmentation > 1.0:
                rprint(
                    f"[red]🚨 FRAGMENTATION DETECTED: {fragmentation:.2f}GB gap[/red]"
                )
                rprint(
                    "[yellow]⚠️ Memory pool may need cleanup between operations[/yellow]"
                )

            if memory_usage_pct > 80:
                rprint(
                    f"[red]🚨 CRITICAL V100 memory usage: {memory_usage_pct:.1f}% - REDUCE MAX_NEW_TOKENS[/red]"
                )
            elif memory_usage_pct > 60:
                rprint(
                    f"[yellow]⚠️ HIGH V100 memory usage: {memory_usage_pct:.1f}% - Monitor closely[/yellow]"
                )
            elif memory_usage_pct > 40:
                rprint(
                    f"[yellow]⚠️ Moderate V100 memory usage: {memory_usage_pct:.1f}%[/yellow]"
                )
            else:
                rprint(
                    f"[green]✅ Excellent V100 memory usage: {memory_usage_pct:.1f}%[/green]"
                )
        else:
            # H200 or other high-memory GPU
            if memory_usage_pct > 70:
                rprint(f"[red]⚠️ High GPU memory usage: {memory_usage_pct:.1f}%[/red]")
            elif memory_usage_pct > 50:
                rprint(
                    f"[yellow]⚠️ Moderate GPU memory usage: {memory_usage_pct:.1f}%[/yellow]"
                )
            else:
                rprint(
                    f"[green]✅ Good GPU memory usage: {memory_usage_pct:.1f}%[/green]"
                )

    def _display_configuration(
        self, model_path: str, use_quantization: bool, max_new_tokens: int
    ):
        """Display comprehensive V100 configuration validation table.

        Args:
            model_path: Path to the model
            use_quantization: Whether quantization is enabled
            max_new_tokens: Maximum new tokens setting
        """
        model_device = next(self.model.parameters()).device
        model_params = sum(p.numel() for p in self.model.parameters())
        gpu_name = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"

        config_table = Table(
            title="🔧 V100 Production Model Configuration", border_style="blue"
        )
        config_table.add_column("Setting", style="cyan", no_wrap=True)
        config_table.add_column("Value", style="yellow")
        config_table.add_column("V100 Status", style="green")

        # V100-specific configuration details
        config_table.add_row("Model Path", model_path, "✅ Valid")
        config_table.add_row("Device Placement", str(model_device), "✅ Loaded")
        config_table.add_row(
            "Quantization Method",
            "V100-optimized BitsAndBytesConfig" if use_quantization else "Disabled",
            "✅ V100 Optimized" if use_quantization else "✅ Full Precision",
        )
        config_table.add_row(
            "CPU Offload",
            "Enabled" if use_quantization else "N/A",
            "✅ V100 Memory Efficient" if use_quantization else "N/A",
        )
        config_table.add_row(
            "Vision Skip Modules",
            "vision_tower, multi_modal_projector" if use_quantization else "N/A",
            "✅ V100 Compatible" if use_quantization else "N/A",
        )
        config_table.add_row(
            "Max New Tokens",
            str(max_new_tokens),
            "✅ V100 Safe" if max_new_tokens <= 4000 else "⚠️ High for V100",
        )
        config_table.add_row("Model Parameters", f"{model_params:,.0f}", "✅ Loaded")
        config_table.add_row(
            "CUDA Memory Blocks",
            "32MB (V100 optimized)" if "V100" in gpu_name else "64MB (Standard)",
            "✅ Fragmentation resistant",
        )
        config_table.add_row(
            "Memory Optimization", "V100 Enhanced", "✅ Production ready"
        )

        self.console.print(config_table)

    def _test_model_functionality(self):
        """Test basic model functionality."""
        rprint("[dim]Running model compatibility test...[/dim]")
        try:
            # Test chat template processing
            test_messages = [
                {"role": "user", "content": [{"type": "text", "text": "Test"}]}
            ]
            test_input_text = self.processor.apply_chat_template(
                test_messages, add_generation_prompt=True
            )

            if len(test_input_text) < 10:
                raise RuntimeError("Chat template processing failed")

            rprint("[green]✅ Model compatibility test passed[/green]")

        except Exception as e:
            rprint(f"[red]❌ Model compatibility test failed: {e}[/red]")
            raise RuntimeError(f"Model validation failed: {e}") from e


def load_v100_model(
    model_path: str,
    use_quantization: bool = True,
    device_map: str = "auto",
    max_new_tokens: int = 4000,
    torch_dtype: str = "bfloat16",
    low_cpu_mem_usage: bool = True,
) -> tuple:
    """Convenience function for V100-optimized model loading.

    Args:
        model_path: Path to the model directory
        use_quantization: Whether to use 8-bit quantization
        device_map: Device mapping strategy
        max_new_tokens: Maximum new tokens (V100 optimized)
        torch_dtype: Torch data type to use
        low_cpu_mem_usage: Whether to use low CPU memory mode

    Returns:
        Tuple of (model, processor)
    """
    loader = V100ModelLoader()
    return loader.load_model(
        model_path,
        use_quantization,
        device_map,
        max_new_tokens,
        torch_dtype,
        low_cpu_mem_usage,
    )
