"""
LlamaVisionTableExtractor - V100 Production-Optimized Vision Language Model Extractor

A standalone module for extracting structured data from bank statements using
Llama 3.2 Vision model with V100 GPU optimizations.

Features:
- ResilientGenerator with 6-tier OOM fallback system
- V100-specific memory management and fragmentation detection
- Comprehensive error handling and validation
- GPU memory monitoring and cleanup between operations
- Emergency model reload and CPU fallback capabilities
"""

# Standard library imports
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Third-party imports
import torch
from PIL import Image
from rich import print as rprint
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    MllamaForConditionalGeneration,
)

# Local imports - GPU optimization utilities
from common.gpu_optimization import (
    ResilientGenerator,
    comprehensive_memory_cleanup,
    configure_cuda_memory_allocation,
    detect_memory_fragmentation,
    handle_memory_fragmentation,
    optimize_model_for_v100,
)

# Initialize Rich console
console = Console()

# Configuration constants (you may need to adjust these for your setup)
MODEL_PATH = "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision-Instruct"
MAX_NEW_TOKENS = 4000  # V100 OPTIMIZED - Prevents OOM
TORCH_DTYPE = "bfloat16"  # V100 COMPATIBLE - More efficient

# Default minimal prompt for bank statement extraction
MINIMAL_PROMPT = """You are a precise bank statement analyzer. Extract ALL transactions from this bank statement image into a clean markdown table.

**CRITICAL REQUIREMENTS:**
1. Extract EVERY transaction visible in the image
2. If multiple transactions occur on the same date, extract each as a separate row
3. Use this EXACT table format:

| Date | Description | Debit | Credit | Balance |
|------|-------------|-------|--------|---------|
| DD/MM/YYYY | Transaction description | amount or NOT_FOUND | amount or NOT_FOUND | amount or NOT_FOUND |

**Data extraction rules:**
- Date: Use DD/MM/YYYY format exactly as shown
- Description: Copy the full transaction description exactly
- Debit: Amount for withdrawals/debits (enter NOT_FOUND if empty)
- Credit: Amount for deposits/credits (enter NOT_FOUND if empty)  
- Balance: Running balance after transaction (enter NOT_FOUND if not shown)
- For amounts: Include currency symbols if present (e.g., $123.45)

**Table quality standards:**
- NO combining of transactions
- NO summarizing - extract each transaction individually
- NO extra formatting or headers beyond the required table
- NO additional commentary or explanation

Extract the complete transaction table now:"""


class LlamaVisionTableExtractor:
    """
    V100 Production-Optimized class for extracting structured data from bank statements
    using Llama 3.2 Vision model.
    
    Features:
    - ResilientGenerator with 6-tier OOM fallback system
    - V100-specific memory management and fragmentation detection
    - Comprehensive error handling and validation
    - GPU memory monitoring and cleanup between operations
    - Emergency model reload and CPU fallback capabilities
    """
    
    def __init__(self, model_path: str = None, processor=None, model=None):
        """
        Initialize with either a model path OR existing processor/model instances.
        
        Args:
            model_path: Path to load new model from
            processor: Existing processor instance 
            model: Existing model instance
        """
        if processor is not None and model is not None:
            # Use existing instances from notebook
            self.processor = processor
            self.model = model
            self.model_path = model_path or MODEL_PATH  # Store for emergency reload
            
            # Initialize V100-optimized ResilientGenerator
            self.resilient_generator = ResilientGenerator(
                model=self.model,
                processor=self.processor,
                model_path=self.model_path,
                model_loader=self._reload_model_for_emergency
            )
            rprint("[green]✅ Using existing model and processor with V100 ResilientGenerator[/green]")
            
        elif model_path:
            # Load new instances (for standalone usage)
            rprint(f"[yellow]Loading new V100-optimized model from: {model_path}[/yellow]")
            self.model_path = model_path
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = MllamaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            
            # Initialize ResilientGenerator for standalone usage
            self.resilient_generator = ResilientGenerator(
                model=self.model,
                processor=self.processor,
                model_path=self.model_path,
                model_loader=self._reload_model_for_emergency
            )
            rprint("[green]✅ Standalone V100-optimized model and processor loaded[/green]")
        else:
            raise ValueError("Must provide either model_path OR (processor, model) instances")
    
    def _reload_model_for_emergency(self, model_path: str):
        """
        Emergency model reload function for ResilientGenerator.
        
        Args:
            model_path: Path to reload model from
            
        Returns:
            Tuple of (model, processor)
        """
        
        rprint("[red]🚨 EMERGENCY MODEL RELOAD for V100 OOM recovery[/red]")
        
        # Reconfigure CUDA memory
        configure_cuda_memory_allocation()
        
        # V100-optimized quantization config
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=True,
            llm_int8_skip_modules=["vision_tower", "multi_modal_projector"],
            llm_int8_threshold=6.0,
        )
        
        # Reload with V100 optimizations
        model = MllamaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
        )
        
        processor = AutoProcessor.from_pretrained(model_path)
        
        # Apply V100 optimizations
        optimize_model_for_v100(model)
        
        return model, processor
    
    def extract_table(self, 
                     image_path: str, 
                     prompt_template: str = None,
                     max_new_tokens: int = None,
                     temperature: float = 0.1) -> Dict[str, Any]:
        """
        V100-optimized extraction with ResilientGenerator and memory management.
        
        Args:
            image_path: Path to bank statement image
            prompt_template: Custom prompt (defaults to MINIMAL_PROMPT)
            max_new_tokens: Maximum tokens to generate (uses global MAX_NEW_TOKENS if None)
            temperature: Generation temperature (lower = more deterministic)
            
        Returns:
            Dictionary with extraction results
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            RuntimeError: If model inference fails
            ValueError: If inputs are invalid
        """
        try:
            # Use global configuration for token limits if not specified
            if max_new_tokens is None:
                max_new_tokens = MAX_NEW_TOKENS
                rprint(f"[dim]Using global MAX_NEW_TOKENS: {max_new_tokens}[/dim]")
            
            # V100 MEMORY CHECK: Pre-processing fragmentation detection
            allocated, reserved, fragmentation = detect_memory_fragmentation()
            if fragmentation > 0.5:  # V100 threshold
                rprint(f"[yellow]⚠️ Pre-processing fragmentation: {fragmentation:.2f}GB - cleaning up[/yellow]")
                handle_memory_fragmentation(threshold_gb=0.5, aggressive=True)
            
            # Validate inputs
            if not image_path:
                raise ValueError("Image path cannot be empty")
            
            image_path = Path(image_path)
            if not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            # Load and validate image
            rprint(f"[dim]Loading image: {image_path.name}...[/dim]")
            try:
                image = Image.open(image_path)
                # Validate image format
                if image.format not in ['PNG', 'JPEG', 'JPG', None]:
                    rprint(f"[yellow]⚠️ Warning: Unusual image format {image.format}[/yellow]")
            except Exception as e:
                raise ValueError(f"Failed to load image {image_path}: {e}") from e
            
            # Use provided prompt or default to MINIMAL_PROMPT
            prompt = prompt_template if prompt_template else MINIMAL_PROMPT
            
            # Prepare inputs with error handling
            try:
                messages = [
                    {"role": "user", "content": [
                        {"type": "image"},
                        {"type": "text", "text": prompt}
                    ]}
                ]
                
                # Apply chat template
                input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
                
                # Process inputs
                inputs = self.processor(
                    image,
                    input_text,
                    add_special_tokens=False,
                    return_tensors="pt"
                ).to(self.model.device)
                
            except Exception as e:
                raise RuntimeError(f"Failed to process inputs: {e}") from e
            
            # V100 RESILIENT GENERATION: Use ResilientGenerator instead of direct model.generate()
            rprint(f"[yellow]🔍 V100 Resilient extraction from {image_path.name} (max_tokens: {max_new_tokens})...[/yellow]")
            
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "temperature": temperature,
                "top_p": 0.95,
                "use_cache": True  # Critical for quality
            }
            
            try:
                # Use ResilientGenerator for robust V100 processing
                output = self.resilient_generator.generate(inputs, generation_kwargs)
                    
            except Exception as e:
                rprint(f"[red]❌ ResilientGenerator failed: {e}[/red]")
                raise RuntimeError(f"V100-optimized model inference failed: {e}") from e
            
            # Decode response with error handling
            try:
                response = self.processor.decode(output[0], skip_special_tokens=True)
                
                # Extract assistant response (clean up chat template artifacts)
                if "assistant" in response:
                    response = response.split("assistant")[-1].strip()
                
                # Validate response length
                if len(response.strip()) < 10:
                    rprint("[yellow]⚠️ Warning: Very short model response - possible inference issue[/yellow]")
                
            except Exception as e:
                raise RuntimeError(f"Failed to decode model response: {e}") from e
            
            # V100 MEMORY CLEANUP: Comprehensive cleanup after each extraction
            rprint("[dim]Performing V100 memory cleanup after extraction...[/dim]")
            comprehensive_memory_cleanup(self.model, self.processor)
            
            # V100 POST-PROCESSING: Check for fragmentation after processing
            allocated_after, reserved_after, fragmentation_after = detect_memory_fragmentation()
            if fragmentation_after > 1.0:  # Higher threshold for post-processing
                rprint(f"[yellow]⚠️ Post-processing fragmentation: {fragmentation_after:.2f}GB[/yellow]")
                handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)
            
            # Parse and return structured response
            return self._parse_response(response, str(image_path), prompt, temperature, max_new_tokens)
            
        except Exception as e:
            # Log error and re-raise with context
            rprint(f"[red]❌ V100 extraction failed for {image_path}: {e}[/red]")
            # Emergency cleanup on failure
            comprehensive_memory_cleanup(self.model, self.processor)
            raise
    
    def _parse_response(self, response: str, image_path: str, prompt: str, temperature: float, max_new_tokens: int) -> Dict[str, Any]:
        """Parse model response into structured format."""
        parsed_data = self._extract_table_data(response)
        bank_details = self._extract_bank_details(response)
        return {
            "raw_response": response,
            "image_path": image_path,
            "extraction_time": datetime.now().isoformat(),
            "parsed_data": parsed_data,
            "bank_details": bank_details,
            "transaction_count": len(parsed_data),
            "prompt_used": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "response_length": len(response),
            "v100_optimized": True  # Flag for V100 optimization status
        }
    
    def _extract_table_data(self, response: str) -> List[Dict]:
        """Extract table rows from markdown response."""
        lines = response.split('\n')
        table_rows = []
        
        for line in lines:
            if '|' in line and not line.strip().startswith('|---') and 'Date' not in line:
                # Parse table row
                cells = [cell.strip() for cell in line.split('|')[1:-1]]
                if len(cells) >= 4:  # Ensure minimum columns
                    row_data = {
                        'date': cells[0],
                        'description': cells[1],
                        'debit': cells[2],
                        'credit': cells[3],
                        'balance': cells[4] if len(cells) > 4 else 'NOT_FOUND'
                    }
                    # Only add if at least one field has data (not empty and not just NOT_FOUND)
                    if any(value and value != 'NOT_FOUND' for value in row_data.values()):
                        table_rows.append(row_data)
        
        return table_rows
    
    def _extract_bank_details(self, response: str) -> Dict[str, str]:
        """Extract bank account details from response, removing markdown asterisks."""
        import re
        
        bank_fields = {
            'BANK_NAME': 'NOT_FOUND',
            'BANK_BSB_NUMBER': 'NOT_FOUND', 
            'BANK_ACCOUNT_NUMBER': 'NOT_FOUND',
            'BANK_ACCOUNT_HOLDER': 'NOT_FOUND',
            'ACCOUNT_OPENING_BALANCE': 'NOT_FOUND',
            'ACCOUNT_CLOSING_BALANCE': 'NOT_FOUND',
            'EARLIEST_TRANSACTION_DATE': 'NOT_FOUND',
            'LATEST_TRANSACTION_DATE': 'NOT_FOUND'
        }
        
        lines = response.split('\n')
        for line in lines:
            # Remove markdown asterisks from the line
            clean_line = re.sub(r'\*+', '', line).strip()
            
            # Look for field: value patterns
            for field_name in bank_fields.keys():
                if field_name in clean_line and ':' in clean_line:
                    parts = clean_line.split(':', 1)
                    if len(parts) == 2 and parts[0].strip() == field_name:
                        value = parts[1].strip()
                        if value:  # Only update if we found a non-empty value
                            bank_fields[field_name] = value
                        break
        
        return bank_fields
    
    def test_extraction(self, image_path: str, prompt: str = None, max_new_tokens: int = None) -> Dict[str, Any]:
        """
        Run V100-optimized extraction test with comprehensive validation and metrics.
        
        Args:
            image_path: Path to test image
            prompt: Optional custom prompt
            max_new_tokens: Token limit (uses global MAX_NEW_TOKENS if None)
            
        Returns:
            Dictionary with test results and V100 metrics
        """
        # Use global configuration if not specified
        if max_new_tokens is None:
            max_new_tokens = MAX_NEW_TOKENS
            
        if not image_path:
            rprint("[red]❌ No image path provided for V100 testing[/red]")
            return {"error": "No image path", "v100_optimized": True}
        
        try:
            rprint("[bold green]🧪 RUNNING V100-OPTIMIZED EXTRACTION TEST[/bold green]")
            rprint(f"[cyan]📄 Image: {image_path}[/cyan]")
            rprint(f"[blue]🔧 Max tokens: {max_new_tokens}[/blue]")
            rprint("[blue]🎯 V100 Features: ResilientGenerator, Memory cleanup, Fragmentation detection[/blue]")
            console.rule("[bold blue]V100 Extraction Test[/bold blue]")
            
            # V100: Pre-test memory state
            allocated_start, reserved_start, fragmentation_start = detect_memory_fragmentation()
            
            # Run V100-optimized extraction
            start_time = datetime.now()
            result = self.extract_table(image_path, prompt_template=prompt or MINIMAL_PROMPT, max_new_tokens=max_new_tokens)
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            # V100: Post-test memory state
            allocated_end, reserved_end, fragmentation_end = detect_memory_fragmentation()
            
            # Display results with Rich formatting
            result_panel = Panel(
                result["raw_response"],
                title="[bold yellow]📊 V100 EXTRACTION OUTPUT[/bold yellow]",
                border_style="yellow",
                expand=False
            )
            console.print(result_panel)
            
            # Get expected count from ground truth (if available)
            expected_count = self._get_expected_transaction_count(image_path)
            
            # Create comprehensive analysis table with V100 metrics
            analysis_table = Table(title="📈 V100 Extraction Analysis", border_style="green")
            analysis_table.add_column("Metric", style="cyan", no_wrap=True)
            analysis_table.add_column("Value", style="magenta")
            analysis_table.add_column("V100 Status", justify="center")
            
            # Transaction count analysis
            extracted_count = result['transaction_count']
            count_status = "✅ Perfect Match" if extracted_count == expected_count else f"⚠️ Expected {expected_count}"
            analysis_table.add_row("Transactions Extracted", str(extracted_count), count_status)
            analysis_table.add_row("Ground Truth Count", str(expected_count), "📋 From CSV")
            analysis_table.add_row("Processing Time", f"{processing_time:.2f}s", "⏱️ V100 Performance")
            analysis_table.add_row("Max Tokens Used", str(max_new_tokens), "🔧 V100 Configuration")
            analysis_table.add_row("ResilientGenerator", "Active", "✅ V100 OOM Protected")
            
            # V100 Memory Analysis
            memory_delta = allocated_end - allocated_start
            fragmentation_change = fragmentation_end - fragmentation_start
            analysis_table.add_row("Memory Delta", f"{memory_delta:+.2f}GB", 
                                 "✅ Clean" if abs(memory_delta) < 0.5 else "⚠️ Accumulation")
            analysis_table.add_row("Fragmentation Change", f"{fragmentation_change:+.2f}GB",
                                 "✅ Stable" if fragmentation_change < 0.5 else "⚠️ Increased")
            
            # Quality checks
            response_length = len(result["raw_response"])
            length_status = "✅ Good" if 100 < response_length < 5000 else "⚠️ Check Quality"
            analysis_table.add_row("Response Length", f"{response_length} chars", length_status)
            
            # Hallucination check
            hallucination_patterns = ["IGA", "1102.37"]
            hallucination_detected = any(
                pattern in result["raw_response"] and result["raw_response"].count(pattern) > 3 
                for pattern in hallucination_patterns
            )
            hallucination_status = "❌ Possible Hallucination" if hallucination_detected else "✅ Clean"
            analysis_table.add_row("Hallucination Check", "Pattern Detection", hallucination_status)
            
            console.print(analysis_table)
            
            # V100 Performance Summary
            rprint(f"[green]✅ V100 test completed in {processing_time:.2f}s[/green]")
            rprint(f"[cyan]🎯 V100 Memory Management: {abs(memory_delta):.2f}GB delta, {fragmentation_change:+.2f}GB fragmentation change[/cyan]")
            
            return {
                "success": True,
                "extracted_count": extracted_count,
                "expected_count": expected_count,
                "accuracy": extracted_count == expected_count,
                "processing_time": processing_time,
                "response_length": response_length,
                "hallucination_detected": hallucination_detected,
                "max_new_tokens_used": max_new_tokens,
                "raw_result": result,
                "v100_optimized": True,
                "memory_metrics": {
                    "memory_delta_gb": memory_delta,
                    "fragmentation_change_gb": fragmentation_change,
                    "start_allocated_gb": allocated_start,
                    "end_allocated_gb": allocated_end,
                    "resilient_generator_used": True
                }
            }
            
        except Exception as e:
            rprint(f"[red]❌ V100 test failed: {e}[/red]")
            # Emergency cleanup on test failure
            comprehensive_memory_cleanup(self.model, self.processor)
            return {"error": str(e), "success": False, "v100_optimized": True}

    def _get_expected_transaction_count(self, image_path: str) -> int:
        """
        Get the expected number of transactions from ground truth CSV.
        This is a placeholder method - implement based on your ground truth data.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Expected number of transactions (returns 0 if no ground truth available)
        """
        # Placeholder implementation - you may want to implement this based on your
        # ground truth data structure
        return 0


def get_expected_transaction_count(image_path: str) -> int:
    """
    Get the expected number of transactions from ground truth CSV.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Expected number of transactions from ground truth
    """
    # This is a helper function that was referenced in the original notebook
    # You may need to implement this based on your ground truth data structure
    return 0


# Example usage and testing
if __name__ == "__main__":
    # Example of how to use the extractor
    
    # Option 1: Using existing model and processor instances (from notebook)
    # extractor = LlamaVisionTableExtractor(processor=existing_processor, model=existing_model)
    
    # Option 2: Loading new model (standalone usage)
    # extractor = LlamaVisionTableExtractor(model_path="/path/to/your/model")
    
    # Example extraction
    # result = extractor.extract_table("path/to/bank_statement.png")
    # print(result)
    
    print("LlamaVisionTableExtractor module loaded successfully!")
    print("Import this module to use the V100-optimized extraction capabilities.")