#!/usr/bin/env python3
"""
Single Image Debug Test for InternVL3

This script processes a single image with maximum debugging to identify
why InternVL3 field extraction is failing. Run this on the GPU machine
after the main diagnostic tests pass.
"""

import sys
import time
import traceback
from pathlib import Path

from rich import print as rprint
from rich.console import Console

console = Console()

def load_model_components():
    """Load InternVL3 model with diagnostic output."""
    try:
        rprint("[bold blue]🚀 Loading InternVL3 model components...[/bold blue]")

        from common.internvl3_model_loader import load_internvl3_model

        # Use same config as notebook
        MODEL_PATH = "/home/jovyan/nfs_share/models/InternVL3-8B"

        model, tokenizer = load_internvl3_model(
            model_path=MODEL_PATH,
            use_quantization=False,
            device_map='auto',
            max_new_tokens=4000,
            torch_dtype='bfloat16',
            low_cpu_mem_usage=True,
            verbose=True  # Enable verbose loading
        )

        rprint("[green]✅ Model components loaded successfully[/green]")
        return model, tokenizer

    except Exception as e:
        rprint(f"[red]❌ Model loading failed: {e}[/red]")
        rprint(f"[red]Traceback: {traceback.format_exc()}[/red]")
        return None, None

def test_single_image_processing(model, tokenizer, image_path):
    """Test processing a single image with maximum debugging."""
    try:
        rprint(f"[bold blue]🔍 Testing single image: {Path(image_path).name}[/bold blue]")
        rprint("=" * 80)

        # Create handler with full config
        from internvl3_document_aware_handler import DocumentAwareInternVL3Handler

        config = {
            'MAX_NEW_TOKENS': 4000,
            'PROMPT_CONFIG': {
                'detection_file': 'prompts/document_type_detection.yaml',
                'detection_key': 'detection',
                'extraction_files': {
                    'INVOICE': 'prompts/internvl3_prompts.yaml',
                    'RECEIPT': 'prompts/internvl3_prompts.yaml',
                    'BANK_STATEMENT': 'prompts/internvl3_prompts.yaml'
                },
                'extraction_keys': {
                    'INVOICE': 'invoice',
                    'RECEIPT': 'receipt',
                    'BANK_STATEMENT': 'bank_statement'
                }
            }
        }

        rprint("[cyan]Creating DocumentAwareInternVL3Handler...[/cyan]")
        handler = DocumentAwareInternVL3Handler(
            model=model,
            tokenizer=tokenizer,
            config=config
        )

        rprint("[bold green]✅ Handler created successfully[/bold green]")
        rprint("=" * 80)

        # Step 1: Document detection with MAXIMUM verbose
        rprint("[bold cyan]STEP 1: Document Type Detection[/bold cyan]")
        rprint("-" * 80)

        start_time = time.time()
        classification_info = handler.detect_and_classify_document(
            image_path,
            verbose=True  # MAXIMUM VERBOSE
        )
        detection_time = time.time() - start_time

        rprint(f"[green]Detection completed in {detection_time:.2f}s[/green]")
        rprint(f"[yellow]Classification result: {classification_info}[/yellow]")
        rprint("=" * 80)

        # Step 2: Document extraction with MAXIMUM verbose
        rprint("[bold cyan]STEP 2: Document-Aware Extraction[/bold cyan]")
        rprint("-" * 80)

        start_time = time.time()
        extraction_result = handler.process_document_aware(
            image_path,
            classification_info,
            verbose=True  # MAXIMUM VERBOSE
        )
        extraction_time = time.time() - start_time

        rprint(f"[green]Extraction completed in {extraction_time:.2f}s[/green]")
        rprint("=" * 80)

        # Step 3: Analyze results
        rprint("[bold cyan]STEP 3: Result Analysis[/bold cyan]")
        rprint("-" * 80)

        extracted_data = extraction_result.get('extracted_data', {})
        found_fields = [k for k, v in extracted_data.items() if v != "NOT_FOUND"]

        rprint(f"[yellow]Extraction result keys: {list(extraction_result.keys())}[/yellow]")
        rprint(f"[yellow]Total fields in extracted_data: {len(extracted_data)}[/yellow]")
        rprint(f"[yellow]Non-NOT_FOUND fields: {len(found_fields)}[/yellow]")
        rprint(f"[yellow]Found fields: {found_fields[:10]}{'...' if len(found_fields) > 10 else ''}[/yellow]")

        if found_fields:
            rprint("[bold green]✅ FIELDS EXTRACTED SUCCESSFULLY![/bold green]")
            rprint("[green]Sample extracted values:[/green]")
            for field in found_fields[:5]:
                value = extracted_data[field]
                rprint(f"[green]  {field}: {value[:100]}{'...' if len(str(value)) > 100 else ''}[/green]")
        else:
            rprint("[bold red]❌ NO FIELDS EXTRACTED - THIS IS THE PROBLEM![/bold red]")
            rprint("[red]All fields returned 'NOT_FOUND'[/red]")

            # Show first few fields for debugging
            rprint("[yellow]Sample extracted_data entries:[/yellow]")
            for _i, (field, value) in enumerate(list(extracted_data.items())[:5]):
                rprint(f"[yellow]  {field}: {value}[/yellow]")

        rprint("=" * 80)
        return extraction_result

    except Exception as e:
        rprint(f"[red]❌ Single image test failed: {e}[/red]")
        rprint(f"[red]Traceback: {traceback.format_exc()}[/red]")
        return None

def main():
    """Run single image diagnostic test."""
    rprint("[bold green]🚀 InternVL3 Single Image Debug Test[/bold green]")
    rprint("This script will process one image with MAXIMUM debugging output")
    rprint("to identify why field extraction is failing.")
    rprint("=" * 80)

    # Define test image path (use first available image)
    base_data_path = "/home/jovyan/nfs_share/tod/evaluation_data"
    data_dir = Path(base_data_path)

    # Find first available image
    image_files = []
    for pattern in ["*.png", "*.jpg", "*.jpeg"]:
        image_files.extend(data_dir.glob(pattern))

    if not image_files:
        rprint(f"[red]❌ No images found in {data_dir}[/red]")
        return False

    test_image = str(image_files[0])  # Use first image
    rprint(f"[cyan]Test image: {test_image}[/cyan]")

    if not Path(test_image).exists():
        rprint(f"[red]❌ Test image not found: {test_image}[/red]")
        return False

    # Load model components
    model, tokenizer = load_model_components()
    if model is None or tokenizer is None:
        rprint("[red]❌ Failed to load model components[/red]")
        return False

    # Test single image processing
    result = test_single_image_processing(model, tokenizer, test_image)

    if result is None:
        rprint("[bold red]❌ Single image test failed![/bold red]")
        return False

    # Final summary
    rprint("[bold green]🎉 Single image test completed![/bold green]")
    rprint("[yellow]Check the diagnostic output above to identify the exact issue.[/yellow]")

    extracted_data = result.get('extracted_data', {})
    found_fields = [k for k, v in extracted_data.items() if v != "NOT_FOUND"]

    if found_fields:
        rprint(f"[bold green]✅ SUCCESS: {len(found_fields)} fields extracted[/bold green]")
    else:
        rprint("[bold red]❌ CONFIRMED: Field extraction is broken[/bold red]")
        rprint("[yellow]Look for diagnostic messages above to identify the root cause.[/yellow]")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)