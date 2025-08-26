#!/usr/bin/env python3
"""
Experimental Prompt Tester - Quick one-off prompt testing for Llama and InternVL3

This utility allows rapid testing of experimental prompts without modifying the 
main schema-driven pipeline. Perfect for prompt engineering and A/B testing.

Usage:
    python experimental_prompt_tester.py --model llama --prompt "Extract the total amount from this invoice"
    python experimental_prompt_tester.py --model internvl3 --prompt-file my_test_prompt.txt --image test_image.png
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Add project root to path - ensure we can import project modules
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from common.extraction_parser import discover_images
    from models.internvl3_processor import InternVL3Processor
    from models.llama_processor import LlamaProcessor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"💡 Current directory: {Path.cwd()}")
    print(f"💡 Project root: {project_root}")
    print("💡 Make sure you're running from the LMM_POC directory")
    print("💡 Required files: common/extraction_parser.py, models/llama_processor.py, models/internvl3_processor.py")
    sys.exit(1)


class ExperimentalPromptTester:
    """Simple utility for testing experimental prompts on vision-language models."""
    
    def __init__(self, model_name: str, debug: bool = True):
        self.model_name = model_name.lower()
        self.debug = debug
        self.processor = None
        
        print(f"🚀 Initializing {model_name} for experimental prompt testing...")
        
        if self.model_name == "llama":
            self.processor = LlamaProcessor()
        elif self.model_name == "internvl3":
            self.processor = InternVL3Processor()
        else:
            raise ValueError(f"Unsupported model: {model_name}. Use 'llama' or 'internvl3'")
        
        print(f"✅ {model_name} initialized successfully")
    
    def test_prompt(self, prompt: str, image_path: str) -> Dict[str, Any]:
        """Test a single experimental prompt on an image."""
        
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print(f"\n🔍 Testing experimental prompt on: {Path(image_path).name}")
        print(f"📝 Prompt length: {len(prompt)} characters")
        
        if self.debug:
            print("\n📋 EXPERIMENTAL PROMPT:")
            print("-" * 60)
            print(prompt)
            print("-" * 60)
        
        # Time the extraction
        start_time = time.time()
        
        try:
            # Use the processor's direct generation method
            if hasattr(self.processor, '_generate_response'):
                response = self.processor._generate_response(image_path, prompt)
            elif hasattr(self.processor, 'process_single_image'):
                # Temporarily override the prompt generation
                original_method = getattr(self.processor.schema_loader, 'generate_dynamic_prompt', None)
                if original_method:
                    # Monkey patch for testing
                    self.processor.schema_loader.generate_dynamic_prompt = lambda *_args, **_kwargs: prompt
                
                result = self.processor.process_single_image(image_path)
                response = result.get('raw_response', str(result))
                
                # Restore original method
                if original_method:
                    self.processor.schema_loader.generate_dynamic_prompt = original_method
            else:
                raise AttributeError(f"No suitable method found in {self.model_name} processor")
            
            processing_time = time.time() - start_time
            
            print(f"\n🎯 MODEL RESPONSE ({processing_time:.2f}s):")
            print("-" * 60)
            print(response)
            print("-" * 60)
            
            return {
                'model': self.model_name,
                'image_path': image_path,
                'prompt': prompt,
                'response': response,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"❌ Error during processing: {e}")
            raise
    
    def batch_test_prompts(self, prompts: list, image_path: str) -> list:
        """Test multiple prompts on the same image for comparison."""
        
        print(f"\n🔬 Batch testing {len(prompts)} prompts on: {Path(image_path).name}")
        results = []
        
        for i, prompt in enumerate(prompts, 1):
            print(f"\n{'='*20} PROMPT {i}/{len(prompts)} {'='*20}")
            try:
                result = self.test_prompt(prompt, image_path)
                results.append(result)
            except Exception as e:
                print(f"❌ Prompt {i} failed: {e}")
                continue
        
        return results
    
    def compare_with_baseline(self, experimental_prompt: str, image_path: str) -> Dict[str, Any]:
        """Compare experimental prompt with current schema-driven baseline."""
        
        print(f"\n⚖️  Comparing experimental prompt vs baseline on: {Path(image_path).name}")
        
        # Get baseline result (current schema approach)
        print("\n📊 Running baseline (schema-driven)...")
        baseline_start = time.time()
        baseline_result = self.processor.process_single_image(image_path)
        baseline_time = time.time() - baseline_start
        
        print(f"✅ Baseline completed in {baseline_time:.2f}s")
        
        # Get experimental result
        print("\n🧪 Running experimental prompt...")
        experimental_result = self.test_prompt(experimental_prompt, image_path)
        
        # Simple comparison
        comparison = {
            'baseline': {
                'processing_time': baseline_time,
                'response': baseline_result.get('raw_response', str(baseline_result))
            },
            'experimental': experimental_result,
            'time_difference': experimental_result['processing_time'] - baseline_time
        }
        
        print("\n📈 COMPARISON RESULTS:")
        print(f"⏱️  Baseline time: {baseline_time:.2f}s")
        print(f"⏱️  Experimental time: {experimental_result['processing_time']:.2f}s")
        print(f"📊 Time difference: {comparison['time_difference']:+.2f}s")
        
        return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Test experimental prompts on vision-language models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with inline prompt
  python experimental_prompt_tester.py --model llama --prompt "What is the total amount?" --image sample.png
  
  # Test prompt from file
  python experimental_prompt_tester.py --model internvl3 --prompt-file my_prompt.txt --image invoice.png
  
  # Compare with baseline
  python experimental_prompt_tester.py --model llama --prompt "Extract key details" --image test.png --compare-baseline
  
  # Batch test multiple prompts
  python experimental_prompt_tester.py --model internvl3 --prompt-file prompts.txt --image test.png --batch-separator "---"
        """
    )
    
    parser.add_argument('--model', choices=['llama', 'internvl3'], required=True,
                       help='Model to test (llama or internvl3)')
    
    parser.add_argument('--prompt', type=str,
                       help='Experimental prompt text (inline)')
    
    parser.add_argument('--prompt-file', type=Path,
                       help='File containing experimental prompt(s)')
    
    parser.add_argument('--image', type=str, 
                       help='Path to test image (default: use first image from evaluation_data/)')
    
    parser.add_argument('--compare-baseline', action='store_true',
                       help='Compare experimental prompt with current baseline')
    
    parser.add_argument('--batch-separator', type=str, default='---',
                       help='Separator for multiple prompts in prompt file (default: ---)')
    
    parser.add_argument('--no-debug', action='store_true',
                       help='Disable debug output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.prompt and not args.prompt_file:
        parser.error("Must provide either --prompt or --prompt-file")
    
    # Get prompt(s)
    prompts = []
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        if not args.prompt_file.exists():
            parser.error(f"Prompt file not found: {args.prompt_file}")
        
        content = args.prompt_file.read_text().strip()
        if args.batch_separator in content:
            prompts = [p.strip() for p in content.split(args.batch_separator) if p.strip()]
        else:
            prompts = [content]
    
    # Get image path
    image_path = args.image
    if not image_path:
        # Use first image from evaluation data
        try:
            images = discover_images("evaluation_data/")
            if images:
                image_path = str(images[0])
                print(f"🎯 Using default test image: {Path(image_path).name}")
            else:
                parser.error("No images found in evaluation_data/ and no --image specified")
        except Exception as e:
            parser.error(f"Could not find default image: {e}")
    
    # Initialize tester
    try:
        tester = ExperimentalPromptTester(args.model, debug=not args.no_debug)
        
        if len(prompts) == 1:
            # Single prompt test
            prompt = prompts[0]
            
            if args.compare_baseline:
                result = tester.compare_with_baseline(prompt, image_path)
                print("\n💾 Results available in comparison object")
            else:
                result = tester.test_prompt(prompt, image_path)
                print("\n💾 Result available in result object")
        
        else:
            # Batch test
            results = tester.batch_test_prompts(prompts, image_path)
            print(f"\n💾 {len(results)} results available in results list")
        
        print("\n✅ Experimental prompt testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()