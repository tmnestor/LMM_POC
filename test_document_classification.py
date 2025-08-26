#!/usr/bin/env python3
"""
Document Type Classification Testing - Phase 1

Test the document type detection accuracy on existing evaluation data.
This script provides immediate feedback on classification performance.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from common.document_type_detector import DocumentTypeDetector
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running from the project root directory")
    sys.exit(1)


def test_classification_with_processor(processor_type: str = "llama"):
    """
    Test document classification with a specific processor.
    
    Args:
        processor_type: "llama" or "internvl3"
    """
    print(f"🚀 Testing Document Classification with {processor_type.upper()}")
    print("="*60)
    
    # Import processor
    try:
        if processor_type.lower() == "llama":
            from models.llama_processor import LlamaProcessor
            processor = LlamaProcessor()
        elif processor_type.lower() == "internvl3":
            from models.internvl3_processor import InternVL3Processor
            processor = InternVL3Processor()
        else:
            raise ValueError(f"Unsupported processor type: {processor_type}")
            
        print(f"✅ {processor_type.upper()} processor initialized")
        
    except Exception as e:
        print(f"❌ Failed to initialize {processor_type} processor: {e}")
        print("💡 Make sure you're running on a machine with GPU and model access")
        return None
    
    # Initialize detector
    detector = DocumentTypeDetector(processor)
    print("✅ Document type detector initialized")
    print(f"🎯 Confidence threshold: {detector.confidence_threshold}")
    
    return detector


def run_single_image_test(detector: DocumentTypeDetector, image_path: str):
    """Test classification on a single image."""
    print(f"\n🔍 Testing single image: {Path(image_path).name}")
    print("-" * 40)
    
    try:
        result = detector.detect_document_type(image_path)
        
        print(f"📄 Image: {Path(image_path).name}")
        print(f"🏷️  Type: {result['type']}")
        print(f"🎯 Confidence: {result['confidence']:.3f}")
        print(f"⏱️  Time: {result['processing_time']:.2f}s")
        print(f"💭 Reasoning: {result.get('reasoning', 'None provided')}")
        
        if result.get('fallback_used'):
            print("⚠️  Fallback classification used")
        if result.get('manual_review_needed'):
            print("🔍 Manual review recommended")
            
        return result
        
    except Exception as e:
        print(f"❌ Classification failed: {e}")
        return None


def run_batch_test(detector: DocumentTypeDetector, test_directory: str = "evaluation_data"):
    """Test classification on all images in a directory."""
    print(f"\n🔬 Running batch classification test on {test_directory}/")
    print("-" * 50)
    
    try:
        # Run batch classification
        results = detector.batch_classify_images(test_directory)
        
        if not results:
            print("❌ No results to analyze")
            return []
        
        # Generate and display report
        report = detector.generate_classification_report(results)
        print(report)
        
        # Show detailed results for each image
        print("\n📋 DETAILED RESULTS:")
        print("-" * 30)
        
        for result in results:
            if result.get('error'):
                print(f"❌ {result['image_name']}: ERROR - {result.get('reasoning', 'Unknown error')}")
            else:
                confidence_icon = "✅" if result['confidence'] >= detector.confidence_threshold else "⚠️"
                fallback_note = " (fallback)" if result.get('fallback_used') else ""
                print(f"{confidence_icon} {result['image_name']}: {result['type'].upper()} ({result['confidence']:.2f}){fallback_note}")
        
        return results
        
    except Exception as e:
        print(f"❌ Batch test failed: {e}")
        return []


def analyze_known_document_types(results: list):
    """
    Analyze results against known document types (if available).
    This would require ground truth data or filename conventions.
    """
    print("\n🔍 GROUND TRUTH ANALYSIS")
    print("-" * 30)
    
    # Simple filename-based analysis
    correct_predictions = 0
    total_predictions = 0
    
    for result in results:
        if result.get('error'):
            continue
            
        filename = result['image_name'].lower()
        predicted_type = result['type']
        
        # Simple filename-based ground truth detection
        actual_type = None
        if 'invoice' in filename:
            actual_type = 'invoice'
        elif 'statement' in filename or 'bank' in filename:
            actual_type = 'bank_statement'
        elif 'receipt' in filename:
            actual_type = 'receipt'
        
        if actual_type:
            total_predictions += 1
            if predicted_type == actual_type:
                correct_predictions += 1
                print(f"✅ {result['image_name']}: {predicted_type} == {actual_type}")
            else:
                print(f"❌ {result['image_name']}: {predicted_type} != {actual_type}")
        else:
            print(f"❔ {result['image_name']}: Unknown ground truth")
    
    if total_predictions > 0:
        accuracy = (correct_predictions / total_predictions) * 100
        print(f"\n📊 ACCURACY: {correct_predictions}/{total_predictions} ({accuracy:.1f}%)")
        
        if accuracy >= 90:
            print("🎉 Excellent classification accuracy!")
        elif accuracy >= 80:
            print("👍 Good classification accuracy")
        elif accuracy >= 70:
            print("⚠️ Moderate accuracy - consider tuning")
        else:
            print("🚨 Low accuracy - needs improvement")
    else:
        print("💡 No ground truth available for accuracy calculation")
        print("💡 Consider using descriptive filenames for testing")


def main():
    """Main testing routine."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test document type classification")
    parser.add_argument('--model', choices=['llama', 'internvl3'], default='llama',
                       help='Model to test (default: llama)')
    parser.add_argument('--image', type=str,
                       help='Test single image instead of batch')
    parser.add_argument('--directory', type=str, default='evaluation_data',
                       help='Directory for batch testing (default: evaluation_data)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = test_classification_with_processor(args.model)
    if not detector:
        return
    
    # Run tests
    if args.image:
        # Single image test
        if not Path(args.image).exists():
            print(f"❌ Image not found: {args.image}")
            return
        result = run_single_image_test(detector, args.image)
        
    else:
        # Batch test
        if not Path(args.directory).exists():
            print(f"❌ Directory not found: {args.directory}")
            return
        
        results = run_batch_test(detector, args.directory)
        if results:
            analyze_known_document_types(results)
    
    print("\n✅ Testing completed!")
    print("\n💡 Next steps:")
    print("   1. Review classification accuracy")
    print("   2. Adjust confidence threshold if needed")
    print("   3. Refine prompts for better accuracy")
    print("   4. Test with different document types")


if __name__ == "__main__":
    main()