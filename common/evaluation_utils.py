"""
Shared evaluation utilities for vision model assessment.

This module contains functions for evaluating extraction results against
ground truth data, calculating accuracies, and managing evaluation data.
"""

import re
from pathlib import Path

import pandas as pd

from .config import EXTRACTION_FIELDS


def discover_images(directory_path):
    """
    Discover all image files in the specified directory.
    
    Args:
        directory_path (str): Path to directory containing images
        
    Returns:
        list: List of image file paths found in directory
    """
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    image_files = []
    
    for extension in image_extensions:
        image_files.extend(str(p) for p in Path(directory_path).glob(extension))
    
    # Sort for consistent processing order
    image_files.sort()
    return image_files


def parse_extraction_response(response_text, clean_conversation_artifacts=False):
    """
    Parse structured extraction response into dictionary.
    
    This function handles model responses that may contain conversation artifacts
    or formatting issues, extracting only the key-value pairs.
    
    Args:
        response_text (str): Raw model response containing key-value pairs
        clean_conversation_artifacts (bool): Whether to clean Llama-style artifacts
        
    Returns:
        dict: Parsed key-value pairs with all expected fields
    """
    if not response_text:
        return {field: "N/A" for field in EXTRACTION_FIELDS}
    
    # Clean Llama-specific conversation artifacts if requested
    if clean_conversation_artifacts:
        # Remove common Llama conversation patterns
        clean_patterns = [
            r"I'll extract.*?\n",
            r"I can extract.*?\n",
            r"Here (?:is|are) the.*?\n",
            r"Based on.*?\n",
            r"Looking at.*?\n",
            r"<\|start_header_id\|>.*?<\|end_header_id\|>",
            r"<image>",
            r"assistant\n\n",
            r"^\s*Extract.*?below\.\s*\n",
        ]
        
        for pattern in clean_patterns:
            response_text = re.sub(pattern, "", response_text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Initialize with N/A for all fields
    extracted_data = {field: "N/A" for field in EXTRACTION_FIELDS}
    
    # Process each line looking for key-value pairs
    lines = response_text.strip().split('\n')
    
    for line in lines:
        # Skip empty lines and non-key-value lines
        if not line.strip() or ':' not in line:
            continue
        
        # Clean the line from markdown formatting
        clean_line = re.sub(r'\*+([^*]+)\*+', r'\1', line)
        
        # Extract key and value
        parts = clean_line.split(':', 1)
        if len(parts) == 2:
            key = parts[0].strip().upper()
            value = parts[1].strip()
            
            # Store if it's an expected field
            if key in extracted_data:
                # Don't overwrite if we already have a non-N/A value
                if extracted_data[key] == "N/A" or not extracted_data[key]:
                    extracted_data[key] = value if value else "N/A"
    
    return extracted_data


def create_extraction_dataframe(results):
    """
    Create structured DataFrames from extraction results.
    
    Args:
        results (list): List of extraction result dictionaries
        
    Returns:
        tuple: (main_df, metadata_df) - Main extraction data and metadata
    """
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    
    # Main extraction DataFrame
    rows = []
    metadata_rows = []
    
    for result in results:
        # Main data row
        row = {'image_name': result['image_name']}
        row.update(result['extracted_data'])
        rows.append(row)
        
        # Metadata row
        if 'response_completeness' in result or 'content_coverage' in result:
            metadata_row = {
                'image_name': result['image_name'],
                'response_completeness': result.get('response_completeness', 0),
                'content_coverage': result.get('content_coverage', 0),
                'processing_time': result.get('processing_time', 0),
                'raw_response_length': result.get('raw_response_length', 0),
                'extracted_fields_count': result.get('extracted_fields_count', 0)
            }
            metadata_rows.append(metadata_row)
    
    # Create DataFrames
    main_df = pd.DataFrame(rows)
    metadata_df = pd.DataFrame(metadata_rows) if metadata_rows else pd.DataFrame()
    
    # Ensure column order: image_name first, then alphabetical fields
    if not main_df.empty:
        columns = ['image_name'] + sorted([col for col in main_df.columns if col != 'image_name'])
        main_df = main_df[columns]
    
    return main_df, metadata_df


def load_ground_truth(csv_path, show_sample=False):
    """
    Load ground truth data from CSV file.
    
    Args:
        csv_path (str): Path to ground truth CSV file
        show_sample (bool): Whether to display sample data
        
    Returns:
        dict: Mapping of image names to ground truth data
    """
    try:
        # Load CSV
        ground_truth_df = pd.read_csv(csv_path)
        
        print(f"📊 Ground truth CSV loaded with {len(ground_truth_df)} rows and {len(ground_truth_df.columns)} columns")
        print(f"📋 Available columns: {list(ground_truth_df.columns)}")
        
        # Check for image name column (try different variations)
        image_name_col = None
        for col_name in ['image_name', 'Image_Name', 'image_file', 'filename', 'file_name', 'IMAGE_NAME', 'IMAGE_FILE']:
            if col_name in ground_truth_df.columns:
                image_name_col = col_name
                break
        
        if image_name_col is None:
            # If no image name column found, use the first column
            image_name_col = ground_truth_df.columns[0]
            print(f"⚠️ No 'image_name' column found, using '{image_name_col}' as image identifier")
        else:
            print(f"✅ Using '{image_name_col}' as image identifier column")
        
        # Create mapping from image name to ground truth
        ground_truth_map = {}
        for _, row in ground_truth_df.iterrows():
            image_name = str(row[image_name_col]).strip()
            ground_truth = {col: str(row[col]) if pd.notna(row[col]) else "N/A" 
                          for col in ground_truth_df.columns if col != image_name_col}
            ground_truth_map[image_name] = ground_truth
        
        print(f"✅ Ground truth mapping created for {len(ground_truth_map)} images")
        
        # Show sample if requested
        if show_sample and ground_truth_map:
            print("\n📋 Sample ground truth data:")
            sample_image = list(ground_truth_map.keys())[0]
            sample_data = ground_truth_map[sample_image]
            print(f"Image: {sample_image}")
            for _i, (field, value) in enumerate(list(sample_data.items())[:5]):
                print(f"  {field}: {value[:50]}..." if len(str(value)) > 50 else f"  {field}: {value}")
            if len(sample_data) > 5:
                print(f"  ... and {len(sample_data) - 5} more fields")
        
        return ground_truth_map
        
    except FileNotFoundError:
        print(f"❌ Ground truth file not found: {csv_path}")
        return {}
    except Exception as e:
        print(f"❌ Error loading ground truth: {e}")
        print("💡 Please check that the CSV file:")
        print("   - Has a header row with column names")
        print("   - Contains an image name/filename column")
        print("   - Is properly formatted CSV")
        return {}


def calculate_field_accuracy(extracted_value, ground_truth_value, field_name):
    """
    Calculate accuracy for a specific field with specialized comparison logic.
    
    This function handles different types of fields with appropriate comparison
    methods (exact match, numeric comparison, date parsing, etc.).
    
    Args:
        extracted_value (str): Value extracted by the model
        ground_truth_value (str): Ground truth value
        field_name (str): Name of the field being compared
        
    Returns:
        float: Accuracy score (0.0 to 1.0)
    """
    # Convert to strings and clean
    extracted = str(extracted_value).strip() if extracted_value else "N/A"
    ground_truth = str(ground_truth_value).strip() if ground_truth_value else "N/A"
    
    # Both N/A is correct
    if extracted.upper() == "N/A" and ground_truth.upper() == "N/A":
        return 1.0
    
    # One is N/A but not the other
    if (extracted.upper() == "N/A") != (ground_truth.upper() == "N/A"):
        return 0.0
    
    # Normalize for comparison
    extracted_lower = extracted.lower()
    ground_truth_lower = ground_truth.lower()
    
    # Remove common formatting
    for char in [',', '$', '%', '(', ')', ' ']:
        extracted_lower = extracted_lower.replace(char, '')
        ground_truth_lower = ground_truth_lower.replace(char, '')
    
    # Exact match after normalization
    if extracted_lower == ground_truth_lower:
        return 1.0
    
    # Field-specific comparison logic
    if field_name in ['ABN', 'BSB_NUMBER', 'BANK_ACCOUNT_NUMBER']:
        # Numeric identifiers - exact match required
        extracted_digits = re.sub(r'\D', '', extracted)
        ground_truth_digits = re.sub(r'\D', '', ground_truth)
        return 1.0 if extracted_digits == ground_truth_digits else 0.0
    
    elif field_name in ['TOTAL', 'SUBTOTAL', 'GST', 'OPENING_BALANCE', 'CLOSING_BALANCE']:
        # Monetary values - numeric comparison
        try:
            extracted_num = float(re.sub(r'[^\d.-]', '', extracted))
            ground_truth_num = float(re.sub(r'[^\d.-]', '', ground_truth))
            # Allow 1% tolerance for rounding
            tolerance = abs(ground_truth_num * 0.01) if ground_truth_num != 0 else 0.01
            return 1.0 if abs(extracted_num - ground_truth_num) <= tolerance else 0.0
        except (ValueError, AttributeError):
            return 0.0
    
    elif field_name in ['INVOICE_DATE', 'DUE_DATE', 'STATEMENT_PERIOD']:
        # Date fields - flexible matching
        # Extract date components
        extracted_numbers = re.findall(r'\d+', extracted)
        ground_truth_numbers = re.findall(r'\d+', ground_truth)
        
        # Check if same date components are present
        if set(extracted_numbers) == set(ground_truth_numbers):
            return 1.0
        
        # Partial match for dates
        common = set(extracted_numbers) & set(ground_truth_numbers)
        if common and len(common) >= 2:  # At least month and day match
            return 0.8
        
        return 0.0
    
    elif field_name in ['DESCRIPTIONS', 'PRICES', 'QUANTITIES']:
        # List fields - check overlap
        # These fields may contain multiple items
        extracted_items = [item.strip() for item in re.split(r'[,;|\n]', extracted) if item.strip()]
        ground_truth_items = [item.strip() for item in re.split(r'[,;|\n]', ground_truth) if item.strip()]
        
        if not ground_truth_items:
            return 1.0 if not extracted_items else 0.0
        
        # Calculate overlap
        matches = sum(1 for item in extracted_items if any(
            item.lower() in gt_item.lower() or gt_item.lower() in item.lower() 
            for gt_item in ground_truth_items
        ))
        
        return matches / max(len(ground_truth_items), len(extracted_items)) if ground_truth_items else 0.0
    
    else:
        # Text fields - fuzzy matching
        # Check for substring match
        if extracted_lower in ground_truth_lower or ground_truth_lower in extracted_lower:
            return 0.9
        
        # Check word overlap for longer text
        extracted_words = set(extracted_lower.split())
        ground_truth_words = set(ground_truth_lower.split())
        
        if ground_truth_words:
            overlap = len(extracted_words & ground_truth_words) / len(ground_truth_words)
            if overlap >= 0.8:
                return overlap
        
        return 0.0


def evaluate_extraction_results(extraction_results, ground_truth_map):
    """
    Evaluate extraction results against ground truth data.
    
    Args:
        extraction_results (list): List of extraction result dictionaries
        ground_truth_map (dict): Mapping of image names to ground truth
        
    Returns:
        dict: Comprehensive evaluation summary with metrics
    """
    if not extraction_results or not ground_truth_map:
        return {
            'total_images': 0,
            'overall_accuracy': 0.0,
            'field_accuracies': {field: 0.0 for field in EXTRACTION_FIELDS},
            'evaluation_data': []
        }
    
    evaluation_data = []
    field_accuracies = {field: [] for field in EXTRACTION_FIELDS}
    overall_accuracies = []
    
    for result in extraction_results:
        image_name = result['image_name']
        extracted_data = result['extracted_data']
        
        # Get ground truth for this image
        if image_name not in ground_truth_map:
            image_base = Path(image_name).stem
            matching_keys = [k for k in ground_truth_map.keys() if Path(k).stem == image_base]
            
            if matching_keys:
                ground_truth = ground_truth_map[matching_keys[0]]
            else:
                print(f"⚠️ No ground truth found for {image_name}")
                continue
        else:
            ground_truth = ground_truth_map[image_name]
        
        # Calculate field-by-field accuracy
        image_accuracies = {}
        for field in EXTRACTION_FIELDS:
            extracted_value = extracted_data.get(field, "N/A")
            ground_truth_value = ground_truth.get(field, "N/A")
            
            accuracy = calculate_field_accuracy(extracted_value, ground_truth_value, field)
            image_accuracies[field] = accuracy
            field_accuracies[field].append(accuracy)
        
        # Calculate overall accuracy for this image
        overall_accuracy = sum(image_accuracies.values()) / len(image_accuracies)
        overall_accuracies.append(overall_accuracy)
        
        # Store evaluation data
        eval_record = {
            'image_name': image_name,
            'overall_accuracy': overall_accuracy,
            'field_accuracies': image_accuracies,
            'extracted_data': extracted_data,
            'ground_truth': ground_truth
        }
        evaluation_data.append(eval_record)
    
    # Calculate summary statistics
    total_accuracy = sum(overall_accuracies) / len(overall_accuracies) if overall_accuracies else 0.0
    
    # Find best and worst performing images
    if evaluation_data:
        best_image = max(evaluation_data, key=lambda x: x['overall_accuracy'])
        worst_image = min(evaluation_data, key=lambda x: x['overall_accuracy'])
    else:
        best_image = worst_image = None
    
    evaluation_summary = {
        'total_images': len(evaluation_data),
        'overall_accuracy': total_accuracy,
        'best_performing_image': best_image['image_name'] if best_image else "N/A",
        'best_performance_accuracy': best_image['overall_accuracy'] if best_image else 0.0,
        'worst_performing_image': worst_image['image_name'] if worst_image else "N/A",
        'worst_performance_accuracy': worst_image['overall_accuracy'] if worst_image else 0.0,
        'perfect_documents': sum(1 for acc in overall_accuracies if acc >= 0.99),
        'field_accuracies': {
            field: sum(accs) / len(accs) if accs else 0.0
            for field, accs in field_accuracies.items()
        },
        'evaluation_data': evaluation_data
    }
    
    return evaluation_summary