# Jupyter Notebook Comparison: Llama vs InternVL3

## Overview

Both notebooks demonstrate the **simplified YAML-first architecture** with identical document detection approaches, showcasing the unified system across both models.

## Notebook Structure (Identical)

### 1. **Title & Introduction**
- **Llama**: "Llama Document-Aware Extraction Demo" 
- **InternVL3**: "InternVL3 Document-Aware Extraction Demo"
- Both emphasize **YAML-first architecture** and **document-specific field filtering**

### 2. **Import Section** 
- **Llama**: `from llama_document_aware import DocumentAwareLlamaHandler`
- **InternVL3**: `from internvl3_document_aware import DocumentAwareInternVL3Handler`
- Same supporting imports and project setup

### 3. **Configuration**
- **Llama**: `/path/to/Llama-3.2-11B-Vision-Instruct`
- **InternVL3**: `/path/to/InternVL3-8B`  
- Same test images and ground truth paths

### 4. **Demo Image Display**
- Identical image display logic
- Same visual presentation

### 5. **Step 1: Document Type Detection**
- **Both**: Use simplified YAML-first detection
- **Both**: Same `detect_and_classify_document()` signature
- **Both**: Identical field reduction metrics (20 receipt, 29 invoice, 16 bank)

### 6. **Step 2: Prompt Generation**
- **Llama**: Shows complete YAML-first prompt via `generate_dynamic_prompt()`
- **InternVL3**: Shows document-specific field list (direct extraction approach)
- Different visualization but same underlying YAML-first principle

### 7. **Step 3: Document-Aware Extraction**
- **Both**: Use `process_document_aware()` method
- **Both**: Same timing and field coverage metrics
- **Both**: Identical result structure

### 8. **Step 4: Data Visualization**
- Identical extracted data display with ✅/❌ indicators
- Same summary statistics and field counting

### 9. **Step 5: Ground Truth Evaluation**
- **Both**: Use `evaluate_document_aware()` method
- **Both**: Same accuracy metrics and ATO compliance checks
- Identical evaluation framework

### 10. **Final Performance Analysis**
- Same comprehensive timing breakdown
- Same throughput and efficiency calculations
- Model-specific optimizations highlighted

## Key Differences

### **Model-Specific Features**

| Aspect | Llama Notebook | InternVL3 Notebook |
|--------|----------------|-------------------|
| **Model Path** | Llama-3.2-11B-Vision-Instruct | InternVL3-8B |
| **Memory Usage** | ~22GB VRAM | ~4GB VRAM |
| **Prompt Display** | Full YAML prompt (detailed) | Field list (concise) |
| **Performance Target** | 45s threshold | 30s threshold (faster) |
| **Memory Advantage** | Large model accuracy | 5.5x memory efficiency |
| **V100 Suitability** | Requires 8-bit quantization | Native V100 friendly |

### **Architectural Consistency** 

| Feature | Both Notebooks |
|---------|----------------|
| **Detection Method** | ✅ Identical YAML-first approach |
| **Field Reduction** | ✅ Same document-specific filtering |
| **Schema Loading** | ✅ Same `config/fields.yaml` |
| **Evaluation Framework** | ✅ Same metrics and thresholds |
| **Code Structure** | ✅ Mirror cell-by-cell organization |

## Usage Examples

### **Llama Notebook**
```bash
# Uses Llama-3.2-11B-Vision for high accuracy
MODEL_PATH = "/path/to/Llama-3.2-11B-Vision-Instruct"
handler = DocumentAwareLlamaHandler(MODEL_PATH, debug=True)
```

### **InternVL3 Notebook** 
```bash
# Uses InternVL3-8B for memory efficiency
MODEL_PATH = "/path/to/InternVL3-8B"  
handler = DocumentAwareInternVL3Handler(MODEL_PATH, debug=True)
```

## Testing Scenarios

### **Same Test Images Work Across Both**
- `image_004.png` - Receipt (20 fields)
- `image_042.png` - Invoice (29 fields)  
- `image_006.png` - Bank Statement (16 fields)

### **Identical Workflow**
1. Initialize handler → 2. Detect document type → 3. Extract fields → 4. Evaluate accuracy
2. Both show same YAML-first detection success
3. Both demonstrate document-aware field filtering benefits

## Performance Comparison Output

### **Expected Results**
- **Field Reduction**: Both show 41-67% fewer fields than universal
- **Detection Method**: Both show "YAML-first detection" 
- **Processing Speed**: InternVL3 typically faster due to smaller model
- **Memory Usage**: InternVL3 shows significant memory advantage
- **Accuracy**: Both should achieve >95% with document-aware filtering

## Architecture Benefits Demonstrated

### **Both Notebooks Showcase**
1. **Simplified YAML-First Detection**: Same approach, no complex 518-line detector
2. **Document-Aware Efficiency**: Intelligent field filtering based on document type
3. **Consistent Configuration**: Same YAML prompts and schema files
4. **Production-Ready Metrics**: Comprehensive timing and throughput analysis
5. **Model Choice Flexibility**: User can pick Llama (accuracy) or InternVL3 (efficiency)

The notebooks prove that the **YAML-first migration was successful** - both models now use identical detection approaches while maintaining their unique strengths (Llama: accuracy, InternVL3: efficiency).