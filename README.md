# Vision Model Comparison Framework (LMM_POC)

A comprehensive evaluation framework for comparing vision-language models on business document key-value extraction tasks. This project provides a complete pipeline for benchmarking Llama-3.2-Vision and InternVL3 models with ground truth evaluation, GPU memory optimization, and detailed performance reporting.

## 🔬 Overview

This framework implements a rigorous comparison methodology for vision-language models, specifically focused on extracting structured data from business documents (invoices, statements, receipts). It features modular architecture, advanced GPU memory optimization, comprehensive evaluation metrics, and production-ready reporting capabilities.

### Key Features

- **Multi-Model Support**: Llama-3.2-11B-Vision-Instruct and InternVL3 (2B/8B) models with automatic variant detection
- **Advanced GPU Optimization**: V100-specific optimizations, memory fragmentation detection, and resilient processing
- **Model-Size-Aware Processing**: Intelligent batch sizing based on model variant and available GPU memory
- **Comprehensive Evaluation**: 25-field business document extraction with ground truth validation
- **Production Pipeline**: Batch processing, accuracy metrics, deployment readiness assessment
- **Interactive Notebooks**: Jupyter notebooks for experimentation and single-document processing
- **Memory Management**: 8-bit quantization, fallback strategies, and emergency cleanup procedures
- **Detailed Reporting**: Executive summaries, technical evaluation, and deployment checklists

## 🏗 Architecture

```
LMM_POC/
├── README.md                      # This file
├── environment.yml                # Conda environment specification
├── unified_setup.sh              # Automated setup script
├── CLAUDE.md                      # Project-specific development guidelines
│
├── common/                       # Shared utilities and configuration
│   ├── config.py                # Centralized configuration with dynamic model detection
│   ├── evaluation_utils.py      # Evaluation and parsing utilities
│   ├── reporting.py             # Report generation and formatting
│   └── gpu_optimization.py      # V100 optimizations and memory management
│
├── models/                       # Model-specific processors
│   ├── llama_processor.py       # Llama-3.2-Vision with V100 optimizations
│   └── internvl3_processor.py   # InternVL3 (2B/8B) with size-aware processing
│
├── notebooks/                    # Interactive Jupyter notebooks
│   ├── llama_VQA.ipynb         # Llama visual Q&A
│   ├── internvl3_VQA.ipynb     # InternVL3 visual Q&A
│   ├── llama_keyvalue.ipynb    # Llama field extraction
│   └── internvl3_keyvalue.ipynb # InternVL3 field extraction
│
├── evaluation_data/             # Sample data and ground truth
│   ├── synthetic_invoice_*.png  # Sample business documents
│   └── evaluation_ground_truth.csv # Reference extraction data
│
├── llama_keyvalue.py           # Llama batch evaluation script
├── internvl3_keyvalue.py       # InternVL3 batch evaluation script
└── compare_models.py           # Direct model comparison utility
```

### Modular Design Patterns

#### 1. **Unified Processor Interface**
Both model implementations follow a consistent interface:
```python
class ModelProcessor:
    def process_single_image(image_path: str) -> dict
    def process_image_batch(image_files: list) -> tuple
    def get_extraction_prompt() -> str
```

#### 2. **GPU Optimization Architecture**
Advanced GPU memory management with V100-specific optimizations:
```python
# Automatic memory optimization
configure_cuda_memory_allocation()
handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)
optimize_model_for_v100(model)
comprehensive_memory_cleanup(model, tokenizer)
```

#### 3. **Model-Size-Aware Configuration**
Intelligent batch sizing based on model variant and available memory:
```python
# Automatic model detection and batch sizing
size_aware_model_name = get_model_name_with_size("internvl3", model_path)
batch_size = get_auto_batch_size(size_aware_model_name, available_memory)
```

#### 4. **Shared Configuration System**
Centralized configuration in `common/config.py`:
- **EXTRACTION_FIELDS**: 25 business document fields with validation
- **Model paths**: Dynamic deployment configuration
- **Batch sizing**: Conservative, balanced, and aggressive strategies
- **GPU thresholds**: Memory-based optimization triggers
- **Evaluation metrics**: Deployment readiness criteria

#### 5. **Comprehensive Evaluation Pipeline**
1. **Image Discovery** → Find and validate document images
2. **Model Processing** → Extract structured data with timing metrics
3. **Response Parsing** → Convert model output to structured format
4. **Accuracy Evaluation** → Compare against ground truth with fuzzy matching
5. **Report Generation** → Executive summaries and deployment checklists

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **CUDA-compatible GPU recommended** (any modern GPU supported)
- **Conda package manager**
- **Model weights downloaded locally**

### 1. Environment Setup

The conda environment is preconfigured and ready to use:

```bash
# Add the activation alias to your bash profile (one-time setup)
echo 'alias activate_internvl="conda activate /efs/shared/.conda/envs/intern_env"' >> ~/.bashrc

# Reload your bash profile
source ~/.bashrc

# Now activate the environment using the alias
activate_internvl
```

### 2. Model Configuration

Edit paths in `common/config.py`:
```python
# Update these paths to your model locations
LLAMA_MODEL_PATH = "/path/to/Llama-3.2-11B-Vision-Instruct"
INTERNVL3_MODEL_PATH = "/path/to/InternVL3-2B"  # or InternVL3-8B

# The system automatically detects model variants and optimizes accordingly
```

### 3. Quick Evaluation

```bash
# Run Llama evaluation pipeline
python llama_keyvalue.py

# Run InternVL3 evaluation pipeline  
python internvl3_keyvalue.py

# Compare models directly
python compare_models.py

# Launch interactive notebooks
jupyter notebook
# Select kernel: "Python (Vision Notebooks)"
```

## 📊 Usage Examples

### Batch Processing with GPU Optimization

```python
from models.llama_processor import LlamaProcessor
from common.evaluation_utils import discover_images, evaluate_extraction_results

# Initialize processor (automatically applies GPU optimizations)
processor = LlamaProcessor(model_path="/path/to/model")

# Discover images
image_files = discover_images("./evaluation_data")

# Process batch with automatic memory management
results, statistics = processor.process_image_batch(image_files)

# Evaluate against ground truth
ground_truth = load_ground_truth("./evaluation_data/evaluation_ground_truth.csv")
evaluation = evaluate_extraction_results(results, ground_truth)
```

### Model-Size-Aware Processing

```python
from models.internvl3_processor import InternVL3Processor

# Processor automatically detects 2B vs 8B variant and optimizes batch size
processor = InternVL3Processor(model_path="/path/to/InternVL3-8B")
# Output: "Auto-detected batch size: 1 (GPU Memory: 15.8GB, Model: internvl3-8b)"

processor_2b = InternVL3Processor(model_path="/path/to/InternVL3-2B") 
# Output: "Auto-detected batch size: 4 (GPU Memory: 15.8GB, Model: internvl3-2b)"
```

### Single Document Processing

```python
# Process single image with automatic optimization
result = processor.process_single_image("document.png")

# Extract specific fields
abn = result['extracted_data'].get('ABN', 'Not found')
total = result['extracted_data'].get('TOTAL', 'Not found')
processing_time = result['processing_time']
```

### Interactive Notebooks

1. **VQA (Visual Question Answering)**
   - `llama_VQA.ipynb`: Natural language questions about documents
   - `internvl3_VQA.ipynb`: Compare response styles and accuracy

2. **Key-Value Extraction**
   - `llama_keyvalue.ipynb`: Structured field extraction with Llama
   - `internvl3_keyvalue.ipynb`: Structured field extraction with InternVL3

## 🎯 Evaluation Framework

### Extraction Fields (25 Total)
Business document fields extracted and evaluated:
- **Identification**: ABN, SUPPLIER, DOCUMENT_TYPE
- **Financial**: TOTAL, SUBTOTAL, GST, PRICES, QUANTITIES
- **Dates**: INVOICE_DATE, DUE_DATE, STATEMENT_PERIOD
- **Contact Info**: BUSINESS_ADDRESS, BUSINESS_PHONE, PAYER_EMAIL
- **Banking**: BANK_NAME, BSB_NUMBER, BANK_ACCOUNT_NUMBER, ACCOUNT_HOLDER
- **Additional**: DESCRIPTIONS, OPENING_BALANCE, CLOSING_BALANCE

### Accuracy Metrics
- **Field-level accuracy**: Exact and fuzzy matching for each of 25 fields
- **Document-level accuracy**: Overall extraction success per document
- **Deployment readiness**: Quality distribution (Good/Fair/Poor)
- **Performance metrics**: Processing time (extraction vs pipeline), memory usage, success rates

### Quality Thresholds
```python
DEPLOYMENT_READY_THRESHOLD = 0.9   # 90% accuracy for production
PILOT_READY_THRESHOLD = 0.8        # 80% accuracy for pilot testing
NEEDS_OPTIMIZATION_THRESHOLD = 0.7  # Below 70% needs major improvements
```

## 📈 Performance Characteristics

The framework reports **measured performance metrics only**. Performance varies based on:

### Hardware Dependencies
- **GPU Memory**: Affects batch size and processing speed
- **GPU Architecture**: V100 optimizations available, works on modern GPUs
- **System Memory**: Affects model loading and data processing

### Model Specifications
| Model | Parameters | Memory Requirements | 
|-------|------------|---------------------|
| **Llama-3.2-Vision** | 11B | GPU recommended, 8-bit quantization available |
| **InternVL3-2B** | 2B | Minimal GPU requirements, CPU fallback |
| **InternVL3-8B** | 8B | Moderate GPU requirements, 8-bit quantization |

### Measured Metrics (Examples)
All metrics are measured and reported during evaluation:
- **Extraction accuracy**: Reported per evaluation run
- **Processing speed**: Measured per document (extraction vs pipeline time)
- **Success rate**: Percentage of successful extractions
- **Memory usage**: GPU/CPU utilization during processing

## 🔧 GPU Optimization Features

### V100-Specific Optimizations
```python
# Automatic CUDA memory configuration
PYTORCH_CUDA_ALLOC_CONF = "max_split_size_mb:512,garbage_collection_threshold:0.6"

# Memory fragmentation detection and handling
handle_memory_fragmentation(threshold_gb=1.0, aggressive=True)

# Model-specific optimizations
optimize_model_for_v100(model)
```

### Memory Management Strategies
- **Conservative**: Safe batch sizes for limited memory
- **Balanced**: Default configuration for typical hardware  
- **Aggressive**: Maximum performance for high-end GPUs
- **Automatic**: Detects available memory and chooses strategy

### Fallback Mechanisms
- **Batch size reduction**: Automatic fallback on OOM errors
- **CPU processing**: Emergency fallback when GPU fails
- **Model reloading**: Complete memory reset for recovery
- **Graceful degradation**: Continues processing despite individual failures

## 📁 Output Files

Each evaluation run generates timestamped reports with clear timing distinctions:

```
output/
├── {model}_batch_extraction_{timestamp}.csv      # Main results
├── {model}_extraction_metadata_{timestamp}.csv   # Processing stats
├── {model}_evaluation_results_{timestamp}.json   # Detailed metrics
├── {model}_executive_summary_{timestamp}.md      # Stakeholder summary
└── {model}_deployment_checklist_{timestamp}.md   # Production readiness
```

### Performance Metrics Reported
- **Extraction time**: Core model inference only
- **Pipeline time**: Total time including loading, evaluation, reporting
- **Success rate**: Percentage of successful extractions
- **Memory usage**: Peak GPU/CPU utilization
- **Accuracy**: Field-level and document-level performance

## 🛠 Configuration Options

### Batch Size Strategies
```python
# Available in common/config.py
CONSERVATIVE_BATCH_SIZES = {
    "llama": 1,
    "internvl3-2b": 2,  # Safe for 2B model
    "internvl3-8b": 1,  # Conservative for 8B model
}

DEFAULT_BATCH_SIZES = {
    "llama": 1,
    "internvl3-2b": 4,  # Efficient for 2B model
    "internvl3-8b": 1,  # Balanced for 8B model
}
```

### Model Path Configuration
```python
# Dynamic model switching
switch_model("internvl3", "InternVL3-8B")  # Switch to 8B variant
switch_deployment("AISandbox")  # Switch deployment environment
show_current_config()  # Display current configuration
```

### GPU Memory Thresholds
```python
GPU_MEMORY_THRESHOLDS = {
    "low": 8,     # GB - Use conservative batching
    "medium": 16, # GB - Use default batching  
    "high": 24,   # GB - Use aggressive batching
}
```

## 🚨 Troubleshooting

### Common Issues

#### GPU Memory Management
```bash
# The framework handles OOM automatically with fallback strategies
# Monitor with: nvidia-smi
# Memory optimization is applied automatically
```

#### Model Loading Issues
```bash
# Check model paths in common/config.py
# Verify model files are downloaded correctly
# Use show_current_config() to verify configuration
```

#### Performance Optimization
- **Slow processing**: Framework automatically detects and optimizes for available hardware
- **Memory issues**: Automatic fallback strategies handle memory constraints
- **Batch size**: Automatically optimized based on model variant and GPU memory

### Hardware Requirements

#### Minimum Requirements
- **RAM**: 16GB system memory
- **Storage**: 50GB for models and data
- **GPU**: Optional (CPU fallback available)

#### Recommended Configuration
- **GPU**: Any modern CUDA-compatible GPU (optimization automatic)
- **CPU**: 8+ cores for efficient batch processing
- **SSD**: Fast storage for model loading

## 📚 Development

### Adding New Models
1. Create `models/newmodel_processor.py` following the interface pattern
2. Implement GPU optimization support using `common/gpu_optimization.py`
3. Add model-specific configuration to `common/config.py`
4. Create evaluation script using shared infrastructure

### Code Quality
```bash
# Apply formatting and checks
ruff check . --fix
ruff format .

# The project includes pre-commit hooks for quality control
```

### Testing
```bash
# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Test with sample data
python llama_keyvalue.py  # Processes sample documents
python internvl3_keyvalue.py  # Processes with different model
```


## 🙋‍♂️ Support

For questions and issues:
1. **Check configuration**: Use `show_current_config()` to verify setup
2. **Review CLAUDE.md**: Project-specific development guidelines
3. **Monitor GPU usage**: Use `nvidia-smi` to check memory and utilization
4. **Validate environment**: Ensure conda environment is activated
5. **Test with samples**: Run evaluation scripts with provided sample data

---

**Vision Model Comparison Framework** - Comprehensive evaluation with advanced GPU optimization for business document processing with vision-language models.