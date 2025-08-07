# Vision Model Comparison Framework (LMM_POC)

A comprehensive evaluation framework for comparing vision-language models on business document key-value extraction tasks. This project provides a complete pipeline for benchmarking Llama-3.2-Vision and InternVL3 models with ground truth evaluation and detailed performance reporting.

## 🔬 Overview

This framework implements a rigorous comparison methodology for vision-language models, specifically focused on extracting structured data from business documents (invoices, statements, receipts). It features modular architecture, comprehensive evaluation metrics, and production-ready reporting capabilities.

### Key Features

- **Multi-Model Support**: Llama-3.2-11B-Vision-Instruct and InternVL3 (2B/8B) models
- **Comprehensive Evaluation**: 25-field business document extraction with ground truth validation
- **Production Pipeline**: Batch processing, accuracy metrics, deployment readiness assessment
- **Interactive Notebooks**: Jupyter notebooks for experimentation and single-document processing
- **Memory Optimization**: 8-bit quantization support for GPU memory constraints
- **Detailed Reporting**: Executive summaries, technical evaluation, and deployment checklists

## 🏗 Architecture

```
LMM_POC/
├── README.md                      # This file
├── environment.yml                # Conda environment specification
├── unified_setup.sh              # Automated setup script
│
├── common/                       # Shared utilities and configuration
│   ├── config.py                # Centralized configuration
│   ├── evaluation_utils.py      # Evaluation and parsing utilities
│   └── reporting.py             # Report generation
│
├── models/                       # Model-specific processors
│   ├── llama_processor.py       # Llama-3.2-Vision implementation
│   └── internvl3_processor.py   # InternVL3 (2B/8B) implementation
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
│
└── docs/                       # Additional documentation
    ├── CLAUDE.md               # Development guidelines
    ├── REFACTORING_SUMMARY.md  # Code architecture notes
    └── ground_truth_evaluation_system.md
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

#### 2. **Shared Configuration**
Centralized configuration in `common/config.py`:
- **EXTRACTION_FIELDS**: 25 business document fields
- **Model paths**: Development vs production deployment paths
- **Evaluation thresholds**: Deployment readiness criteria
- **Output patterns**: Consistent file naming conventions

#### 3. **Comprehensive Evaluation Pipeline**
1. **Image Discovery** → Find and validate document images
2. **Model Processing** → Extract structured data with timing metrics
3. **Response Parsing** → Convert model output to structured format
4. **Accuracy Evaluation** → Compare against ground truth with fuzzy matching
5. **Report Generation** → Executive summaries and deployment checklists

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+**
- **CUDA-compatible GPU recommended** (16GB+ VRAM for Llama, 4GB+ for InternVL3)
- **Conda package manager**
- **Model weights downloaded locally**

### 1. Environment Setup

```bash
# Clone and navigate to project
cd LMM_POC

# Automated setup (recommended)
source unified_setup.sh

# Manual setup alternative
conda env create -f environment.yml
conda activate vision_notebooks
python -m ipykernel install --user --name vision_notebooks --display-name "Python (Vision Notebooks)"
```

### 2. Model Configuration

Edit paths in `common/config.py`:
```python
# Update these paths to your model locations
LLAMA_MODEL_PATH = "/path/to/Llama-3.2-11B-Vision-Instruct"
INTERNVL3_MODEL_PATH = "/path/to/InternVL3-2B"  # or InternVL3-8B

# Update data paths if needed
DATA_DIR = "./evaluation_data"
OUTPUT_DIR = "./output"
```

### 3. Quick Evaluation

```bash
# Run Llama evaluation pipeline
python llama_keyvalue.py

# Run InternVL3 evaluation pipeline  
python internvl3_keyvalue.py

# Launch interactive notebooks
jupyter notebook
# Select kernel: "Python (Vision Notebooks)"
```

## 📊 Usage Examples

### Batch Processing

```python
from models.llama_processor import LlamaProcessor
from common.evaluation_utils import discover_images, evaluate_extraction_results

# Initialize processor
processor = LlamaProcessor(model_path="/path/to/model")

# Discover images
image_files = discover_images("./evaluation_data")

# Process batch
results, statistics = processor.process_image_batch(image_files)

# Evaluate against ground truth
ground_truth = load_ground_truth("./evaluation_data/evaluation_ground_truth.csv")
evaluation = evaluate_extraction_results(results, ground_truth)
```

### Single Document Processing

```python
# Process single image
result = processor.process_single_image("document.png")

# Extract specific fields
abn = result.get('ABN', 'Not found')
total = result.get('TOTAL', 'Not found')
invoice_date = result.get('INVOICE_DATE', 'Not found')
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
- **Performance metrics**: Processing time, memory usage, success rates

### Quality Thresholds
```python
DEPLOYMENT_READY_THRESHOLD = 0.9   # 90% accuracy for production
PILOT_READY_THRESHOLD = 0.8        # 80% accuracy for pilot testing
NEEDS_OPTIMIZATION_THRESHOLD = 0.7  # Below 70% needs major improvements
```

## 📈 Performance Comparison

| Model | Parameters | Memory Usage | Processing Speed | Typical Accuracy |
|-------|------------|--------------|------------------|------------------|
| **Llama-3.2-Vision** | 11B | ~22GB VRAM | 3-5s per doc | 80-95% |
| **InternVL3-2B** | 2B | ~4GB VRAM | 1-3s per doc | 75-90% |
| **InternVL3-8B** | 8B | ~16GB VRAM | 2-4s per doc | 80-92% |

### Model Characteristics

#### Llama-3.2-11B-Vision-Instruct
- **Strengths**: Detailed responses, built-in preprocessing, high accuracy
- **Requirements**: 16GB+ VRAM (or 8-bit quantization)
- **Use Case**: High-accuracy production deployments

#### InternVL3 (2B/8B)
- **InternVL3-2B**: Ultra-efficient, 4GB+ VRAM, ideal for rapid prototyping
- **InternVL3-8B**: Balanced performance, 16GB+ VRAM, higher accuracy than 2B
- **Strengths**: Memory efficient vs Llama, fast inference, simple API, CPU fallback available
- **Use Case**: Resource-constrained environments (2B), balanced performance needs (8B)

## 📁 Output Files

Each evaluation run generates timestamped reports:

```
output/
├── {model}_batch_extraction_{timestamp}.csv      # Main results
├── {model}_extraction_metadata_{timestamp}.csv   # Processing stats
├── {model}_evaluation_results_{timestamp}.json   # Detailed metrics
├── {model}_executive_summary_{timestamp}.md      # Stakeholder summary
└── {model}_deployment_checklist_{timestamp}.md   # Production readiness
```

### Report Contents

#### Executive Summary
- **Overall Performance**: Accuracy percentages and document quality distribution
- **Field Analysis**: Best and worst performing fields
- **Deployment Readiness**: Go/no-go recommendation with supporting metrics

#### Technical Evaluation
- **Field-level accuracy**: Detailed breakdown for all 25 fields
- **Processing statistics**: Timing, memory usage, error rates
- **Quality metrics**: Document-level performance distribution

#### Deployment Checklist
- **Infrastructure requirements**: GPU memory, processing capacity
- **Accuracy thresholds**: Field-specific performance criteria
- **Production considerations**: Batch processing capabilities, error handling

## 🔧 Configuration

### Environment Variables
```bash
export CUDA_VISIBLE_DEVICES=0  # Specify GPU
export KMP_DUPLICATE_LIB_OK=TRUE  # Fix OpenMP conflicts
```

### Model Optimization
```python
# 8-bit quantization for memory constraints
load_in_8bit = True  # Configured in processors

# Batch size adjustment
batch_size = 1  # Conservative for memory-constrained environments
```

### Custom Fields
To add new extraction fields:
1. Update `EXTRACTION_FIELDS` in `common/config.py`
2. Add corresponding entries in ground truth CSV
3. Update prompts in model processors if needed

## 🛠 Development

### Adding New Models
1. Create `models/newmodel_processor.py`:
```python
class NewModelProcessor:
    def __init__(self, model_path: str):
        # Model initialization
        
    def process_single_image(self, image_path: str) -> dict:
        # Single document processing
        
    def process_image_batch(self, image_files: list) -> tuple:
        # Batch processing with statistics
```

2. Create main evaluation script:
```python
from models.newmodel_processor import NewModelProcessor
from common.evaluation_utils import *
from common.reporting import *

# Use shared evaluation infrastructure
```

### Testing
```bash
# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Test with sample data
python llama_keyvalue.py  # Should process 20 sample invoices
```

### Code Quality
```bash
# Apply formatting (if ruff available)
ruff check . --fix
ruff format .
```

## 🚨 Troubleshooting

### Common Issues

#### GPU Memory Errors
```bash
# Error: CUDA out of memory
# Solution: Enable 8-bit quantization (already configured)
# Alternative: Use CPU processing (slower but functional)
```

#### Model Loading Failures
```bash
# Error: Model not found
# Solution: Update paths in common/config.py
# Check: Model files downloaded correctly
```

#### Import Errors
```bash
# Error: ModuleNotFoundError
# Solution: Ensure conda environment activated
conda activate vision_notebooks

# Verify: All dependencies installed
pip list | grep transformers
```

#### Performance Issues
- **Slow processing**: Check GPU availability with `nvidia-smi`
- **High memory usage**: Reduce batch size or use 8-bit quantization
- **CPU fallback**: Expected behavior when GPU unavailable

### Hardware Requirements

#### Minimum Requirements
- **RAM**: 16GB system memory
- **Storage**: 50GB for models and data
- **GPU**: Optional but recommended (any CUDA-compatible GPU)

#### Recommended Configuration
- **GPU**: 22GB+ VRAM for Llama, 4GB+ VRAM for InternVL3-2B, 16GB+ VRAM for InternVL3-8B
- **CPU**: 8+ cores for efficient batch processing
- **SSD**: Fast storage for model loading

## 📚 Additional Resources

### Documentation
- **CLAUDE.md**: Development guidelines and git workflow
- **REFACTORING_SUMMARY.md**: Architecture decisions and code organization
- **ground_truth_evaluation_system.md**: Evaluation methodology details

### Model Resources
- **Llama-3.2-Vision**: [Hugging Face Model Hub](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
- **InternVL3-2B**: [Hugging Face Model Hub](https://huggingface.co/OpenGVLab/InternVL3-2B)
- **InternVL3-8B**: [Hugging Face Model Hub](https://huggingface.co/OpenGVLab/InternVL3-8B)

### Dependencies
- **transformers**: 4.45.2 (pinned for Llama compatibility)
- **torch**: 2.0.0+ with CUDA support
- **timm**: 0.9.0+ (required for InternVL3)
- **einops**: 0.6.0+ (required for InternVL3)

## 🤝 Contributing

### Development Guidelines
1. **Follow modular architecture**: Use shared utilities when possible
2. **Maintain consistency**: Follow existing patterns for new models
3. **Document thoroughly**: Add docstrings and inline comments
4. **Test comprehensively**: Verify with sample data before committing

### Adding Features
1. **New models**: Follow processor interface pattern
2. **New evaluation metrics**: Extend `evaluation_utils.py`
3. **New report formats**: Extend `reporting.py`
4. **New fields**: Update configuration and ground truth data

## 📄 License

This project is intended for research and evaluation purposes. Model-specific licensing terms apply to downloaded model weights.

## 🙋‍♂️ Support

For questions and issues:
1. Check the troubleshooting section above
2. Review existing documentation in the `docs/` directory
3. Examine sample notebooks for usage patterns
4. Validate environment setup with `unified_setup.sh`

---

**Vision Model Comparison Framework** - Comprehensive evaluation for business document processing with vision-language models.