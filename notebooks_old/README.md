# Notebooks Directory

Interactive Jupyter notebooks for vision-language model evaluation, comparison, and experimentation in the LMM POC project.

## 📋 Overview

This directory contains comprehensive Jupyter notebooks that provide interactive interfaces for evaluating and comparing large multimodal models (LMMs). All notebooks follow a consistent 7-cell architecture pattern and leverage the modular codebase design.

## 🗂️ Notebook Collection

### Model Evaluation Notebooks

#### `llama_keyvalue.ipynb`
**Purpose:** Comprehensive evaluation of Llama-3.2-11B-Vision-Instruct for business document key-value extraction

**Features:**
- **Modular Architecture Integration:** Uses `common/` modules for configuration, evaluation, and reporting
- **Professional Visualization:** Business-grade charts with inline display and PNG export
- **Comprehensive Analysis:** 25 business document fields with accuracy metrics
- **Performance Metrics:** Processing speed, memory usage, deployment readiness assessment
- **Executive Reporting:** Generates stakeholder-ready reports and visualizations

**Key Capabilities:**
- Document discovery and batch processing
- Ground truth evaluation with fuzzy matching
- Field-level accuracy analysis with business categorization
- Performance dashboard generation (2x2 layout)
- Memory optimization with 8-bit quantization support

#### `internvl3_keyvalue.ipynb`  
**Purpose:** Comprehensive evaluation of InternVL3-2B for business document key-value extraction

**Features:**
- **Memory Efficient:** ~4GB VRAM requirement (5x more efficient than Llama)
- **Fast Processing:** 1-3 seconds per document average
- **Tile-Based Preprocessing:** Advanced dynamic image preprocessing
- **Modular Design:** Same architecture as Llama notebook for consistency
- **Comparative Analysis:** Direct performance comparison capabilities

**Key Capabilities:**
- Optimized batch processing with memory management
- Dynamic preprocessing with tile-based approach
- Same 25-field business document analysis
- Professional visualization suite
- Resource usage monitoring and optimization

### Visual Question Answering (VQA) Notebooks

#### `llama_VQA.ipynb`
**Purpose:** Interactive visual question answering with Llama-3.2-11B-Vision

**Features:**
- **Simple Interface:** Direct question-answer interaction with document images
- **Conversation Support:** Advanced chat template handling
- **High Detail:** Comprehensive responses with reasoning
- **File Output:** Save responses for further analysis

**Use Cases:**
- Interactive document analysis
- Content understanding and verification
- Complex reasoning tasks
- Research and experimentation

#### `internvl3_VQA.ipynb`
**Purpose:** Interactive visual question answering with InternVL3-2B

**Features:**
- **Lightweight Processing:** Fast inference with low memory footprint
- **Concise Responses:** Direct, focused answers
- **Efficient Architecture:** Streamlined for quick experimentation
- **Flexible Input:** Support for various image formats and questions

**Use Cases:**
- Quick document queries
- Rapid prototyping
- Resource-constrained environments
- High-throughput scenarios

### Model Comparison

#### `compare_models.ipynb`
**Purpose:** Comprehensive side-by-side comparison of Llama-3.2-Vision vs InternVL3-2B

**Features:**
- **Auto-Discovery:** Automatically finds latest evaluation JSON files
- **Manual Override:** Custom file path specification for specific comparisons
- **Professional Dashboards:** 2x2 performance comparison layouts
- **Field-Level Heatmaps:** Detailed accuracy comparison across 25 business fields
- **Performance Delta Analysis:** Visual analysis of relative strengths and weaknesses
- **Executive Reporting:** Deployment recommendations and technical specifications
- **Interactive Visualization:** All charts display inline with professional styling

**Comparison Dimensions:**
- Overall accuracy and document processing statistics
- Field-level performance analysis with win/loss tracking
- Resource requirements (memory, processing speed)
- Quality distribution and perfect document rates
- Performance range analysis (best vs worst documents)
- Deployment suitability assessment

## 🏗️ Architecture Patterns

### Consistent 7-Cell Structure

All evaluation notebooks follow this standardized pattern:

1. **Cell 0: Documentation (Markdown)**
   - Purpose, features, and architecture overview
   - Model characteristics and technical specifications
   - Use case descriptions and capabilities

2. **Cell 1: Imports and Setup**
   - Modular architecture imports from `common/`
   - Model-specific processor imports
   - Environment configuration and validation

3. **Cell 2: Configuration Validation**
   - Path validation and configuration display
   - Model specifications and deployment information
   - Field extraction configuration summary

4. **Cell 3: Model Initialization**
   - Processor initialization with automatic configuration
   - Model loading with quantization and optimization
   - Generation configuration and batch size optimization

5. **Cell 4: Processing and Evaluation**
   - Image discovery and batch processing
   - Ground truth evaluation and metrics calculation
   - Real-time progress monitoring and statistics

6. **Cell 5: Comprehensive Reporting**
   - Report generation using shared utilities
   - Performance summary and deployment recommendations
   - File output management and validation

7. **Cell 6: Visualization and Analysis**
   - Professional chart generation with business styling
   - Inline visualization with matplotlib integration
   - Field-level analysis and performance categorization
   - Executive summary with deployment guidance

### Modular Design Integration

**Configuration Management:**
```python
from common.config import (
    DATA_DIR, OUTPUT_DIR, EXTRACTION_FIELDS,
    FIELD_COUNT, VIZ_COLORS, CHART_DPI
)
```

**Evaluation Utilities:**
```python
from common.evaluation_utils import (
    discover_images, create_extraction_dataframe,
    load_ground_truth, evaluate_extraction_results
)
```

**Professional Reporting:**
```python
from common.reporting import (
    generate_comprehensive_reports,
    print_evaluation_summary
)
```

**Model Processors:**
```python
from models.llama_processor import LlamaProcessor
from models.internvl3_processor import InternVL3Processor
```

## 🚀 Getting Started

### Environment Setup

1. **Conda Environment:**
   ```bash
   conda env create -f environment.yml
   conda activate vision_notebooks
   ```

2. **Start Jupyter:**
   ```bash
   jupyter notebook
   ```

3. **Kernel Selection:**
   - Select "Python (Vision Notebooks)" kernel
   - Ensure all dependencies are available

### Model Path Configuration

**CRITICAL:** Update model paths before running any notebooks:

**Development Paths (Local):**
```python
# In notebook cells or common/config.py
LLAMA_MODEL_PATH = "/path/to/Llama-3.2-11B-Vision-Instruct"
INTERNVL3_MODEL_PATH = "/path/to/InternVL3-2B" 
```

**Production Paths (Remote/Server):**
```python
# Toggle commented paths in common/config.py
LLAMA_MODEL_PATH = "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision-Instruct"
INTERNVL3_MODEL_PATH = "/home/jovyan/nfs_share/models/InternVL3-8B"
```

### Hardware Requirements

**Minimum System Requirements:**
- **RAM:** 16GB system memory
- **GPU:** Any CUDA-compatible GPU (CPU fallback available)
- **Storage:** 50GB available space for models and outputs

**Model-Specific Requirements:**
| Model | VRAM | Processing Speed | Best Use Case |
|-------|------|-----------------|---------------|
| **Llama-3.2-Vision** | ~22GB | 3-5s per document | High accuracy, detailed analysis |
| **InternVL3-2B** | ~4GB | 1-3s per document | Memory efficient, high throughput |

**Quantization Support:**
- Both models support 8-bit quantization for memory reduction
- Automatic fallback to CPU processing if GPU memory insufficient
- Dynamic batch size optimization based on available resources

### Data Requirements

**Evaluation Data Structure:**
```
evaluation_data/
├── images/                     # Document images for processing
│   ├── synthetic_invoice_*.png
│   └── real_document_*.jpg
└── evaluation_ground_truth.csv # Ground truth for accuracy evaluation
```

**Ground Truth Format:**
- CSV file with image names and expected field values
- 25 business document fields (ABN, TOTAL, INVOICE_NUMBER, etc.)
- Supports exact matching and fuzzy matching for evaluation

## 📊 Output Files

### Generated Reports and Visualizations

**Model Evaluation Outputs:**
- `{model}_batch_extraction_{timestamp}.csv` - Structured extraction results
- `{model}_evaluation_results_{timestamp}.json` - Detailed evaluation metrics
- `{model}_executive_summary_{timestamp}.md` - Executive summary report
- `{model}_field_accuracy_{timestamp}.png` - Field accuracy visualization
- `{model}_performance_dashboard_{timestamp}.png` - Performance overview

**Model Comparison Outputs:**
- `model_comparison_dashboard_{timestamp}.png` - Side-by-side performance comparison
- `field_accuracy_heatmap_{timestamp}.png` - Field-level accuracy heatmap
- `model_comparison_summary_{timestamp}.md` - Executive comparison report

**File Locations:**
- All outputs saved to `OUTPUT_DIR` (configured in `common/config.py`)
- Timestamped filenames prevent overwrites
- Professional PNG charts at 300 DPI for presentations
- Markdown reports compatible with business documentation systems

## 🔧 Development Workflow

### Notebook Execution Strategy

**For Local Development:**
1. **Code Development:** Edit notebooks locally with full IDE support
2. **Syntax Validation:** Run `ruff check` on extracted Python code
3. **Remote Execution:** Transfer to GPU server for model evaluation
4. **Results Analysis:** Download outputs for local analysis and reporting

**For Remote Execution:**
1. **Model Loading:** All models must run on remote GPU servers
2. **Batch Processing:** Leverage optimized batch processing for efficiency  
3. **Memory Management:** Automatic optimization based on available resources
4. **Progress Monitoring:** Real-time progress updates and error handling

### Code Quality Standards

**Python Standards:**
- All Python code must pass `ruff check` validation
- Line length maximum 108 characters
- Type hints using Python 3.11+ features
- Comprehensive error handling and user feedback

**Notebook Standards:**
- All code cells must have `execution_count` field (even if null)
- Consistent cell structure following 7-cell pattern
- Professional markdown documentation with proper headers
- Inline visualization with business-grade styling

## 🎯 Use Cases

### Business Applications

**Document Processing Pipeline:**
- Invoice processing and validation
- Bank statement analysis
- Contract key information extraction
- Compliance document review

**Model Selection and Deployment:**
- Performance benchmarking across document types
- Resource requirement analysis for deployment planning
- Accuracy vs efficiency trade-off analysis
- Deployment readiness assessment

### Research and Development

**Model Experimentation:**
- Prompt engineering and optimization
- Parameter tuning and performance analysis
- Comparative analysis across model architectures
- Ablation studies and feature analysis

**Performance Analysis:**
- Field-level accuracy deep dives
- Processing speed optimization
- Memory usage profiling and optimization
- Error analysis and failure case investigation

## 📚 Related Documentation

**Architecture Documentation:**
- [`docs/architecture_comparison.md`](../docs/architecture_comparison.md) - Detailed technical comparison
- [`docs/FIELD_MANAGEMENT_WORKFLOWS.md`](../docs/FIELD_MANAGEMENT_WORKFLOWS.md) - Field configuration management

**Configuration Files:**
- [`common/config.py`](../common/config.py) - Centralized configuration
- [`environment.yml`](../environment.yml) - Conda environment specification
- [`CLAUDE.md`](../CLAUDE.md) - Development guidelines and standards

**Scripts and Utilities:**
- [`compare_models.py`](../compare_models.py) - Command-line model comparison
- [`unified_setup.sh`](../unified_setup.sh) - Automated environment setup

## 🤝 Contributing

### Adding New Notebooks

**Follow the 7-Cell Pattern:**
1. Markdown documentation cell with comprehensive overview
2. Imports and setup with modular architecture integration
3. Configuration validation and path management  
4. Model initialization with optimization
5. Processing and evaluation with real-time feedback
6. Comprehensive reporting with file output
7. Visualization and analysis with business styling

**Code Quality Requirements:**
- All notebooks must pass ruff validation
- Include comprehensive error handling
- Provide user-friendly progress feedback
- Generate professional outputs suitable for business use

**Testing and Validation:**
- Test with both sample and real document data
- Validate output file generation and formatting
- Ensure compatibility with existing modular architecture
- Document hardware requirements and performance characteristics

---

*This notebooks directory provides a complete interactive environment for vision-language model evaluation, comparison, and deployment planning. All notebooks are production-ready and follow enterprise standards for code quality, documentation, and output generation.*