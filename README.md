# Vision Model Comparison Notebooks

A standalone package for comparing **Llama-3.2-Vision** and **InternVL3** models on two key tasks:
- **Visual Question Answering (VQA)**: Simple questions about images
- **Key-Value Extraction**: Structured field extraction from business documents

## 📋 What's Included

### Notebooks
- `notebooks/llama_VQA.ipynb` - Llama-3.2-Vision VQA example
- `notebooks/internvl3_VQA.ipynb` - InternVL3 VQA example  
- `notebooks/llama_keyvalue.ipynb` - Llama structured field extraction
- `notebooks/internvl3_keyvalue.ipynb` - InternVL3 structured field extraction

### Supporting Files
- `environment.yml` - Conda environment with all dependencies
- `unified_setup.sh` - Automated setup script
- `sample_data/synthetic_invoice_014.png` - Test image for all notebooks

## 🚀 Quick Start

### 1. Get the Notebooks
Choose one of these methods:

#### **Option A: Clone Branch Directly** (Recommended)
```bash
# Clone just the notebooks branch
git clone -b vision-notebooks --single-branch https://github.com/tmnestor/unified_vision_processor_minimal.git
cd unified_vision_processor_minimal/
```

#### **Option B: Download from GitHub** (Easiest)
1. Visit: https://github.com/tmnestor/unified_vision_processor_minimal/tree/vision-notebooks
2. Click "Code > Download ZIP" 
3. Extract and `cd` into the folder

### 2. Setup Environment
```bash
# Run the setup script (this creates conda environment and Jupyter kernel)
source unified_setup.sh

# Alternative manual setup:
conda env create -f environment.yml
conda activate vision_notebooks
python -m ipykernel install --user --name vision_notebooks --display-name "Python (Vision Notebooks)"
```

### 3. Configure Model Paths
**IMPORTANT**: Edit the `model_id` paths in each notebook to point to your models:

```python
# In Llama notebooks, change this line:
model_id = "/home/jovyan/nfs_share/models/Llama-3.2-11B-Vision-Instruct"
# To your actual path:
model_id = "/your/path/to/Llama-3.2-11B-Vision-Instruct"

# In InternVL3 notebooks, change this line:
model_id = "/home/jovyan/nfs_share/models/InternVL3-2B"  
# To your actual path:
model_id = "/your/path/to/InternVL3-2B"
```

### 4. Run Notebooks
```bash
# Launch Jupyter
jupyter notebook

# Select kernel: "Python (Vision Notebooks)"
# Open any notebook and run all cells
```

## 📊 Model Comparison Overview

| Model | Parameters | Memory Usage | Strengths |
|-------|------------|--------------|-----------|
| **Llama-3.2-Vision** | 11B | ~22GB VRAM | Detailed structured responses, built-in preprocessing |
| **InternVL3** | 2B | ~4GB VRAM | Memory efficient, fast inference, simple API |

## 🔍 Task Comparison

### Visual Question Answering (VQA)
**Question**: "How much did Jessica pay?"

**Llama Response Style**: Detailed, structured analysis with calculations
**InternVL3 Response Style**: Direct, concise answers

### Key-Value Extraction  
**Task**: Extract 25 structured fields from business documents

**Both models use identical prompts** but different processing approaches:
- **Llama**: Complex message structure with chat templates
- **InternVL3**: Simple `<image>\nPrompt` format

## 🛠️ Hardware Requirements

### Minimum Requirements
- **CPU**: Any modern multi-core processor
- **RAM**: 16GB system RAM
- **GPU**: Optional but recommended
  - Llama-3.2-Vision: 16GB+ VRAM (or use 8-bit quantization)
  - InternVL3: 4GB+ VRAM

### Recommended Setup
- **GPU**: NVIDIA V100 (16GB) or better
- **CUDA**: 11.x or 12.x compatible
- Both models support 8-bit quantization for memory optimization

## 📝 Notebook Details

### VQA Notebooks
- **Purpose**: Simple visual question answering
- **Input**: Image + natural language question
- **Output**: Natural language answer
- **Use Case**: Interactive image analysis, content understanding

### Key-Value Notebooks  
- **Purpose**: Structured data extraction from business documents
- **Input**: Document image + extraction template (25 fields)
- **Output**: Structured key-value pairs
- **Use Case**: Document digitization, automated data entry

## 🔧 Troubleshooting

### Common Issues

#### "Model not found" Error
```python
# Update the model_id path in the notebook:
model_id = "/your/actual/path/to/model"
```

#### CUDA Out of Memory
```python
# Enable 8-bit quantization (already configured):
# For Llama: torch_dtype=torch.bfloat16 with device_map="auto"
# For InternVL3: torch_dtype=torch.bfloat16 with quantization
```

#### Kernel Not Found
```bash
# Re-register the Jupyter kernel:
conda activate vision_notebooks
python -m ipykernel install --user --name vision_notebooks --display-name "Python (Vision Notebooks)"
```

### Environment Issues
```bash
# Recreate environment if needed:
conda env remove -n vision_notebooks
conda env create -f environment.yml
source unified_setup.sh
```

## 📈 Expected Results

### VQA Task
- **Llama**: Detailed explanations with step-by-step reasoning
- **InternVL3**: Direct answers, often more concise

### Key-Value Extraction
- **Both models**: 25-line structured output with field names
- **Success rate**: Typically 15-20 fields extracted accurately
- **Format**: `FIELD_NAME: value` or `FIELD_NAME: N/A`

## 🧪 Testing Your Setup

1. **Run VQA test**: Start with `llama_VQA.ipynb` or `internvl3_VQA.ipynb`
2. **Check output**: Should get natural language answer about the invoice
3. **Run extraction test**: Try `*_keyvalue.ipynb` notebooks  
4. **Verify fields**: Should extract ~15-20 fields from the sample invoice

## 📋 Dependencies

### Core ML Libraries
- `transformers==4.45.2` (fixed for compatibility)
- `torch>=2.0.0` + `torchvision`
- `accelerate` (device mapping)
- `bitsandbytes` (8-bit quantization)

### InternVL3 Specific
- `einops>=0.6.0` (tensor operations)
- `timm>=0.9.0` (vision encoder)

### Utilities
- `PIL` (image processing)
- `numpy`, `pandas` (data handling)
- `ipykernel` (Jupyter support)

## 💡 Tips for Success

1. **Start with VQA**: Simpler to test, faster feedback
2. **Use sample image**: `synthetic_invoice_014.png` works with all notebooks
3. **Monitor memory**: Use `nvidia-smi` to watch GPU usage
4. **Try both models**: Compare responses on the same input
5. **Experiment**: Modify questions and prompts to explore capabilities

## 🔗 Model Sources

- **Llama-3.2-Vision**: [Meta's Llama repository](https://huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct)
- **InternVL3**: [OpenGVLab InternVL3-2B](https://huggingface.co/OpenGVLab/InternVL3-2B)

## 📞 Support

If you encounter issues:
1. Check model paths are correct
2. Verify GPU memory availability  
3. Try CPU-only mode (slower but works)
4. Recreate conda environment if dependencies conflict

---

**Ready to explore vision-language models?** Start with the VQA notebooks and see how different models approach the same visual reasoning tasks!