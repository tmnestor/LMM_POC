# Transaction Linking Experiment

Three-stage receipt-to-bank-statement matching using InternVL3 vision-language model.

## Approach

Each receipt and bank statement image is processed independently through three inference stages:

1. **Stage 1** (receipt image only) - Extract store name, date, and total for each receipt
2. **Stage 2a** (bank statement image only) - Extract column headers to discover layout dynamically
3. **Stage 2b** (bank statement image only) - Match receipts to bank transactions using Stage 1 results as text context and Stage 2a column mappings

## Results

- **94.4% field accuracy** (17/18 fields correct across 3 receipt-statement pairs)
- 3/3 pairs processed successfully
- Handles single-receipt, dual-receipt, and multi-receipt pages

## Project Structure

```
staged_transaction_linking.ipynb        # Main notebook

common/                                 # Shared utilities
  evaluation_metrics.py                 # Ground truth loading, F1 scoring
  field_config.py                       # Field schema accessors
  field_definitions_loader.py           # YAML field definitions loader
  pipeline_config.py                    # PipelineConfig dataclass
  simple_prompt_loader.py               # Prompt YAML loader
  unified_bank_extractor.py             # Column detection and response parsing

models/                                 # Model loading and preprocessing
  internvl3_image_preprocessor.py       # InternVL3 image tiling and transforms
  registry.py                           # Model registry with lazy loading

config/                                 # Configuration files
  experiment_config.yml                 # Experiment settings (model, prompts, data)
  run_config.yml                        # Model paths and loading options
  bank_column_patterns.yaml             # Bank statement column header patterns
  field_definitions.yaml                # Document field schemas

prompts/
  staged_transaction_linking.yaml       # Three-stage prompt definitions

evaluation_data/                        # Test images and ground truth
  transaction_link_ground_truth.csv     # Ground truth for evaluation
  synthetic_*.png                       # Synthetic receipt/statement images
  bank_*.png                            # Real bank statement images

synthetic_receipt_generator/            # Tools to generate synthetic test data
  generate_receipt.py                   # Single receipt generator
  generate_multi_receipt_page.py        # Multi-receipt composite page generator
  generate_bank_statement.py            # Bank statement generator
```

## Running the Experiment

1. Configure `config/run_config.yml` with your model path
2. Open `staged_transaction_linking.ipynb`
3. Run all cells

### Requirements

- Python 3.12+
- PyTorch with CUDA
- transformers, Pillow, pandas, numpy, PyYAML, rich
- InternVL3.5-8B model weights
