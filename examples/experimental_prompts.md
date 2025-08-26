# Experimental Prompt Testing Examples

This document shows how to use the experimental prompt tester for quick prompt engineering and A/B testing.

## Quick Start

### Basic Usage
```bash
# Test a simple prompt on Llama
python experimental_prompt_tester.py --model llama --prompt "Extract the total amount from this invoice" --image evaluation_data/synthetic_invoice_001.png

# Test on InternVL3 with auto-selected image
python experimental_prompt_tester.py --model internvl3 --prompt "What is the business name and total amount?"
```

### Compare with Current System
```bash
# Compare experimental prompt vs current schema-driven approach
python experimental_prompt_tester.py --model llama --prompt "List all monetary amounts you can see" --compare-baseline --image evaluation_data/synthetic_invoice_001.png
```

## Example Experimental Prompts

### Minimalist Approach
```
Extract: Business name, ABN, total amount. Format: Name|ABN|Total
```

### Ultra-Specific
```
You are analyzing an Australian business invoice. Extract exactly:
1. BUSINESS_ABN (11 digits)  
2. TOTAL_AMOUNT (final amount due with $)

Output format:
ABN: [value]
TOTAL: [value]
```

### Alternative CoT Approach
```
Look at this invoice image. Think step by step:
- First, identify the document type
- Then, find the business details section  
- Extract the ABN (11 digits)
- Find the totals section
- Extract final amount due

Provide: ABN and TOTAL_AMOUNT only.
```

## Batch Testing Multiple Prompts

Create a file `test_prompts.txt`:
```
Extract the business ABN and total amount from this document.
---
Find the 11-digit Australian Business Number and final payment amount.
---
What are the business registration number and total amount due?
---
Business ABN: [find 11 digits]
Total: [find final amount with $]
```

Run batch test:
```bash
python experimental_prompt_tester.py --model internvl3 --prompt-file test_prompts.txt --image evaluation_data/synthetic_invoice_001.png
```

## A/B Testing Workflow

1. **Establish Baseline**:
```bash
python experimental_prompt_tester.py --model llama --prompt "Standard extraction test" --compare-baseline --image test_image.png
```

2. **Test Variations**:
```bash
# Test shorter prompts
python experimental_prompt_tester.py --model llama --prompt "Extract: ABN, Total" --image test_image.png

# Test different phrasing  
python experimental_prompt_tester.py --model llama --prompt "Find business number and amount due" --image test_image.png
```

3. **Measure Results**: Compare accuracy, processing time, and response quality

## Common Experimental Patterns

### Minimalist Extraction
- **Goal**: Test if shorter prompts work better
- **Pattern**: "Extract: [field1], [field2], [field3]"
- **Benefit**: Reduces cognitive load, faster processing

### Format-First Prompts
- **Goal**: Force specific output format upfront
- **Pattern**: "Output format: FIELD: value\nNow extract from this image:"
- **Benefit**: Reduces parsing errors

### Question-Based Prompts  
- **Goal**: Natural language approach
- **Pattern**: "What is the business ABN? What is the total amount?"
- **Benefit**: More conversational, may improve understanding

### Visual-First Prompts
- **Goal**: Emphasize visual analysis
- **Pattern**: "Look at this image carefully. Focus on numbers and dollar signs. Extract..."
- **Benefit**: May improve OCR accuracy

## Tips for Effective Experimentation

1. **Keep Variables Minimal**: Change one thing at a time
2. **Test on Multiple Images**: Don't optimize for single examples  
3. **Measure Objectively**: Track accuracy, not just subjective "better"
4. **Document Everything**: Save prompts that work and ones that don't
5. **Consider Token Budget**: Shorter isn't always better, but bloat is bad

## Integration with Main Pipeline

Once you find an effective experimental prompt:

1. **Analyze what works**: Identify key patterns or phrases
2. **Integrate gradually**: Add successful elements to schema templates
3. **A/B test in production**: Compare with current baseline
4. **Document lessons learned**: Update schema methodology

## Performance Tracking

The experimental tester tracks:
- **Processing Time**: How fast each prompt runs
- **Response Quality**: Raw model output for manual review
- **Comparison Data**: Side-by-side with current system

Use this data to make informed decisions about prompt improvements.

## Markdown Conversion Prompts

### Simple & Direct
```
Convert this image to markdown format. Include all text, structure, and formatting.
```

### Structured Approach (Recommended)
```
Analyze this image and convert it to markdown. Include:
- Headers (use # ## ### etc.)
- Lists (- or 1. 2. 3.)
- Bold/italic text (**bold** *italic*)
- Tables if present
- Code blocks if any

Output clean markdown only.
```

### Document-Aware
```
You are a markdown converter. Look at this document image and recreate it as markdown:

1. Identify the document structure (headers, paragraphs, lists)
2. Preserve formatting (bold, italic, emphasis)
3. Convert tables to markdown table format
4. Output only the markdown, no explanations

Begin:
```

### OCR-First Approach
```
Extract all text from this image and format it as proper markdown. Maintain the visual hierarchy:
- Main titles as # headers
- Subtitles as ## or ### 
- Bullet points as - lists
- Numbered items as 1. 2. 3.
- Preserve line breaks and spacing

Markdown output:
```

### Quick Test Commands for Markdown
```bash
# Test the simple approach
python quick_prompt_test.py
# (Edit EXPERIMENTAL_PROMPT to: "Convert this image to markdown format.")

# Or use the full tester
python experimental_prompt_tester.py --model internvl3 --prompt "Convert this image to markdown format. Include all text, structure, and formatting." --image your_document.png
```

### Specialized Markdown Variants

#### For Code Documentation
```
Convert this code documentation image to markdown. Include:
- Code blocks with ```language
- Function names as `code spans`  
- Headers for sections
- Proper indentation for nested lists
```

#### For Business Documents
```
Extract this business document as markdown, preserving:
- Company headers as # titles
- Section headings as ## 
- Data lists as markdown tables
- Important amounts as **bold**
```

#### For Academic Papers
```
Convert this academic document to markdown format:
- Title as # header
- Authors and affiliations  
- Abstract as block quote (>)
- Sections as ## headers
- References as numbered lists
```

### Markdown Conversion Tips

1. **Test on Different Document Types**: Try on invoices, code docs, academic papers
2. **Compare Models**: Test same prompt on both Llama and InternVL3
3. **Iterate Based on Results**: If structure is missed, emphasize it more
4. **Check Special Characters**: Some models handle markdown syntax better than others
5. **Use Structured Approach**: The structured approach prompt above is usually the best starting point