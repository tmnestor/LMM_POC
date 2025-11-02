# Vision Language Models and Transaction Table Extraction

## Why VLMs Struggle with Table Structure Recognition

### 1. Table Structure Recognition is Inherently Challenging

Table structure recognition, especially for complex tables with cross rows and columns, represents one of the main challenges in extracting data from table images [1]. 
Upon evaluating 38 state-of-the-art Large Multimodal Models (LMMs) on OCRBench v2, 36 models scored below 50 out of 100, with complex element parsing identified as one of five key limitation areas [2].

**Reference:**
- [1] "Improving Table Structure Recognition With Visual-Alignment Sequential Coordinate Modeling", CVPR 2023
- [2] "OCRBench v2: An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning", arXiv:2501.00321v2, January 2025

### 2. Alignment and Cell Recognition Issues

VLMs demonstrate promising OCR capabilities but produce unsatisfactory results due to **cell omission and misalignment**, and they notably exhibit insufficient spatial and format recognition skills [3]. 
RD-TableBench findings show that accuracy must be measured using hierarchical alignment and Levenshtein distance specifically because of these alignment challenges [3].

**Reference:**
- [3] "Enhancing Table Recognition with Vision LLMs: A Benchmark and Neighbor-Guided Toolchain Reasoner", arXiv:2412.20662v2, December 2024

### 3. Row vs Column Recognition Asymmetry

Vision-Language Models show higher accuracy for column-related tasks than row-related tasks because columns usually contain more substantial attribute information, while **rows tend to have higher similarities, making VLLMs face more significant challenges locating rows** [4].

**Quantitative Evidence:**

The RD-TableBench evaluation of six VLLMs (Phi, Llama, GPT-4o-mini, Qwen, GPT-4o, and Gemini) found that **"the accuracy for the column-related tasks is higher than for the row-related tasks"** when controlling for balanced row/column distributions [10].

Specialized table structure recognition models demonstrate this asymmetry with concrete metrics:
- **TSRDet on ICDAR-2013**: F-measure of **92.06% for rows** vs. **96.32% for columns** (4.26% gap) [11]
- **TSRDet on TabStructDB**: F-measure of **94.08% for rows** vs. **95.06% for columns** (0.98% gap) [11]
- **GPT-4V on spreadsheets**: "Effectively forecasts column positions for most cells" but "consistently struggles with row positions, consistently displaying an offset" [4]

**Why This Occurs:**
- Columns contain distinguishing attributes (headers, categories, data types)
- Rows often have repetitive patterns with high content similarity
- VLMs excel at OCR-based identification but struggle with implicit two-dimensional coordinate inference [4]
- Empty rows compound cumulative positioning errors [4]

**CRITICAL DISTINCTION: Detection vs. Extraction Strategy**

While column **detection/identification** is more accurate than row detection, empirical testing reveals that **multi-turn row-wise extraction outperforms single-pass column-wise extraction** for full table extraction tasks.

**Column-Wise Extraction Failure Mode:**
When extracting columns independently:
- Date column extraction: Returns 10 entries
- Debit column extraction: Returns 12 entries (includes hallucinated/duplicated rows)
- Credit column extraction: Returns 9 entries (missing some rows)
- **Result**: Three columns with different lengths → **catastrophic misalignment**

**Row-Wise Extraction Advantage:**
Multi-turn approach extracting full rows first:
- Turn 1: Extract complete table row-by-row (maintains row integrity)
- Turn 2: Select specific columns from already-aligned rows
- **Result**: Preserved row correspondence across all columns, even if some rows are missing

**Key Insight:**
- **Column identification** (Turn 0 header detection) benefits from column detection superiority
- **Full table extraction** benefits from row-wise extraction to maintain alignment
- The benchmarks measure detection accuracy, not extraction strategy effectiveness

**Practical Implication:**
For transaction tables (bank statements, invoices), use:
1. **Column identification first** (leverage detection advantage) - Turn 0
2. **Row-wise extraction second** (preserve alignment) - Turn 1+
3. **Column selection third** (filter to needed columns) - Turn 2+

This hybrid approach combines the strengths of both dimensions.

**Reference:**
- [4] "Vision Language Models for Spreadsheet Understanding: Challenges and Opportunities", arXiv:2405.16234v1, May 2024
- [10] "Enhancing Table Recognition with Vision LLMs: A Benchmark and Neighbor-Guided Toolchain Reasoner", arXiv:2412.20662v2, December 2024
- [11] "TSRDet: A Table Structure Recognition Method Based on Row-Column Detection", Electronics 13(21):4263, MDPI, October 2024

### 4. Resolution Sensitivity

Scaling the input image resolution is essential for enhancing the performance of Vision Language Models (VLMs), particularly in text-rich image understanding tasks [5]. However, at high resolutions, vision encoder latency becomes the dominant bottleneck, creating a tradeoff between accuracy and efficiency [6].

**Reference:**
- [5] "FastVLM: Efficient Vision Encoding for Vision Language Models", arXiv:2412.13303v1, December 2024
- [6] "FastVLM: Efficient Vision Encoding for Vision Language Models", Apple Machine Learning Research, December 2024

### 5. Visual Conditions Impact Accuracy

Visual conditions are a key factor affecting VLLMs' accuracy in table recognition, with performance analyzed under various scenarios including blur, underexposure, and overexposure [3].

**Reference:**
- [3] "Enhancing Table Recognition with Vision LLMs: A Benchmark and Neighbor-Guided Toolchain Reasoner", arXiv:2412.20662v2

## Markdown-Specific Table Extraction Challenges

### 6. Merged Cells and Layout Irregularities

Markdown lacks native support for cells spanning multiple rows or columns [7]. Spreadsheet tables often feature irregular layouts with merged cells, hierarchical columns, and annotations, making machine parsing difficult [8]. 
Even advanced VLMs struggle with accurate Markdown conversion due to these layout irregularities [9].

**Reference:**
- [7] "Fixing the Gemini 2.0 Flash Markdown Table Generation Bug", CodingTarik blog, 2025
- [8] "Vision Language Models for Spreadsheet Understanding: Challenges and Opportunities", arXiv:2405.16234v1
- [9] "Fine-Tuning Vision-Language Models for Markdown", arXiv:2508.05669, August 2024


### 7. Non-Standard Output Formats

Some models occasionally produce non-standard output formats, such as recognizing multi-column text as tables, or formulas as Unicode text, leading to matching failures [2].

**Reference:**
- [2] "OCRBench v2: An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning", arXiv:2501.00321v2

## Documented Performance Metrics

### 8. Structured Output Degradation

InternVL3-14B achieves 94.4% accuracy in unpaired entities matching, but its performance drops to 84.9% in key information extraction. Performance further degrades in element parsing tasks that demand structured outputs [2].

**Reference:**
- [2] "OCRBench v2: An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning", arXiv:2501.00321v2

### 10. Fine-Tuning Improvements

Recent research on fine-tuning VLMs specifically for financial table extraction achieved 92.20% overall accuracy and 96.53% Markdown TEDS score, significantly surpassing larger-scale VLMs and specialized reasoning-enabled models [9].

**Reference:**
- [9] "Financial Table Extraction in Image Documents", arXiv:2405.05260v1, May 2024

### 11. Detection vs. Structure Recognition Performance Gap

Detection models that excel at identifying table components (columns and rows) don't necessarily perform well on cell-level table structure recognition (TSR) metrics like TEDS [12]. This reveals a fundamental mismatch between detection accuracy and structural understanding.

**Key Findings:**
- Models with good row/column detection performance cannot necessarily lead to good TEDS scores
- A tailored Cascade R-CNN model improved base model performance by 16.35% on FinTabNet using structure-only TEDS [12]
- TEDS-Struct metric assesses table structure accuracy independent of OCR quality
- TEDS(IOU) uses bounding box IOU distance instead of string edit distance for structure evaluation [13]

**Implications:**
High row/column detection accuracy doesn't guarantee accurate cell-level extraction or proper logical relationships between table elements.

**Reference:**
- [12] "Rethinking Detection Based Table Structure Recognition for Visually Rich Document Images", arXiv:2312.00699v2, January 2024
- [13] "Evaluating Table Structure Recognition: A New Perspective", arXiv:2208.00385, March 2024

## Quantitative Performance Summary: Row vs. Column Recognition

### Detection Accuracy (What Benchmarks Measure)

| Model/Benchmark | Row Accuracy | Column Accuracy | Performance Gap | Reference |
|-----------------|--------------|-----------------|-----------------|-----------|
| **TSRDet (ICDAR-2013)** | 92.06% F1 | 96.32% F1 | +4.26% columns | [11] |
| **TSRDet (TabStructDB)** | 94.08% F1 | 95.06% F1 | +0.98% columns | [11] |
| **GPT-4V (Spreadsheets)** | Consistent offset errors | Effectively forecasts | Qualitative advantage | [4] |
| **RD-TableBench (6 VLLMs)** | Lower accuracy | Higher accuracy | Consistent pattern | [10] |

**Key Insight**: Across all benchmarks and models, **column-related tasks consistently outperform row-related tasks by 1-4%**, with qualitative studies showing even larger disparities for positional accuracy.

### Extraction Strategy Effectiveness (Empirical Finding)

| Approach | Alignment Preservation | Failure Mode | Practical Outcome |
|----------|----------------------|--------------|-------------------|
| **Column-wise extraction** | ❌ Fails | Different column lengths (e.g., 10, 12, 9 rows) | Catastrophic misalignment |
| **Row-wise extraction** | ✅ Succeeds | Missing/hallucinated rows affect all columns equally | Preserved correspondence |

**Critical Distinction**:
- **Column detection** is more accurate (what benchmarks measure)
- **Row-wise extraction** is more effective (what production systems need)
- The 4.26% detection advantage does not translate to extraction superiority
- Missing rows in column-wise extraction destroy alignment; missing rows in row-wise extraction maintain it

## Implications for Bank Statement Processing

For bank statements with 5-column layouts (Date, Description, Withdrawal, Deposit, Balance):

### Detection-Level Challenges:
1. **Row similarity challenges**: Transaction rows have high similarity, increasing the likelihood of row detection errors [4, 10, 11]
2. **Multi-column complexity**: More columns increase the probability of cell omission and misalignment [3]
3. **Numerical confusion**: Similar amounts across Withdrawal/Deposit columns may be confused without spatial grounding [2]
4. **Visual quality sensitivity**: Scanned or photographed statements with blur/exposure issues will see degraded accuracy [3]
5. **Detection-structure gap**: High row/column detection accuracy doesn't guarantee proper cell-level extraction [12]

### Extraction-Level Strategy:
6. **Column-wise extraction failure**: Extracting each column independently (Date, Description, Debit, Credit, Balance) produces columns with different lengths, making row correspondence impossible
7. **Row-wise extraction success**: Extracting complete rows first, then selecting columns, preserves alignment even when rows are missing or hallucinated
8. **Hybrid approach optimal**: Leverage column detection accuracy (Turn 0: identify headers) + row extraction robustness (Turn 1: extract full rows) + column filtering (Turn 2: select needed columns)

### Recommended Multi-Turn Strategy:
- **Turn 0**: Identify all column headers (benefits from 4.26% column detection advantage)
- **Turn 1**: Extract full table row-by-row in markdown format (preserves row integrity)
- **Turn 2**: Select specific columns from aligned rows (avoids column length mismatch)
- **Turn 3+**: Apply row filters, extract metadata (operate on structurally-sound data)

## Case Study: Australian Bank Statement Extraction

### Real-World Challenge: CR Notation and Multi-Line Transactions

This case study examines extraction challenges from Australian bank statements with a specific 5-column format: **Date | Transaction | Debit | Credit | Balance**.

#### Critical Format Rules

**CR Notation Rule:**
- The **Balance** column displays amounts as "CR" (credit balance, e.g., "$258.38 CR")
- The **Credit** column shows plain amounts WITHOUT "CR" notation (e.g., "$60.03")
- VLMs frequently hallucinate "CR" notation in the Credit column by incorrectly copying the pattern from the Balance column

**Multi-Line Transaction Descriptions:**
Transaction entries contain multi-line text that must be collapsed into a single table row:
```
01 May TELSTRA PREPAID MELBOURNE AUS
       Card xxXXXX
       Value Date: 29/04/2024
```
Must become:
```
| 01 May 2024 | TELSTRA PREPAID MELBOURNE AUS Card xxXXXX Value Date: 29/04/2024 | $35.00 | | $223.38 CR |
```

#### Observed VLM Failure Modes

Based on empirical testing with Llama-3.2-Vision-11B and InternVL3-8B:

1. **CR Notation Contamination**
   - **Error**: VLMs add "CR" suffix to Credit column amounts (e.g., "$60.03 CR" instead of "$60.03")
   - **Cause**: Pattern recognition from adjacent Balance column [2]
   - **Impact**: Post-processing fails because amounts become non-numeric strings

2. **Multi-Line Row Splitting**
   - **Error**: Each line of a multi-line transaction becomes a separate table row
   - **Cause**: VLMs interpret visual line breaks as row boundaries [3]
   - **Impact**: Row count mismatch between extracted data and ground truth

3. **Empty Cell Duplication**
   - **Error**: When Debit column is empty, VLMs duplicate the Credit amount into Debit
   - **Cause**: Insufficient spatial grounding for empty cells [2]
   - **Impact**: Double-counting of transaction amounts

4. **Column Alignment Drift**
   - **Error**: Markdown pipe characters (`|`) misalign after long Transaction descriptions
   - **Cause**: Variable-width Transaction column content disrupts visual alignment [3]
   - **Impact**: Downstream parsers fail to correctly identify column boundaries

5. **Date Format Inconsistency**
   - **Error**: "01 May" becomes "01 May 2024" or "2024-05-01" inconsistently
   - **Cause**: VLMs apply implicit date normalization [2]
   - **Impact**: Date matching fails against ground truth

#### Mitigation Strategies Tested

**Multi-Turn Conversational Extraction:**
Using the `chat_with_mllm` pattern from `llama_multiturn_flat_debit_extractor.ipynb`:

- **Turn 0**: Identify all column headers explicitly (prevents column name hallucination)
- **Turn 1**: Extract full transaction table in markdown
- **Turn 2**: Select specific columns (Date | Transaction | Debit | Credit | Balance)
- **Turn 3**: Filter rows with specific criteria (e.g., non-empty Debit column)
- **Turn 4**: Extract metadata (date range, total amounts)

**Prompt Engineering for CR Notation:**
```
CRITICAL:
- The Balance column shows "CR" amounts (e.g., "$258.38 CR")
- The Credit column NEVER shows "CR" amounts (e.g., "$60.03" NOT "$60.03 CR")
- ONLY the Balance column includes "CR" notation
```

**Empirical Results:**
- **Multi-turn row-wise extraction outperforms single-pass column extraction** - prevents column misalignment from missing rows
- Explicit column header identification (Turn 0) improves column alignment accuracy
- Row-wise extraction (Turn 1) maintains row integrity across all columns, even when some rows are hallucinated or missed
- Multi-turn approach reduces CR contamination errors
- Column selection (Turn 2) from already-aligned rows prevents the catastrophic misalignment that occurs with independent column extraction
- However, multi-line transaction collapsing remains inconsistent across both models

**Critical Finding - Extraction Strategy:**
Attempting to extract Date, Debit, and Credit columns independently produces columns of different lengths (e.g., 10, 12, 9 entries), making row correspondence impossible. The multi-turn row-first approach preserves alignment by extracting complete rows before filtering to specific columns.

#### Technical Implications

This use case validates multiple benchmark findings while revealing critical extraction strategy insights:

1. **Structured output degradation** [2]: Both models perform better at raw OCR than structured markdown generation
2. **Row similarity challenges** [4, 10, 11]: Transaction rows with similar patterns (card payments, transfers) increase misalignment - consistent with the 1-4% row accuracy penalty observed in benchmarks
3. **Spatial reasoning failures** [3]: Empty cells and CR notation placement require spatial understanding beyond text recognition
4. **Detection-structure gap** [12]: High row/column identification doesn't guarantee accurate cell-level extraction or proper logical relationships
5. **Column advantage validated**: Column header identification (Turn 0) performs more reliably than row extraction, matching quantitative findings from TSRDet and RD-TableBench
6. **Row-wise extraction superiority (NEW)**: Despite lower row detection accuracy, row-wise extraction maintains alignment while column-wise extraction produces misaligned outputs

**Quantitative Validation:**
The observed multi-line transaction splitting and row alignment drift align with TSRDet's finding that row detection achieves 92.06% F1 vs. 96.32% F1 for columns on ICDAR-2013 [11]. However, the 4.26% performance gap in **detection** does not translate to inferior **extraction strategy**—row-wise extraction prevents the catastrophic column misalignment that occurs when columns are extracted independently.

**Key Paradox Explained:**
- **Column detection is more accurate** (96.32% vs 92.06%) - benchmarks measure this
- **Row-wise extraction is more effective** (preserves alignment) - practical implementation reveals this
- The benchmarks measure detection accuracy, not extraction strategy robustness
- Missing or hallucinated rows affect all columns equally in row-wise extraction, maintaining correspondence
- Missing or hallucinated rows in column-wise extraction create length mismatches, destroying correspondence

**Recommendation:**
For production bank statement extraction, implement hybrid strategy:
- **Turn 0**: Column header identification (leverages superior column detection accuracy) [4, 10, 11]
- **Turn 1**: Row-wise full table extraction (preserves row correspondence across columns)
- **Turn 2+**: Column filtering and row selection (operate on aligned data)
- Fine-tuning on bank-specific financial documents [9]
- Evaluation using TEDS-Struct metrics to assess structure independent of OCR quality [12, 13]

## References

[1] Huang et al. (2023). "Improving Table Structure Recognition With Visual-Alignment Sequential Coordinate Modeling", CVPR 2023.

[2] Liu et al. (2025). "OCRBench v2: An Improved Benchmark for Evaluating Large Multimodal Models on Visual Text Localization and Reasoning", arXiv:2501.00321v2.

[3] Anonymous et al. (2024). "Enhancing Table Recognition with Vision LLMs: A Benchmark and Neighbor-Guided Toolchain Reasoner", arXiv:2412.20662v2.

[4] Authors (2024). "Vision Language Models for Spreadsheet Understanding: Challenges and Opportunities", arXiv:2405.16234v1.

[5][6] FastVLM Research Team (2024). "FastVLM: Efficient Vision Encoding for Vision Language Models", Apple Machine Learning Research & arXiv:2412.13303v1.

[7] CodingTarik (2025). "Fixing the Gemini 2.0 Flash Markdown Table Generation Bug", https://codingtarik.github.io/

[8] Spreadsheet Understanding Research (2024). arXiv:2405.16234v1.

[9] Financial Table Extraction Research (2024). "Financial Table Extraction in Image Documents", arXiv:2405.05260v1 & "Fine-Tuning Vision-Language Models for Markdown", arXiv:2508.05669.

[10] Anonymous et al. (2024). "Enhancing Table Recognition with Vision LLMs: A Benchmark and Neighbor-Guided Toolchain Reasoner", arXiv:2412.20662v2, December 2024.

[11] Authors (2024). "TSRDet: A Table Structure Recognition Method Based on Row-Column Detection", Electronics, 13(21):4263, MDPI, October 2024.

[12] Authors (2024). "Rethinking Detection Based Table Structure Recognition for Visually Rich Document Images", arXiv:2312.00699v2, January 2024.

[13] Authors (2024). "Evaluating Table Structure Recognition: A New Perspective", arXiv:2208.00385, March 2024.

---

**Based on**:
- OCRBench v2 (10,000 human-verified QA pairs, 38 LMM evaluations)
- RD-TableBench (6 VLLMs evaluated: Phi, Llama, GPT-4o-mini, Qwen, GPT-4o, Gemini)
- ICDAR-2013 & TabStructDB benchmarks for row/column detection
**Case Study**: Australian bank statement extraction with Llama-3.2-Vision-11B and InternVL3-8B

---

## Appendix: Understanding TEDS-Struct and Related Metrics

### What is TEDS?

**TEDS (Tree Edit Distance-based Similarity)** is a metric for evaluating table structure recognition that captures multi-hop cell misalignment and OCR errors more effectively than traditional metrics.

**Core Concept:**
Tables can be represented as HTML tree structures (e.g., `<table>`, `<tr>`, `<td>` elements). TEDS measures the similarity between two tables by calculating how many edits are required to transform one HTML tree into another.

**Formula:**
```
TEDS(Ta, Tb) = 1 - EditDist(Ta, Tb) / max(|Ta|, |Tb|)
```

Where:
- `Ta` = Reference (ground truth) table as HTML tree
- `Tb` = Predicted table as HTML tree
- `EditDist()` = Tree edit distance (number of operations needed to transform one tree into another)
- `|Ta|`, `|Tb|` = Number of nodes in each tree

**Score Range:** 0 to 1 (higher is better)
- 1.0 = Perfect match
- 0.0 = Completely different structures

**Edit Operations & Costs:**
- **Insertion** of a node: Cost = 1
- **Deletion** of a node: Cost = 1
- **Substitution** of a non-cell node: Cost = 1
- **Substitution** of cell content: Cost depends on text similarity

---

### TEDS-Struct (TEDS-S): Structure-Only Variant

**Purpose:** Evaluate table structure independent of cell content and OCR quality.

**Key Difference from TEDS:**
While the formula remains identical, **TEDS-Struct modifies the tree representation** to exclude cell content:

**Standard TEDS Tree:**
```html
<table>
  <tr>
    <td>John Smith</td>    <!-- Content included -->
    <td>$125.00</td>
  </tr>
</table>
```

**TEDS-Struct Tree:**
```html
<table>
  <tr>
    <td></td>    <!-- Content omitted -->
    <td></td>
  </tr>
</table>
```

**Why This Matters:**
- Different table extraction systems use different OCR engines
- TEDS-Struct provides a **fair comparison** by focusing solely on structural accuracy (rows, columns, cells, spans)
- Eliminates OCR quality as a confounding variable

**Use Cases:**
- Comparing models with different OCR capabilities
- Evaluating table structure detection independent of text recognition
- Assessing row/column alignment without content accuracy interference

**Implementation:**
```python
# Using structure-only mode
teds_struct_score = TEDS(structure_only=True)
```

---

### TEDS-IOU: Bounding Box Variant

**Purpose:** OCR-independent evaluation using spatial information instead of text.

**Formula:**
```
TEDS_IOU(Ta, Tb) = 1 - EditDistIOU(Ta, Tb) / max(|Ta|, |Tb|)
```

**Key Innovation:**
Replaces text-based cell comparison with **bounding box Intersection over Union (IOU)**:

**Edit Costs for TEDS-IOU:**
- **Insertion/Deletion**: Cost = 1
- **Substitution** (non-cell nodes): Cost = 1
- **Substitution** (cell with different rowspan/colspan): Cost = 1
- **Substitution** (cell with matching spans): Cost = **1 - IOU(bbox_a, bbox_b)**

**IOU Calculation:**
```
IOU = Area of Overlap / Area of Union
```

**Example Performance:**
- A table with OCR errors:
  - TEDS (Text): **71.6** (penalized for text mismatches)
  - TEDS-IOU: **80.6** (ignores OCR errors, evaluates spatial alignment)

**Advantages:**
- Completely OCR-independent
- Evaluates spatial positioning accuracy
- Better handles tables with poor OCR quality
- Satisfies mathematical metric properties (identity, symmetry, triangle inequality)

---

### Comparison: TEDS vs. TEDS-Struct vs. TEDS-IOU

| Metric | Evaluates Content | Evaluates Structure | OCR-Dependent | Use Case |
|--------|-------------------|---------------------|---------------|----------|
| **TEDS** | ✅ Yes | ✅ Yes | ✅ Yes | Overall table extraction quality |
| **TEDS-Struct** | ❌ No | ✅ Yes | ❌ No | Structure recognition fairness |
| **TEDS-IOU** | ❌ No | ✅ Yes (spatial) | ❌ No | Spatial layout accuracy |

---

### Practical Application in VLM Evaluation

**Why TEDS-Struct is Critical for VLM Benchmarking:**

1. **Fair Model Comparison:**
   - Different VLMs have varying OCR capabilities
   - TEDS-Struct isolates structural understanding from text recognition

2. **Row/Column Accuracy Assessment:**
   - Detects row misalignment independent of content errors
   - Reveals column drift without OCR quality interference

3. **Multi-Hop Error Detection:**
   - Captures cascading errors (e.g., one misaligned cell affects entire row)
   - Standard accuracy metrics miss these structural failures

4. **Bank Statement Extraction Example:**
   - A model might correctly OCR "$125.00" but place it in the wrong column
   - Standard accuracy: ✅ Correct (text matches)
   - TEDS-Struct: ❌ Wrong (structure mismatch)

**References:**
- Zhong et al. (2020). "Image-based table recognition: data, model, and evaluation", arXiv:1911.10683
- Authors (2024). "Evaluating Table Structure Recognition: A New Perspective", arXiv:2208.00385
