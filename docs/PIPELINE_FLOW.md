# Document Processing Pipeline Flow

## How a Document Flows Through the System

When a business document image enters the pipeline as a **WRE Document** (Work-Related Expense), it follows this systematic flow:

### Step 1: Document Type Detection
The image first encounters the **Document Type Detection Prompt** which analyzes the visual and textual elements to classify the document. The system asks: "What type of business document is this?"

### Step 2: Classification Decision
At the **Document Type** decision point (diamond), the system determines whether the document is:
- An **Invoice** (from a supplier or vendor)
- A **Receipt** (from a retail transaction)
- A **Bank Statement** (showing financial transactions)

### Step 3: Document-Specific Prompt Selection
Based on the classification, the system selects the appropriate extraction prompt:
- **Invoice** → Invoice Prompt (optimized for supplier details, line items, payment terms)
- **Receipt** → Receipt Prompt (optimized for retail transactions, item lists, totals)
- **Bank Statement** → Bank Statement Prompt (optimized for transaction lists, dates, balances)

### Step 4: Vision-Language Model Processing
The selected prompt and document image are processed together by the **Vision-Language Model** (either Llama-3.2-Vision or InternVL3). The model reads and interprets the document content according to the specific prompt instructions.

### Step 5: Response Processing & Normalization
The raw model output undergoes **Response Processing & Value Normalization** where:
- Extracted values are standardized (dates, currency formats)
- Fields are validated against expected patterns
- Data is cleaned and structured

### Step 6: Structured Output
The final result is **Structured Extraction** containing up to 25 standardized fields (ABN, TOTAL, dates, line items, etc.) ready for downstream processing or storage.

## Key Design Principle
Each document type receives specialized treatment through its own prompt, ensuring optimal extraction accuracy rather than using a generic one-size-fits-all approach. This document-aware pipeline adapts to the specific characteristics and layout patterns of each document type.