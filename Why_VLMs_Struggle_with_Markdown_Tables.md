Why VLMs Struggle with Markdown Tables

OCRBench findings reveal:
1. Table structure recognition is one of the most challenging tasks for VLMs
2. Alignment issues: Models often struggle with column alignment in complex tables
3. Resolution sensitivity: Visual encoder resolution significantly impacts table OCR accuracy
4. Multi-column layouts: Flat tables with 5+ columns (Date, Description, Withdrawal, Deposit,
Balance) are particularly difficult

Common Markdown Table Failures

When VLMs fail at markdown tables, they typically:
- Merge columns incorrectly
- Miss alignment characters (|)
- Duplicate or skip rows
- Confuse similar amounts across columns (Withdrawal vs Deposit)
- Hallucinate table structure when unclear