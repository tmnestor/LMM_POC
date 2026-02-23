# InternVL3.5-8B  - Use Case Assessment

## Overview

A business area has identified four form types where AI solutions could assist with digitisation:

1. Certificate of residency
2. Overseas tax relief
3. Searching for lost super
4. Private rulings

Each of these forms involves manual tasks that could potentially be replaced or supplemented by AI, including:

- Checking for completeness, authorisation and validity
- Summarisation of content including attachments
- Data/content extraction to a digital format
- Consolidating information from core systems into a single user interface

This document assesses how InternVL3.5-8B, the vision-language model under evaluation in our POC, maps to these requirements.

## Form Types

These are all **structured government/financial forms** - exactly the document category our POC targets. Certificate of residency, overseas tax relief, lost super searches, and private rulings are all form-heavy, field-rich documents similar to the invoices, bank statements, and tax documents we already extract from.

## Task Mapping

| Their Task | Our Capability | Fit |
|---|---|---|
| **a. Completeness, authorisation & validity** | Detection pipeline classifies doc type and can flag missing/unexpected fields against `field_definitions.yaml` | Strong  - extend field validation logic |
| **b. Summarisation including attachments** | InternVL3.5-8B handles multi-page/multi-image input natively; we already do multi-turn for bank statements | Strong  - add summarisation prompts |
| **c. Data/content extraction to digital format** | **This is exactly what our POC does** - structured data extraction with high accuracy | Direct match |
| **d. Consolidating information from core systems** | Outside the model itself, but extracted structured data integrates easily into any UI/API | Architectural task, not model limitation |

## Honest Assessment

- **Tasks (b) and (c)** are directly supported by our current architecture with minimal adaptation  - new prompt YAML files for their specific form types, new field definitions, done.
- **Task (a)** is achievable by adding a validation layer that checks extracted fields against completeness rules (e.g., "signature field must not be empty").
- **Task (d)** is a systems integration concern, not a model concern  - but the structured JSON output we produce is designed for exactly this kind of downstream consumption.

## What We'd Need to Demonstrate

1. **New prompt templates** in `prompts/` for their 4 form types
2. **New field definitions** in `config/field_definitions.yaml` for each form
3. **Sample documents** from them to calibrate extraction accuracy
4. A **summarisation prompt** (new capability, but straightforward to add)

## Bottom Line

Yes, our model and pipeline can handle this. Tasks (b) and (c) are already proven. Tasks (a) and (d) require modest extensions to the pipeline, not the model itself.
