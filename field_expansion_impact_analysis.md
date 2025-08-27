# Field Expansion Impact Analysis

## Executive Summary

This document outlines the comprehensive field expansion requirements mandated by management and analyzes the impact on our existing document extraction pipelines.

## Boss's Complete Field Requirements

### Invoice Fields (19 core + 6 line-item fields = 25 total)

#### Invoice Details
1. **Type of document: invoice/receipt/bill** (DOCUMENT_TYPE) ✅ Existing
2. **Invoice number** (INVOICE_NUMBER) ✅ Existing
3. **Date issued** (INVOICE_DATE) ✅ Existing
4. **Supplier name** (SUPPLIER_NAME) ✅ Existing
5. **Supplier ABN** (BUSINESS_ABN) ✅ Existing
6. **Supplier address** (BUSINESS_ADDRESS) ✅ Existing
7. **Supplier email** (SUPPLIER_EMAIL) ❌ **NEW FIELD**
8. **Supplier phone number** (BUSINESS_PHONE) ✅ Existing
9. **Payer name** (PAYER_NAME) ✅ Existing
10. **Payer address** (PAYER_ADDRESS) ✅ Existing
11. **Payer email** (PAYER_EMAIL) ✅ Existing
12. **Payer phone number** (PAYER_PHONE) ✅ Existing
13. **Total discount amount** (TOTAL_DISCOUNT_AMOUNT) ❌ **NEW FIELD**
14. **Total amount paid** (TOTAL_AMOUNT_PAID) ❌ **NEW FIELD**
15. **Balance of payment** (BALANCE_OF_PAYMENT) ❌ **NEW FIELD**
16. **Total amount payable** (TOTAL_AMOUNT_PAYABLE) ❌ **NEW FIELD**
17. **Total GST amount** (GST_AMOUNT) ✅ Existing
18. **Due date** (DUE_DATE) ✅ Existing
19. **Is GST included** (IS_GST_INCLUDED) ❌ **NEW FIELD**

#### Line-Item Details
1. **Line-item quantity** (LINE_ITEM_QUANTITIES) ✅ Existing
2. **Line-item unit price** (LINE_ITEM_PRICES) ✅ Existing
3. **Line-item description** (LINE_ITEM_DESCRIPTIONS) ✅ Existing
4. **Line-item total price** (LINE_ITEM_TOTAL_PRICES) ❌ **NEW FIELD**
5. **Line-item GST amount** (LINE_ITEM_GST_AMOUNTS) ❌ **NEW FIELD**
6. **Line-item discount amount** (LINE_ITEM_DISCOUNT_AMOUNTS) ❌ **NEW FIELD**

### Bank Statement Fields

#### Statement Details
1. **Type of document: bank statement/credit card statement** (DOCUMENT_TYPE) ✅ Existing
2. **Statement period** (STATEMENT_DATE_RANGE) ✅ Existing
3. **Opening balance** (ACCOUNT_OPENING_BALANCE) ✅ Existing
4. **Closing balance** (ACCOUNT_CLOSING_BALANCE) ✅ Existing
5. **Total amount received** (TOTAL_CREDITS) ✅ Existing
6. **Total amount paid** (TOTAL_DEBITS) ✅ Existing
7. **Due date: for credit card statement** (CREDIT_CARD_DUE_DATE) ❌ **NEW FIELD**

#### Account Details
8. **Account name** (BANK_ACCOUNT_HOLDER) ✅ Existing
9. **BSB : some credit card statement doesn't have BSB** (BANK_BSB_NUMBER) ✅ Existing (needs logic update)
10. **Account number** (BANK_ACCOUNT_NUMBER) ✅ Existing

#### Transaction Details (Currently aggregated, needs individual extraction)
1. **Date of transaction** (TRANSACTION_DATES) ❌ **NEW FIELD**
2. **Transaction description** (TRANSACTION_DESCRIPTIONS) ⚠️ Partial (currently aggregated)
3. **Transaction amount paid: 0 if it is a credit transaction** (TRANSACTION_AMOUNTS_PAID) ❌ **NEW FIELD**
4. **Transaction amount received: 0 if it is a debit transaction** (TRANSACTION_AMOUNTS_RECEIVED) ❌ **NEW FIELD**
5. **Transaction balance** (TRANSACTION_BALANCES) ❌ **NEW FIELD**

## Impact Analysis

### 1. Schema Updates Required

#### Field Schema v3 → v4 Migration
- **New Fields to Add**: 15 fields
- **Fields to Modify**: 2 fields (transaction descriptions, BSB logic)
- **Backward Compatibility**: Can be maintained with default NOT_FOUND values

#### New Field Categories
1. **Payment Status Fields** (4 new fields)
   - BALANCE_OF_PAYMENT
   - TOTAL_AMOUNT_PAYABLE
   - TOTAL_AMOUNT_PAID
   - TOTAL_DISCOUNT_AMOUNT

2. **Line-Item Calculations** (3 new fields)
   - LINE_ITEM_TOTAL_PRICES
   - LINE_ITEM_GST_AMOUNTS
   - LINE_ITEM_DISCOUNT_AMOUNTS

3. **Transaction-Level Details** (5 new fields)
   - TRANSACTION_DATES
   - TRANSACTION_AMOUNTS_PAID
   - TRANSACTION_AMOUNTS_RECEIVED
   - TRANSACTION_BALANCES
   - CREDIT_CARD_DUE_DATE

4. **Metadata Fields** (3 new fields)
   - SUPPLIER_EMAIL
   - IS_GST_INCLUDED
   - Credit card specific logic for BSB

### 2. Model Prompt Engineering Impact

#### Current State
- Invoice prompts: 20 fields extracted
- Bank statement prompts: 12 fields extracted
- Receipt prompts: 19 fields extracted

#### Required Changes
- **Invoice prompts**: Expand from 20 → 25 fields
- **Bank statement prompts**: Expand from 12 → 17 fields
- **Transaction extraction**: Change from aggregated to individual line items

#### Prompt Length
- Current average prompt: ~500 tokens
- Projected new prompt: ~750 tokens

### 3. Processing Performance Impact

#### Extraction Time
- **Expected change**: Additional fields to extract
- **Approach**: Field grouping and parallel extraction

#### Memory Usage
- **Transaction-level extraction**: Individual transaction storage required
- **Line-item calculations**: Computed fields

#### Accuracy Considerations
- Additional fields to validate
- Calculated fields (GST per line item)
- Transaction-level matching for statements

### 4. Evaluation Framework Updates

#### Ground Truth CSV Structure
Current columns: 34
Required columns: 49 (+15 new fields)

#### Evaluation Metrics
- Need new evaluation logic for:
  - Calculated fields (line totals = quantity × price)
  - Conditional fields (BSB for credit cards)
  - Transaction-level matching algorithms

#### Test Dataset
- All 42 existing test documents need re-annotation
- Estimated effort: 2-3 days for complete re-annotation

### 5. Database/Storage Impact

#### CSV Storage
- Row width: 34 columns → 49 columns
- File sizes will increase proportionally
- Database storage option for transaction-level data

#### Field Type Definitions
New field types needed:
- `boolean` type for IS_GST_INCLUDED
- `calculated` type for line-item totals
- `transaction_list` type for structured transaction data

### 6. Implementation Phases

#### Phase 1: Quick Wins (1-2 days)
- Add SUPPLIER_EMAIL
- Add IS_GST_INCLUDED
- Add TOTAL_DISCOUNT_AMOUNT

#### Phase 2: Invoice Enhancements (3-4 days)
- Payment status fields (BALANCE_OF_PAYMENT, TOTAL_AMOUNT_PAYABLE, etc.)
- Line-item calculations (totals, GST, discounts)

#### Phase 3: Bank Statement Overhaul (1 week)
- Transaction-level extraction
- Individual transaction parsing
- Credit card specific logic

#### Phase 4: Testing & Validation (1 week)
- Update all test data
- Validate accuracy on new fields
- Performance optimization

## Risk Assessment

### High Priority Items
1. **Transaction-level extraction** - Individual transaction parsing required
2. **Line-item calculations** - Dependent on quantity/price extraction
3. **Context length** - Additional fields in prompts

### Medium Priority Items
1. **Processing time** - Additional fields to process
2. **Ground truth maintenance** - Manual annotation effort
3. **Field validation** - Additional validation rules

### Low Priority Items
1. **Simple field additions** - EMAIL, IS_GST_INCLUDED
2. **Calculated fields** - Can be derived post-extraction
3. **Schema versioning** - Migration path available

## Recommendations

1. **Implementation Approach**
   - Start with simple field additions (Phase 1)
   - Transaction parsing in Phase 3
   - Review field requirements with stakeholders

2. **Technical Approach**
   - Field caching for calculated values
   - Parallel extraction for independent field groups
   - Two-pass extraction option (core fields, then details)

3. **Quality Assurance**
   - Testing on each phase
   - Monitor accuracy metrics
   - Rollback plan for each phase

4. **Documentation**
   - Implementation progress tracking
   - Field mapping documentation
   - User guides for new fields

## Appendix: Field Mapping Table

| Boss's Field Name | Our Implementation Name | Status | Complexity |
|------------------|------------------------|---------|------------|
| Type of document | DOCUMENT_TYPE | ✅ Existing | Low |
| Invoice number | INVOICE_NUMBER | ✅ Existing | Low |
| Date issued | INVOICE_DATE | ✅ Existing | Low |
| Supplier email | SUPPLIER_EMAIL | ❌ New | Low |
| Balance of payment | BALANCE_OF_PAYMENT | ❌ New | Medium |
| Total amount payable | TOTAL_AMOUNT_PAYABLE | ❌ New | Medium |
| Line-item total price | LINE_ITEM_TOTAL_PRICES | ❌ New | High |
| Line-item GST amount | LINE_ITEM_GST_AMOUNTS | ❌ New | High |
| Transaction amount paid | TRANSACTION_AMOUNTS_PAID | ❌ New | High |
| Transaction balance | TRANSACTION_BALANCES | ❌ New | High |

---

**Document Version**: 1.0  
**Date**: 2024-08-27  
**Author**: System Architecture Team  
**Status**: DRAFT - Pending Management Review