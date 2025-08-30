#!/usr/bin/env python3
"""
Document-Aware Grouped Extraction - The Ultimate Hybrid Solution

Combines the 90.6% accuracy grouped extraction strategy with intelligent
document-aware field filtering. Best of both worlds!

Key Innovation:
- Document type detection → Filter relevant groups → Grouped extraction
- Not single-pass overwhelm, but smart group selection per document type
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple

# Field group definitions from the working 90.6% accuracy approach
DOCUMENT_AWARE_FIELD_GROUPS = {
    "regulatory_financial": {
        "fields": ["BUSINESS_ABN", "TOTAL_AMOUNT", "ACCOUNT_OPENING_BALANCE", 
                   "ACCOUNT_CLOSING_BALANCE", "SUBTOTAL_AMOUNT", "GST_AMOUNT"],
        "expertise_frame": "Extract business ID and financial amounts.",
        "cognitive_context": "BUSINESS_ABN is 11 digits. TOTAL_AMOUNT is final amount due. GST_AMOUNT is tax. SUBTOTAL_AMOUNT is pre-tax amount.",
        "focus_instruction": "Find ABN (11 digits) and all dollar amounts. Check decimal places carefully."
    },
    
    "entity_contacts": {
        "fields": ["SUPPLIER_NAME", "BUSINESS_ADDRESS", "BUSINESS_PHONE", "SUPPLIER_WEBSITE",
                   "PAYER_NAME", "PAYER_ADDRESS", "PAYER_PHONE", "PAYER_EMAIL"],
        "expertise_frame": "Extract contact information for supplier and customer.",
        "cognitive_context": "SUPPLIER_NAME, BUSINESS_ADDRESS, BUSINESS_PHONE are supplier details. PAYER_NAME, PAYER_ADDRESS, PAYER_PHONE, PAYER_EMAIL are customer details.",
        "focus_instruction": "Extract all contact details. Australian postcodes are 4 digits. Phone numbers are 10 digits with area code."
    },
    
    "transaction_details": {
        "fields": ["LINE_ITEM_DESCRIPTIONS", "LINE_ITEM_QUANTITIES", "LINE_ITEM_PRICES"],
        "expertise_frame": "Extract ALL line items - scan the ENTIRE document for every item.",
        "cognitive_context": "CRITICAL: Look for ALL line items, even duplicates (e.g., Car Wash appearing twice). DESCRIPTIONS: Every product/service name. QUANTITIES: Every numeric quantity. PRICES: Every unit price with $ symbol. Count carefully - there may be 4, 5, 6+ items.",
        "focus_instruction": "SCAN ENTIRE DOCUMENT for line items. Extract EVERY SINGLE item including duplicates. Use PIPE-SEPARATED format. Count items carefully - don't stop at 4 items, look for more. Match quantities and prices to each description in exact order."
    },
    
    "temporal_data": {
        "fields": ["INVOICE_DATE", "DUE_DATE", "STATEMENT_DATE_RANGE"],
        "expertise_frame": "Extract document dates and periods.",
        "cognitive_context": "INVOICE_DATE is issue date. DUE_DATE is payment due. STATEMENT_DATE_RANGE is for bank statements only. For receipts, look for transaction date or purchase date.",
        "focus_instruction": "Find all dates. Convert to consistent DD/MM/YYYY format where possible. For receipts, focus on transaction/purchase dates."
    },
    
    "banking_payment": {
        "fields": ["BANK_NAME", "BANK_BSB_NUMBER", "BANK_ACCOUNT_NUMBER", "BANK_ACCOUNT_HOLDER"],
        "expertise_frame": "Extract banking information.",
        "cognitive_context": "BANK_NAME is financial institution. BSB_NUMBER is 6 digits. BANK_ACCOUNT_NUMBER varies. ACCOUNT_HOLDER is account name. Typically on bank statements only.",
        "focus_instruction": "Extract banking details if present. BSB is 6 digits, different from 11-digit ABN."
    },
    
    "document_metadata": {
        "fields": ["DOCUMENT_TYPE", "RECEIPT_NUMBER", "TRANSACTION_DATE", "PAYMENT_METHOD", "STORE_LOCATION", "ACCOUNT_OPENING_BALANCE", "ACCOUNT_CLOSING_BALANCE"],
        "expertise_frame": "Extract document identifiers and location information.",
        "cognitive_context": "DOCUMENT_TYPE: invoice, receipt, or statement. RECEIPT_NUMBER: transaction/reference number like R789121. TRANSACTION_DATE: purchase/transaction date. PAYMENT_METHOD: Look for AMEX, Visa, Mastercard, Cash, EFTPOS. STORE_LOCATION: Store location/address (may be same as business address).",
        "focus_instruction": "Extract document type, receipt numbers (R789121 format), payment methods (AMEX, etc), transaction dates, and store locations. Look for location info that may include city and postcode."
    }
}

# Document type to relevant groups mapping - FIXED for better field coverage
DOCUMENT_TYPE_GROUPS = {
    "invoice": ["regulatory_financial", "entity_contacts", "transaction_details", "temporal_data", "document_metadata"],
    "receipt": ["regulatory_financial", "entity_contacts", "transaction_details", "temporal_data", "document_metadata"],  # FIXED: Include document_metadata for DOCUMENT_TYPE
    "statement": ["regulatory_financial", "banking_payment", "temporal_data", "document_metadata"],
    "tax_invoice": ["regulatory_financial", "entity_contacts", "transaction_details", "temporal_data", "document_metadata"],
    "default": ["regulatory_financial", "entity_contacts", "temporal_data", "document_metadata"]  # FIXED: Always include document_metadata
}


class DocumentAwareGroupedExtraction:
    """
    The Ultimate Hybrid: Document awareness + 90.6% accuracy grouped extraction.
    
    This class combines:
    1. Document type detection (from main branch)
    2. Smart group filtering based on document type  
    3. Proven grouped extraction strategy (from working branch)
    
    Result: Better accuracy + fewer unnecessary model calls
    """
    
    def __init__(self, debug: bool = False):
        """Initialize the hybrid extraction system."""
        self.debug = debug
        self.stats = {
            "total_groups_processed": 0,
            "successful_groups": 0,
            "failed_groups": 0,
            "document_type_detections": 0
        }
        
        if self.debug:
            print("🎯 Document-Aware Grouped Extraction initialized")
            print(f"   Available groups: {len(DOCUMENT_AWARE_FIELD_GROUPS)}")
            print(f"   Supported document types: {list(DOCUMENT_TYPE_GROUPS.keys())}")
    
    def extract_with_document_awareness(
        self, 
        image_path: str, 
        model_extractor_func,
        all_fields: List[str]
    ) -> Tuple[Dict[str, str], Dict]:
        """
        Main extraction method that combines document awareness with grouped extraction.
        
        Args:
            image_path: Path to image file
            model_extractor_func: Function that can extract with custom prompts
            all_fields: Complete field list for fallback
            
        Returns:
            Tuple of (extracted_data, metadata)
        """
        start_time = time.time()
        
        try:
            if self.debug:
                print(f"🚀 Starting document-aware grouped extraction for: {Path(image_path).name}")
            
            # Step 1: Document type detection (single model call)
            doc_type = self._detect_document_type(image_path, model_extractor_func)
            self.stats["document_type_detections"] += 1
            
            if self.debug:
                print(f"📋 Document classified as: {doc_type}")
            
            # Step 2: Filter groups by document type (intelligence layer)
            relevant_groups = self._get_relevant_groups_for_document_type(doc_type)
            
            if self.debug:
                print(f"🎯 Selected {len(relevant_groups)} relevant groups: {relevant_groups}")
                skipped_groups = set(DOCUMENT_AWARE_FIELD_GROUPS.keys()) - set(relevant_groups)
                if skipped_groups:
                    print(f"⏭️ Skipped irrelevant groups: {list(skipped_groups)}")
            
            # Step 3: Run grouped extraction on filtered groups (proven strategy)
            extracted_data, group_metadata = self._run_filtered_grouped_extraction(
                image_path, relevant_groups, model_extractor_func, all_fields
            )
            
            # Step 4: Compile metadata
            total_time = time.time() - start_time
            metadata = {
                "document_type": doc_type,
                "relevant_groups": relevant_groups,
                "total_groups_called": len(relevant_groups),
                "total_groups_available": len(DOCUMENT_AWARE_FIELD_GROUPS),
                "groups_skipped": len(DOCUMENT_AWARE_FIELD_GROUPS) - len(relevant_groups),
                "processing_time": total_time,
                "group_details": group_metadata,
                "extraction_strategy": "document_aware_grouped"
            }
            
            if self.debug:
                found_fields = [k for k, v in extracted_data.items() if v != "NOT_FOUND"]
                print("🎉 Document-aware grouped extraction completed!")
                print(f"   📊 Found {len(found_fields)}/{len(all_fields)} fields")
                print(f"   ⏱️ Total time: {total_time:.2f}s")
                print(f"   🎯 Groups called: {len(relevant_groups)}/{len(DOCUMENT_AWARE_FIELD_GROUPS)}")
                print(f"   📈 Efficiency: {100 * len(relevant_groups) / len(DOCUMENT_AWARE_FIELD_GROUPS):.1f}% group usage")
            
            return extracted_data, metadata
            
        except Exception as e:
            if self.debug:
                print(f"❌ Document-aware grouped extraction failed: {e}")
                import traceback
                traceback.print_exc()
            
            # Fallback to basic field initialization
            fallback_data = {field: "NOT_FOUND" for field in all_fields}
            fallback_metadata = {
                "document_type": "unknown",
                "relevant_groups": [],
                "error": str(e),
                "extraction_strategy": "document_aware_grouped_fallback"
            }
            
            return fallback_data, fallback_metadata
    
    def _detect_document_type(self, image_path: str, model_extractor_func) -> str:
        """
        Detect document type using a focused single-field prompt.
        
        This is fast and accurate - just classify the document type.
        """
        if self.debug:
            print("🔍 Detecting document type...")
        
        # Simple, focused document type detection prompt
        doc_type_prompt = """Identify the document type from this business document image.

Look for key indicators:
- Invoice: "INVOICE", "TAX INVOICE", invoice number, due date
- Receipt: "RECEIPT", transaction receipt, purchase receipt  
- Statement: "STATEMENT", bank statement, account statement, transaction history

OUTPUT FORMAT:
DOCUMENT_TYPE: [document type as written on document]

Instructions:
- Extract the document type EXACTLY as written (e.g., "TAX INVOICE", "INVOICE", "RECEIPT", "STATEMENT")
- If it says "TAX INVOICE" - output "TAX INVOICE" 
- If it says "INVOICE" - output "INVOICE"
- Look for the actual document header text
- Base decision on document headers and layout"""

        try:
            # Single model call for document type detection
            response = model_extractor_func(
                image_path, 
                doc_type_prompt,
                max_new_tokens=100  # Slightly increased for safety
            )
            
            # Extract document type from response
            doc_type = self._parse_document_type_response(response)
            
            if self.debug:
                print(f"📋 Document type detection response: '{response.strip()}'")
                print(f"📋 Parsed document type: '{doc_type}'")
            
            return doc_type
            
        except Exception as e:
            if self.debug:
                print(f"⚠️ Document type detection failed: {e}, defaulting to 'invoice'")
            return "invoice"  # Safe default
    
    def _parse_document_type_response(self, response: str) -> str:
        """Parse document type from model response."""
        if not response:
            return "invoice"
        
        response_lower = response.lower().strip()
        
        # Look for document type indicators in order of confidence
        if "receipt" in response_lower:
            return "receipt"
        elif "statement" in response_lower:
            return "statement"  
        elif "invoice" in response_lower or "tax invoice" in response_lower:
            return "invoice"
        else:
            # Fallback analysis
            if "transaction" in response_lower or "purchase" in response_lower:
                return "receipt"
            elif "account" in response_lower or "balance" in response_lower:
                return "statement"
            else:
                return "invoice"  # Safe default
    
    def _get_relevant_groups_for_document_type(self, doc_type: str) -> List[str]:
        """Get the relevant field groups for a specific document type."""
        # Normalize document type
        doc_type_normalized = doc_type.lower().replace(" ", "_").replace("-", "_")
        
        # Get relevant groups, fallback to default if not found
        relevant_groups = DOCUMENT_TYPE_GROUPS.get(doc_type_normalized, DOCUMENT_TYPE_GROUPS["default"])
        
        if self.debug:
            all_groups = list(DOCUMENT_AWARE_FIELD_GROUPS.keys())
            skipped_groups = [g for g in all_groups if g not in relevant_groups]
            print(f"📊 Document type '{doc_type}' → {len(relevant_groups)}/{len(all_groups)} groups")
            if skipped_groups:
                print(f"⏭️ Skipping irrelevant groups: {skipped_groups}")
        
        return relevant_groups
    
    def _run_filtered_grouped_extraction(
        self, 
        image_path: str, 
        relevant_groups: List[str], 
        model_extractor_func,
        all_fields: List[str]
    ) -> Tuple[Dict[str, str], Dict]:
        """
        Run grouped extraction on only the relevant groups for this document type.
        
        This is the core of the hybrid approach - using the proven 90.6% accuracy
        grouped strategy but only on relevant fields.
        """
        if self.debug:
            print(f"🚀 Starting grouped extraction with {len(relevant_groups)} groups")
        
        # Initialize results with all fields as NOT_FOUND
        extracted_data = {field: "NOT_FOUND" for field in all_fields}
        group_results = {}
        
        # Process each relevant group
        for group_name in relevant_groups:
            if group_name not in DOCUMENT_AWARE_FIELD_GROUPS:
                if self.debug:
                    print(f"⚠️ Unknown group: {group_name}, skipping")
                continue
            
            try:
                if self.debug:
                    print(f"🔄 Processing group '{group_name}'")
                
                # Get group configuration
                group_config = DOCUMENT_AWARE_FIELD_GROUPS[group_name]
                group_fields = group_config["fields"]
                
                # Generate group-specific prompt (proven format from 90.6% accuracy version)
                group_prompt = self._generate_group_prompt(group_name, group_config)
                
                if self.debug:
                    print(f"📝 Generated prompt for '{group_name}' ({len(group_fields)} fields, {len(group_prompt)} chars)")
                
                # Extract fields for this group with increased token limit
                start_time = time.time()
                # CRITICAL: Increase token limit for better extraction completeness
                group_response = model_extractor_func(
                    image_path, 
                    group_prompt,
                    max_new_tokens=1500  # Increased from default 1000 for complete extraction
                )
                processing_time = time.time() - start_time
                
                # Parse group response
                group_data = self._parse_group_response(group_response, group_fields)
                
                # Update extracted data with group results
                for field in group_fields:
                    if field in group_data:
                        extracted_data[field] = group_data[field]
                
                # Track group success
                found_in_group = sum(1 for field in group_fields if group_data.get(field, "NOT_FOUND") != "NOT_FOUND")
                group_results[group_name] = {
                    "fields_attempted": len(group_fields),
                    "fields_found": found_in_group,
                    "success_rate": found_in_group / len(group_fields) if group_fields else 0,
                    "processing_time": processing_time,
                    "raw_response": group_response[:100] + "..." if len(group_response) > 100 else group_response
                }
                
                self.stats["total_groups_processed"] += 1
                self.stats["successful_groups"] += 1
                
                if self.debug:
                    print(f"✅ Group '{group_name}' completed: {found_in_group}/{len(group_fields)} fields")
                
            except Exception as e:
                if self.debug:
                    print(f"❌ Group '{group_name}' failed: {e}")
                
                group_results[group_name] = {
                    "fields_attempted": len(group_config.get("fields", [])),
                    "fields_found": 0,
                    "success_rate": 0,
                    "error": str(e)
                }
                
                self.stats["total_groups_processed"] += 1
                self.stats["failed_groups"] += 1
        
        # Compile group metadata
        metadata = {
            "groups_processed": list(group_results.keys()),
            "group_results": group_results,
            "total_fields_attempted": sum(r["fields_attempted"] for r in group_results.values()),
            "total_fields_found": sum(r["fields_found"] for r in group_results.values()),
            "overall_group_success_rate": sum(r["success_rate"] for r in group_results.values()) / len(group_results) if group_results else 0
        }
        
        return extracted_data, metadata
    
    def _generate_group_prompt(self, group_name: str, group_config: Dict) -> str:
        """
        Generate group-specific prompt using the proven format from 90.6% accuracy version.
        
        This exactly replicates the successful prompt format that achieved 90.6%.
        """
        fields = group_config["fields"]
        expertise_frame = group_config["expertise_frame"]
        cognitive_context = group_config["cognitive_context"] 
        focus_instruction = group_config["focus_instruction"]
        
        # Build the exact prompt format that worked
        prompt = f"""TASK: {expertise_frame}

DOCUMENT CONTEXT: You are analyzing a business document image. Consider the document type when extracting fields.

{cognitive_context}

{focus_instruction}

OUTPUT FORMAT - EXACTLY {len(fields)} LINES:
"""
        
        # Add each field with specific instructions for critical fields
        for field in fields:
            if field == "PAYMENT_METHOD":
                prompt += f"{field}: [payment type like AMEX, Visa, Mastercard, Cash, EFTPOS or NOT_FOUND]\n"
            elif field == "LINE_ITEM_QUANTITIES":
                prompt += f"{field}: [pipe-separated quantities for ALL items, scan entire document or NOT_FOUND]\n"
            elif field == "LINE_ITEM_DESCRIPTIONS":
                prompt += f"{field}: [pipe-separated ALL item descriptions, scan entire document for every item or NOT_FOUND]\n"
            elif field == "LINE_ITEM_PRICES":
                prompt += f"{field}: [pipe-separated ALL item prices with $ symbol, scan entire document or NOT_FOUND]\n"
            elif field in ["BUSINESS_PHONE", "PAYER_PHONE"]:
                prompt += f"{field}: [complete phone like (08) 4482 2347 with area code or NOT_FOUND]\n"
            elif field == "BUSINESS_ABN":
                prompt += f"{field}: [11-digit ABN with spaces - read digits carefully or NOT_FOUND]\n"
            elif field in ["PAYER_ADDRESS", "BUSINESS_ADDRESS"]:
                prompt += f"{field}: [complete address with 4-digit postcode like 'Street, City STATE 6000' or NOT_FOUND]\n"
            elif field == "STORE_LOCATION":
                prompt += f"{field}: [store location like 'Perth WA 6000' or NOT_FOUND]\n"
            elif field == "RECEIPT_NUMBER":
                prompt += f"{field}: [receipt/reference number like R789121 or NOT_FOUND]\n"
            elif field in ["TOTAL_AMOUNT", "SUBTOTAL_AMOUNT", "GST_AMOUNT"]:
                prompt += f"{field}: [dollar amount with $ symbol and 2 decimals or NOT_FOUND]\n"
            else:
                prompt += f"{field}: [value or NOT_FOUND]\n"
        
        # Add focus section with enhanced instructions matching 90.6% accuracy format
        if group_name == "regulatory_financial":
            prompt += "\n💡 FOCUS:\n• ABN: Look for 'ABN' label, read digits carefully - 11 digits total\n• Dollar amounts: Include $ symbol, 2 decimal places\n• GST vs Total: GST is tax component, Total is final amount\n• CRITICAL: Read all digits precisely - don't misread similar numbers\n"
        elif group_name == "entity_contacts":
            prompt += "\n💡 FOCUS:\n• Supplier vs Payer: Supplier is business issuing document, Payer is customer\n• Complete addresses: Include street, city, state, FULL 4-digit postcode\n• Phone format: Look for (08) or (02) area codes - Australian format\n• Phone extraction: Look carefully for complete numbers, not partial\n• ABN format: 11 digits, may have spaces between groups\n"
        elif group_name == "transaction_details":
            prompt += "\n💡 FOCUS:\n• SCAN ENTIRE DOCUMENT: Look for ALL line items, don't stop early\n• Include duplicates: Some items may appear multiple times\n• FORMAT: Use PIPE-SEPARATED with spaces: item1 | item2 | item3 | etc.\n• THOROUGHNESS: Look in all table sections, don't miss any rows\n• COMPLETENESS: Each description needs matching quantity and price\n• CRITICAL: Scan the full document area for line items\n• DOUBLE CHECK: Count all items you can see in the document\n"
        elif group_name == "temporal_data":
            prompt += "\n💡 FOCUS:\n• Date format: DD/MM/YYYY preferred\n• Invoice vs Transaction dates: Invoice=issued, Transaction=occurred\n• Due dates: Payment deadline for invoices\n"
        elif group_name == "document_metadata":
            prompt += "\n💡 FOCUS:\n• Document type: Exactly 'invoice', 'receipt', or 'statement'\n• Receipt number: Look for R-numbers like R789121, transaction IDs\n• Payment method: MUST extract card type (AMEX, Visa, Mastercard) or payment type\n• Transaction date: Date of purchase/transaction\n• CRITICAL: Never skip PAYMENT_METHOD - look for any payment indicator\n"
        
        # Add the exact format rules that achieved 90.6% accuracy
        prompt += """
FORMAT RULES:
- Use exactly: KEY: value (colon and space)
- NEVER use: **KEY:** or **KEY** or *KEY* or KEY: or any formatting
- Plain text only - NO markdown, NO bold, NO italic
- Include ALL """ + str(len(fields)) + """ fields even if NOT_FOUND
- Extract ONLY what you can see in the document
- Do NOT guess, calculate, or make up values
- Use NOT_FOUND if field is not visible or not applicable
- Output ONLY these """ + str(len(fields)) + """ lines, nothing else
- For LINE_ITEM fields: use PIPE-SEPARATED format (e.g., value1 | value2 | value3)
- CRITICAL: Find ALL line items in the document - scan thoroughly
- For LINE_ITEM_QUANTITIES: NEVER leave blanks, use 1 if quantity unclear
- SCAN CAREFULLY: Look for all items in the table, don't stop early
- For PAYMENT_METHOD: Look for ANY payment indicator (AMEX, card types, etc)
- For addresses: Include FULL address with postcode
- DO NOT output the same field name multiple times

STOP after the last field. Do not add explanations or comments."""

        return prompt
    
    def _parse_group_response(self, response: str, expected_fields: List[str]) -> Dict[str, str]:
        """
        Parse group response using robust extraction parser instead of custom logic.
        This fixes parsing bugs and ensures consistency with document-aware processors.
        """
        # Use the same robust parser that fixed the Llama processor 
        from common.extraction_parser import parse_extraction_response
        return parse_extraction_response(response, expected_fields=expected_fields)