"""
Dynamic Field Processor

Handles dynamic field list updates based on detected document type.
Provides document-specific field extraction without complex dependencies.
"""

from typing import Any, Dict, List

from rich import print as rprint


class DynamicFieldProcessor:
    """Manages dynamic field processing based on document type."""

    def __init__(self):
        """Initialize with document-specific field configurations."""
        self.document_fields = self._get_document_field_mappings()

    def _get_document_field_mappings(self) -> Dict[str, List[str]]:
        """Get hardcoded document field mappings to avoid schema dependencies."""
        return {
            "invoice": [
                'DOCUMENT_TYPE', 'BUSINESS_ABN', 'SUPPLIER_NAME', 'BUSINESS_ADDRESS',
                'PAYER_NAME', 'PAYER_ADDRESS', 'INVOICE_DATE', 'LINE_ITEM_DESCRIPTIONS',
                'LINE_ITEM_QUANTITIES', 'LINE_ITEM_PRICES', 'LINE_ITEM_TOTAL_PRICES',
                'IS_GST_INCLUDED', 'GST_AMOUNT', 'TOTAL_AMOUNT'
            ],
            "receipt": [
                'DOCUMENT_TYPE', 'BUSINESS_ABN', 'SUPPLIER_NAME', 'BUSINESS_ADDRESS',
                'PAYER_NAME', 'PAYER_ADDRESS', 'INVOICE_DATE', 'LINE_ITEM_DESCRIPTIONS',
                'LINE_ITEM_QUANTITIES', 'LINE_ITEM_PRICES', 'LINE_ITEM_TOTAL_PRICES',
                'IS_GST_INCLUDED', 'GST_AMOUNT', 'TOTAL_AMOUNT'
            ],
            "bank_statement": [
                'DOCUMENT_TYPE', 'STATEMENT_DATE_RANGE', 'LINE_ITEM_DESCRIPTIONS',
                'TRANSACTION_DATES', 'TRANSACTION_AMOUNTS_PAID', 'TRANSACTION_AMOUNTS_RECEIVED',
                'ACCOUNT_BALANCE'
            ]
        }

    def get_document_fields(self, document_type: str) -> List[str]:
        """
        Get field list for specific document type.

        Args:
            document_type: Type of document (invoice, receipt, bank_statement)

        Returns:
            List of field names for the document type
        """
        normalized_type = self._normalize_document_type(document_type)
        return self.document_fields.get(normalized_type, self.document_fields["invoice"])

    def update_processor_fields(self, processor: Any, document_type: str, verbose: bool = True) -> bool:
        """
        Update processor field list and document type based on detection.

        Args:
            processor: DocumentAwareInternVL3Processor instance
            document_type: Detected document type
            verbose: Whether to print update information

        Returns:
            True if successful, False otherwise
        """
        try:
            # Get document-specific fields
            field_list = self.get_document_fields(document_type)
            normalized_type = self._normalize_document_type(document_type)

            # Update processor attributes (like the enhanced version)
            if hasattr(processor, 'field_list'):
                processor.field_list = field_list
            if hasattr(processor, 'field_count'):
                processor.field_count = len(field_list)
            if hasattr(processor, 'detected_document_type'):
                processor.detected_document_type = normalized_type

            if verbose:
                rprint(f"   🔄 Updated processor: {normalized_type} ({len(field_list)} fields)")

            return True

        except Exception as e:
            if verbose:
                rprint(f"   [yellow]⚠️ Failed to update processor fields: {e}[/yellow]")
            return False

    def get_field_statistics(self, document_type: str) -> Dict[str, Any]:
        """Get field statistics for a document type."""
        field_list = self.get_document_fields(document_type)
        return {
            "document_type": self._normalize_document_type(document_type),
            "field_count": len(field_list),
            "fields": field_list,
            "critical_fields": self._get_critical_fields(document_type),
            "financial_fields": self._get_financial_fields(document_type)
        }

    def _get_critical_fields(self, document_type: str) -> List[str]:
        """Get critical fields for document type."""
        critical_mappings = {
            "invoice": ['DOCUMENT_TYPE', 'SUPPLIER_NAME', 'TOTAL_AMOUNT'],
            "receipt": ['DOCUMENT_TYPE', 'SUPPLIER_NAME', 'TOTAL_AMOUNT'],
            "bank_statement": ['DOCUMENT_TYPE', 'STATEMENT_DATE_RANGE', 'ACCOUNT_BALANCE']
        }
        normalized_type = self._normalize_document_type(document_type)
        return critical_mappings.get(normalized_type, [])

    def _get_financial_fields(self, document_type: str) -> List[str]:
        """Get financial fields for document type."""
        financial_mappings = {
            "invoice": ['GST_AMOUNT', 'TOTAL_AMOUNT', 'LINE_ITEM_PRICES', 'LINE_ITEM_TOTAL_PRICES'],
            "receipt": ['GST_AMOUNT', 'TOTAL_AMOUNT', 'LINE_ITEM_PRICES', 'LINE_ITEM_TOTAL_PRICES'],
            "bank_statement": ['TRANSACTION_AMOUNTS_PAID', 'TRANSACTION_AMOUNTS_RECEIVED', 'ACCOUNT_BALANCE']
        }
        normalized_type = self._normalize_document_type(document_type)
        return financial_mappings.get(normalized_type, [])

    def _normalize_document_type(self, doc_type: str) -> str:
        """Normalize document type string."""
        doc_type_lower = doc_type.lower().strip()

        if "invoice" in doc_type_lower or "tax" in doc_type_lower:
            return "invoice"
        elif "bank" in doc_type_lower or "statement" in doc_type_lower:
            return "bank_statement"
        elif "receipt" in doc_type_lower:
            return "receipt"

        return "invoice"  # Default fallback

    def get_all_supported_types(self) -> List[str]:
        """Get list of all supported document types."""
        return list(self.document_fields.keys())

    def validate_processor_compatibility(self, processor: Any) -> bool:
        """Check if processor is compatible with dynamic field updates."""
        required_attributes = ['field_list', 'field_count', 'detected_document_type']

        for attr in required_attributes:
            if not hasattr(processor, attr):
                rprint(f"[yellow]⚠️ Processor missing attribute: {attr}[/yellow]")
                return False

        return True


def main():
    """Test the dynamic field processor."""
    processor = DynamicFieldProcessor()

    # Test field mappings
    for doc_type in ["invoice", "receipt", "bank_statement"]:
        fields = processor.get_document_fields(doc_type)
        stats = processor.get_field_statistics(doc_type)
        print(f"\n{doc_type.upper()}:")
        print(f"  Fields: {len(fields)}")
        print(f"  Critical: {stats['critical_fields']}")
        print(f"  Financial: {stats['financial_fields']}")


if __name__ == "__main__":
    main()