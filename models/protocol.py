"""Formal interface for document extraction processors.

Defines a @runtime_checkable Protocol that captures the duck-typed interface
already expected by BatchDocumentProcessor and BankStatementAdapter.

Batch methods (batch_detect_documents, batch_extract_documents) are NOT part
of the Protocol — they are detected via hasattr() at runtime by batch_processor.py.
"""

from typing import Any, NotRequired, Protocol, TypedDict, runtime_checkable


class ClassificationResult(TypedDict):
    """Result from document type detection."""

    document_type: str
    confidence: float
    raw_response: str
    prompt_used: str
    error: NotRequired[str]


class ExtractionResult(TypedDict):
    """Result from document field extraction."""

    image_name: str
    extracted_data: dict[str, str]
    raw_response: str
    processing_time: float
    response_completeness: float
    content_coverage: float
    extracted_fields_count: int
    field_count: int
    document_type: NotRequired[str]
    extraction_mode: NotRequired[str]


@runtime_checkable
class DocumentProcessor(Protocol):
    """Protocol for document extraction processors.

    Required attributes:
        model: The underlying vision-language model.
        tokenizer: The tokenizer (or processor) for the model.
        batch_size: Current batch size for inference.

    Required methods:
        detect_and_classify_document: Classify the document type from an image.
        process_document_aware: Extract fields based on classification info.
    """

    model: Any
    tokenizer: Any
    batch_size: int

    def detect_and_classify_document(
        self,
        image_path: str,
        verbose: bool = False,
    ) -> ClassificationResult:
        """Classify the document type from an image.

        Args:
            image_path: Path to the document image.
            verbose: Enable debug output.

        Returns:
            ClassificationResult with at least 'document_type' key.
        """
        ...

    def process_document_aware(
        self,
        image_path: str,
        classification_info: dict[str, Any],
        verbose: bool = False,
    ) -> ExtractionResult:
        """Extract fields from a document image based on its classification.

        Args:
            image_path: Path to the document image.
            classification_info: Output from detect_and_classify_document.
            verbose: Enable debug output.

        Returns:
            ExtractionResult with extracted field values.
        """
        ...
