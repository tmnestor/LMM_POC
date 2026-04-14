"""Formal interface for document extraction processors.

Defines @runtime_checkable Protocols that capture the duck-typed interfaces
already expected by DocumentPipeline and UnifiedBankExtractor.

DocumentProcessor: Required interface for all model processors.
BatchCapableProcessor: Deprecated — kept only for BatchDocumentProcessor compat.
    Will be removed when BatchDocumentProcessor is deleted.
"""

from typing import Any, NotRequired, Protocol, TypedDict, runtime_checkable

from PIL import Image


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

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_tokens: int = 1024,
    ) -> str:
        """Run model inference on an image with a text prompt.

        Args:
            image: PIL Image to process.
            prompt: Text prompt for the model.
            max_tokens: Maximum tokens to generate.

        Returns:
            Raw model response string.
        """
        ...


@runtime_checkable
class BatchCapableProcessor(Protocol):
    """Deprecated: kept for BatchDocumentProcessor backward compat only.

    New code should use DocumentPipeline + orchestrator.supports_batch instead.
    Will be removed when BatchDocumentProcessor is deleted.
    """

    def batch_detect_documents(
        self,
        image_paths: list[str],
        verbose: bool = False,
    ) -> list[ClassificationResult]: ...

    def batch_extract_documents(
        self,
        image_paths: list[str],
        classification_infos: list[dict[str, Any]],
        verbose: bool = False,
    ) -> list[ExtractionResult]: ...
