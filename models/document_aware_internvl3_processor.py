#!/usr/bin/env python3
"""
Document-Aware InternVL3 Hybrid Processor - SAFE COPY WITH INTERNVL3 MODEL

A hybrid processor that combines:
- InternVL3 model (for better accuracy potential)
- Llama's proven processing pipeline (for reliable processing)
- ExtractionCleaner integration (for 🧹 CLEANER CALLED output)

ZERO RISK to existing Llama processor - this is a completely independent copy.

DOCUMENT AWARE REDUCTION OPTIMIZED:
- Invoice/Receipt: 11 fields (62-67% reduction) → ~50% faster processing
- Bank Statement: 5 fields (75% reduction) → ~75% faster processing
- Dynamic max_new_tokens automatically scales with reduced field counts
"""

import time
import warnings
from typing import Dict, List, Optional

import torch
import torchvision.transforms as T
from PIL import Image
from transformers import AutoModel, AutoTokenizer

from common.batch_processor import load_document_field_definitions
from common.config import (
    DEFAULT_IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    INTERNVL3_MODEL_PATH,
    get_auto_batch_size,
    get_max_new_tokens,
)
from common.extraction_cleaner import ExtractionCleaner
from common.gpu_optimization import (
    configure_cuda_memory_allocation,
    emergency_cleanup,
    get_available_gpu_memory,
    optimize_model_for_v100,
)
from common.simple_prompt_loader import SimplePromptLoader, load_internvl3_prompt

warnings.filterwarnings("ignore")


class DocumentAwareInternVL3HybridProcessor:
    """Hybrid processor: InternVL3 model + Llama's proven processing pipeline."""

    def __init__(
        self,
        field_list: List[str],
        model_path: str = None,
        device: str = "cuda",
        debug: bool = False,
        batch_size: Optional[int] = None,
        skip_model_loading: bool = False,
        pre_loaded_model=None,
        pre_loaded_tokenizer=None,
        prompt_config: Dict = None,
        max_tiles: int = None,
    ):
        """
        Initialize hybrid processor with InternVL3 model and Llama processing logic.

        Args:
            field_list (List[str]): Fields to extract
            model_path (str): Path to InternVL3 model
            device (str): Device to run on
            debug (bool): Enable debug output
            batch_size (int): Batch size (auto-detected if None)
            skip_model_loading (bool): Skip model loading (use pre-loaded)
            pre_loaded_model: Pre-loaded model instance
            pre_loaded_tokenizer: Pre-loaded tokenizer instance
            prompt_config (Dict): Configuration for prompts (single source of truth)
            skip_model_loading (bool): Skip loading model (for reusing existing model)
            pre_loaded_model: Pre-loaded InternVL3 model (avoids reloading)
            pre_loaded_tokenizer: Pre-loaded InternVL3 tokenizer (avoids reloading)
            max_tiles (int): Max image tiles for preprocessing (REQUIRED - set in notebook CONFIG)
        """
        self.field_list = field_list
        self.field_count = len(field_list)
        self.model_path = model_path or INTERNVL3_MODEL_PATH
        self.device = device
        self.debug = debug
        self.prompt_config = prompt_config  # Single source of truth for prompt configuration
        self.max_tiles = max_tiles  # REQUIRED: Notebook-configurable tile count

        # Initialize components (InternVL3 specific)
        self.model = pre_loaded_model
        self.tokenizer = pre_loaded_tokenizer
        self.generation_config = None

        # Fix pad_token_id if tokenizer is pre-loaded to suppress warnings
        if self.tokenizer is not None and self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Also set pad_token_id on model's generation_config to suppress warnings
        if self.model is not None and self.tokenizer is not None:
            if hasattr(self.model, 'generation_config'):
                self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
            elif hasattr(self.model.config, 'pad_token_id'):
                self.model.config.pad_token_id = self.tokenizer.eos_token_id

        # Detect model variant (2B vs 8B) for tile optimization
        self.is_8b_model = "8B" in self.model_path

        # Initialize extraction cleaner for value normalization (🧹 CLEANER CALLED output)
        self.cleaner = ExtractionCleaner(debug=debug)

        # Document-specific field lists - loaded from config/field_definitions.yaml
        # SINGLE SOURCE OF TRUTH - no hardcoding here
        self.document_field_lists = load_document_field_definitions()

        if self.debug:
            print(
                f"🎯 InternVL3 Hybrid processor initialized for {self.field_count} fields: {field_list[0]} → {field_list[-1]}"
            )

        # Configure CUDA memory allocation
        configure_cuda_memory_allocation()

        # Configure batch processing
        self._configure_batch_processing(batch_size)

        # Configure generation parameters for dynamic field count
        self._configure_generation()

        # Load model and tokenizer (unless skipping for reuse or already provided)
        if not skip_model_loading and (self.model is None or self.tokenizer is None):
            self._load_model()
        elif self.model is not None and self.tokenizer is not None:
            if self.debug:
                print("✅ Using pre-loaded InternVL3 model and tokenizer")
                print(f"🔧 Device: {self.model.device}")
                print(f"💾 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            # Apply V100 optimizations to pre-loaded model
            optimize_model_for_v100(self.model)

    def _configure_batch_processing(self, batch_size: Optional[int]):
        """Configure batch processing parameters."""
        if batch_size is not None:
            self.batch_size = max(1, batch_size)
            if self.debug:
                print(f"🎯 Using manual batch size: {self.batch_size}")
        else:
            # Auto-detect batch size based on available memory
            available_memory = get_available_gpu_memory(self.device)
            self.batch_size = get_auto_batch_size("internvl3", available_memory)
            if self.debug:
                print(
                    f"🤖 Auto-detected batch size: {self.batch_size} (GPU Memory: {available_memory:.1f}GB)"
                )

    def _configure_generation(self):
        """Configure generation parameters for InternVL3."""
        # InternVL3 generation config - chat() method only accepts max_new_tokens and do_sample
        # NOTE: temperature and top_p are NOT valid for InternVL3 chat() and cause warnings
        self.generation_config = {
            "max_new_tokens": get_max_new_tokens("internvl3", self.field_count),
            "do_sample": False,  # Greedy decoding for consistent extraction
            "use_cache": True,
        }

        if self.debug:
            performance_gain = max(
                0, (29 - self.field_count) / 29 * 100
            )  # vs old invoice max
            print(
                f"🎯 DOCUMENT AWARE REDUCTION: {self.field_count} fields (~{performance_gain:.0f}% fewer than original 29)"
            )
            print(
                f"🎯 Generation config: max_new_tokens={self.generation_config['max_new_tokens']}, "
                f"do_sample={self.generation_config['do_sample']} (greedy decoding)"
            )

    def _get_model_device(self):
        """Get the device for pixel_values tensor placement.

        For multi-GPU models with device_map="auto", pixel_values must be placed
        on the device where the vision model's embedding layer resides, since
        that's where image tensors enter the model.
        """
        # For multi-GPU device_map models, get the vision model embedding device
        # This is critical: pixel_values enter through vision_model.embeddings
        if hasattr(self.model, 'vision_model') and hasattr(self.model.vision_model, 'embeddings'):
            try:
                vision_embed_device = next(self.model.vision_model.embeddings.parameters()).device
                return vision_embed_device
            except (StopIteration, AttributeError):
                pass

        # Fallback: check model.device attribute
        if hasattr(self.model, 'device'):
            return self.model.device

        try:
            # Get device from first parameter
            return next(self.model.parameters()).device
        except (StopIteration, AttributeError):
            # Fallback to cuda if available
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _load_model(self):
        """Load InternVL3 model and tokenizer with optimal configuration."""
        if self.debug:
            print(f"🔄 Loading InternVL3 model from: {self.model_path}")

        try:
            # Load InternVL3 model with V100 optimizations - REVERTED TO WORKING CONFIG
            self.model = AutoModel.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",  # REVERTED: Custom device mapping broke the model
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                # use_flash_attn=True,  # Disabled: Poor V100 compatibility
            ).eval()

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False,  # More reliable for structured tasks
            )

            # Fix pad_token_id to suppress warnings during generation
            if self.tokenizer.pad_token_id is None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            if self.debug:
                print("✅ InternVL3 model loaded successfully")
                print(f"🔧 Device: {self.model.device}")
                print(
                    f"💾 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
                )

            # Apply V100 optimizations
            optimize_model_for_v100(self.model)

        except Exception as e:
            if self.debug:
                print(f"❌ Error loading InternVL3 model: {e}")
            raise

    def build_transform(self, input_size=DEFAULT_IMAGE_SIZE):
        """Build InternVL3 image transformation pipeline."""
        transform = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size), interpolation=T.InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Standard InternVL3 find_closest_aspect_ratio from official documentation."""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """Standard InternVL3 dynamic_preprocess from official documentation."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # Standard target ratios calculation
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # Find the closest aspect ratio using standard method
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []

        # Standard tiling process
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        # Add thumbnail if requested (standard InternVL3 feature)
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=None):
        """Complete InternVL3 image loading and preprocessing pipeline."""
        if max_num is None:
            if self.max_tiles is not None:
                max_num = self.max_tiles  # Use notebook-configured value
            else:
                # FAIL FAST: Require explicit max_tiles configuration
                raise ValueError(
                    "❌ FATAL: max_tiles not configured!\n"
                    "💡 Add MAX_TILES to your notebook CONFIG:\n"
                    "   CONFIG = {\n"
                    "       ...\n"
                    "       'MAX_TILES': 36,  # H200: 36, V100-8B: 14, 2B: 18\n"
                    "   }\n"
                    "💡 Pass to processor initialization:\n"
                    "   hybrid_processor = DocumentAwareInternVL3HybridProcessor(\n"
                    "       ...\n"
                    "       max_tiles=CONFIG['MAX_TILES']\n"
                    "   )"
                )

        if self.debug:
            print(f"🔍 LOAD_IMAGE: max_num={max_num}, input_size={input_size}")

        # Load image
        image = Image.open(image_file).convert("RGB")

        # Process into tiles using standard InternVL3 dynamic_preprocess
        images = self.dynamic_preprocess(
            image, min_num=1, max_num=max_num, image_size=input_size, use_thumbnail=True
        )

        # Apply transforms
        transform = self.build_transform(input_size=input_size)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)

        # CRITICAL FIX: Ensure tensor type matches model weights
        # Convert to model's dtype to prevent "Input type (float) and bias type (c10::BFloat16)" error
        try:
            # For 8-bit quantized models, we need to check vision model's weight dtype
            if hasattr(self.model, 'vision_model') and hasattr(self.model.vision_model, 'embeddings'):
                # Get dtype from vision model's embedding layer (most reliable for quantized models)
                vision_dtype = next(self.model.vision_model.embeddings.parameters()).dtype
                pixel_values = pixel_values.to(dtype=vision_dtype)
                if self.debug:
                    print(f"🔧 TENSOR_DTYPE: Using vision model dtype {vision_dtype}")
            elif hasattr(self.model, 'dtype'):
                pixel_values = pixel_values.to(dtype=self.model.dtype)
                if self.debug:
                    print(f"🔧 TENSOR_DTYPE: Using model.dtype {self.model.dtype}")
            elif hasattr(self.model.vision_model, 'dtype'):
                pixel_values = pixel_values.to(dtype=self.model.vision_model.dtype)
                if self.debug:
                    print(f"🔧 TENSOR_DTYPE: Using vision_model.dtype {self.model.vision_model.dtype}")
            else:
                # Try to get dtype from first model parameter
                model_dtype = next(self.model.parameters()).dtype
                pixel_values = pixel_values.to(dtype=model_dtype)
                if self.debug:
                    print(f"🔧 TENSOR_DTYPE: Using parameter dtype {model_dtype}")
        except Exception as e:
            # Fallback dtype detection - check actual model parameter dtype
            # CRITICAL: Must detect float32 models correctly for V100 compatibility
            try:
                # Get dtype from any model parameter
                first_param = next(iter(self.model.parameters()))
                detected_dtype = first_param.dtype
                pixel_values = pixel_values.to(dtype=detected_dtype)
                if self.debug:
                    print(f"🔧 TENSOR_DTYPE: Using detected parameter dtype {detected_dtype}")
            except Exception:
                # Last resort fallback: use float32 for safety (V100 compatible)
                pixel_values = pixel_values.to(dtype=torch.float32)
                if self.debug:
                    print("🔧 TENSOR_DTYPE: Using float32 fallback (V100 safe)")

            if self.debug:
                print(f"⚠️ Primary dtype detection failed: {e}")

        # Move to model's device
        if self.model is not None:
            model_device = self._get_model_device()
            if pixel_values.device != model_device:
                pixel_values = pixel_values.to(model_device)
                if self.debug:
                    print(f"🔧 DEVICE_MOVE: Moved tensor from {pixel_values.device} to {model_device}")
            elif self.debug:
                print(f"✅ DEVICE_OK: Tensor already on {model_device}")

        # CRITICAL V100 DEBUGGING: Verify dtype and device match
        if self.debug:
            try:
                model_param = next(iter(self.model.parameters()))
                print(f"🔍 DTYPE_CHECK: pixel_values={pixel_values.dtype}, model_param={model_param.dtype}")
                print(f"🔍 DEVICE_CHECK: pixel_values={pixel_values.device}, model_param={model_param.device}")
                if pixel_values.dtype != model_param.dtype:
                    print(f"⚠️ DTYPE_MISMATCH: pixel_values ({pixel_values.dtype}) != model ({model_param.dtype})")
            except Exception as debug_err:
                print(f"⚠️ Debug check failed: {debug_err}")

        if self.debug:
            print(
                f"📐 TENSOR_SHAPE: {pixel_values.shape} (batch_size={pixel_values.shape[0]} tiles)"
            )
            print(f"📊 TENSOR_DTYPE: {pixel_values.dtype}")
            print(f"📍 TENSOR_DEVICE: {pixel_values.device}")

        return pixel_values

    def get_extraction_prompt(self, document_type: str = None) -> str:
        """
        Get extraction prompt for document type - uses Llama prompts for consistency.

        Args:
            document_type: Document type ('invoice', 'receipt', 'bank_statement')
                          If None, uses 'universal' prompt

        Returns:
            Complete prompt string loaded from YAML
        """
        if document_type is None:
            document_type = "universal"

        if self.debug:
            print(f"📝 Loading {document_type} prompt for InternVL3 Hybrid")

        try:
            return load_internvl3_prompt(document_type)
        except Exception as e:
            if self.debug:
                print(f"⚠️ Failed to load {document_type} prompt, falling back to universal")
            return load_internvl3_prompt("universal")

    def get_supported_document_types(self) -> List[str]:
        """Get list of supported document types."""
        return SimplePromptLoader.get_available_prompts("internvl3_prompts.yaml")


    def detect_and_classify_document(self, image_path: str, verbose: bool = False) -> Dict:
        """
        Detect document type using InternVL3 model and document detection prompt.
        Compatible with BatchDocumentProcessor workflow.

        Args:
            image_path: Path to the image to classify
            verbose: Whether to show detailed output

        Returns:
            Dict with classification info including document_type
        """
        try:
            from pathlib import Path

            import yaml

            # Use prompt_config if provided (single source of truth), otherwise fallback to YAML
            if self.prompt_config:
                detection_path = Path(self.prompt_config["detection_file"])
                detection_key = self.prompt_config["detection_key"]
                if verbose:
                    print(f"🔧 CONFIG DEBUG - Using prompt_config: detection_key='{detection_key}'")
            else:
                # Fallback to hardcoded YAML (legacy behavior)
                detection_path = Path("prompts/document_type_detection.yaml")
                detection_key = "detection"
                if verbose:
                    print(f"🔧 CONFIG DEBUG - Using fallback: detection_key='{detection_key}'")

            with detection_path.open("r") as f:
                detection_config = yaml.safe_load(f)

            detection_prompt = detection_config["prompts"][detection_key]["prompt"]
            max_tokens = detection_config.get("settings", {}).get("max_new_tokens", 50)

            if verbose:
                # Use direct stdout to bypass Rich console completely for detection
                import sys
                sys.stdout.write(f"🔍 Using InternVL3 document detection prompt: {detection_key}\n")
                sys.stdout.write(f"📝 Prompt: {detection_prompt[:100]}...\n")
                sys.stdout.flush()

            # Load and preprocess image using InternVL3 pipeline
            pixel_values = self.load_image(image_path)

            # Ensure on correct device (backup check)
            model_device = self._get_model_device()
            if pixel_values.device != model_device:
                pixel_values = pixel_values.to(model_device)
                if verbose:
                    print(f"🔧 BACKUP_DEVICE_FIX: Moved tensor to {model_device}")

            # Generate response using InternVL3 model with detection-specific limits
            response = self._resilient_generate(
                pixel_values=pixel_values,
                question=detection_prompt,
                max_new_tokens=max_tokens,
                temperature=0.1,
                do_sample=False,
                is_detection=True  # Enable strict token limits for detection
            )

            if verbose:
                # Use direct stdout to bypass Rich console completely for detection
                import sys
                sys.stdout.write(f"🤖 Model response: {response}\n")
                sys.stdout.flush()

            # Parse document type from response
            document_type = self._parse_document_type_response(response, detection_config)

            if verbose:
                # Use direct stdout to bypass Rich console completely for detection
                import sys
                sys.stdout.write(f"✅ Detected document type: {document_type}\n")
                sys.stdout.flush()

            return {
                "document_type": document_type,
                "confidence": 1.0,  # InternVL3 doesn't provide confidence scores
                "raw_response": response,
                "prompt_used": detection_key
            }

        except Exception as e:
            # ALWAYS show detection errors - critical for debugging
            import sys
            sys.stdout.write(f"❌ DETECTION ERROR: {e}\n")
            sys.stdout.flush()
            if self.debug:
                import traceback
                sys.stdout.write("❌ DETECTION ERROR TRACEBACK:\n")
                sys.stdout.flush()
                traceback.print_exc()

            # Fallback to simple heuristic
            return {
                "document_type": "INVOICE",  # Use uppercase for consistency
                "confidence": 0.1,
                "raw_response": "",
                "prompt_used": "fallback_heuristic",
                "error": str(e)
            }

    def process_document_aware(self, image_path: str, classification_info: Dict, verbose: bool = False) -> Dict:
        """
        Process document using document-specific extraction based on detected type.
        Compatible with BatchDocumentProcessor workflow.

        Args:
            image_path: Path to the image to process
            classification_info: Result from detect_and_classify_document()
            verbose: Whether to show detailed output

        Returns:
            Dict with extraction results in BatchDocumentProcessor format
        """
        try:
            document_type = classification_info["document_type"].lower()

            if verbose:
                print(f"📊 Processing {document_type.upper()} document with InternVL3")

            # For bank statements, use vision-based structure classification
            if document_type == 'bank_statement':
                try:
                    # Try different import methods to handle various execution contexts
                    try:
                        from common.vision_bank_statement_classifier import (
                            classify_bank_statement_structure_vision,
                        )
                    except ImportError:
                        import sys
                        from pathlib import Path
                        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                        from common.vision_bank_statement_classifier import (
                            classify_bank_statement_structure_vision,
                        )

                    if verbose:
                        print("🔍 Running vision-based structure classification for bank statement")

                    # Pass self (the processor) which has the load_image method
                    structure_type = classify_bank_statement_structure_vision(
                        image_path,
                        model=self,  # Pass the processor itself which has load_image
                        processor=None,  # InternVL3 doesn't use separate processor
                        verbose=verbose
                    )

                    # Map structure to specific prompt
                    document_type = f"bank_statement_{structure_type}"

                    if verbose:
                        print(f"🏗️ Bank statement structure: {structure_type}")
                        print(f"📝 Using prompt key: {document_type}")

                except Exception as e:
                    if verbose:
                        print(f"⚠️ Vision classification failed: {e}")
                        print("📝 Falling back to universal prompt")
                    document_type = "universal"

            # Get document-specific prompt using prompt_config (single source of truth)
            if self.prompt_config:
                # Use prompt_config to determine which file and key to use
                # Strip structure suffixes to get base document type
                doc_type_upper = document_type.upper().replace('_FLAT', '').replace('_DATE_GROUPED', '')
                extraction_file = self.prompt_config.get('extraction_files', {}).get(
                    doc_type_upper,
                    'prompts/internvl3_prompts.yaml'
                )

                # Get the prompt key from config (or derive from document type if not specified)
                extraction_keys = self.prompt_config.get('extraction_keys', {})

                if doc_type_upper in extraction_keys:
                    # Use explicitly configured key
                    extraction_key = extraction_keys[doc_type_upper]
                else:
                    # Derive key from document type (already includes structure suffix if present)
                    extraction_key = document_type

                # For bank statements ONLY: if key doesn't include structure suffix, append it
                # This allows config to override by specifying full key like "bank_statement_flat"
                if document_type.startswith('bank_statement') and doc_type_upper == 'BANK_STATEMENT':
                    if '_flat' not in extraction_key and '_date_grouped' not in extraction_key:
                        # Only append if document_type has a structure suffix
                        if '_flat' in document_type:
                            extraction_key = f"{extraction_key}_flat"
                        elif '_date_grouped' in document_type:
                            extraction_key = f"{extraction_key}_date_grouped"

                from pathlib import Path

                from common.simple_prompt_loader import SimplePromptLoader
                loader = SimplePromptLoader()
                extraction_prompt = loader.load_prompt(Path(extraction_file).name, extraction_key)
            else:
                # Fallback to get_extraction_prompt method
                extraction_prompt = self.get_extraction_prompt(document_type)

            if verbose:
                prompt_source = "prompt_config" if self.prompt_config else "get_extraction_prompt"
                print(f"📝 Using {document_type} prompt ({prompt_source}): {len(extraction_prompt)} characters")

            # Get document-specific field list from YAML config (single source of truth)
            doc_type_fields = load_document_field_definitions()
            # Add structure-specific bank statement aliases (same fields as bank_statement)
            if 'bank_statement' in doc_type_fields:
                doc_type_fields['bank_statement_flat'] = doc_type_fields['bank_statement']
                doc_type_fields['bank_statement_date_grouped'] = doc_type_fields['bank_statement']

            # Update field list for document-specific extraction
            original_field_list = self.field_list
            self.field_list = doc_type_fields.get(document_type, doc_type_fields['invoice'])

            # Calculate document-specific max tokens
            from common.config import get_max_new_tokens
            # Use base document type for max tokens calculation
            base_doc_type = document_type.replace('_flat', '').replace('_date_grouped', '')
            doc_specific_tokens = get_max_new_tokens("internvl3", len(self.field_list), base_doc_type)

            # Process with document-specific settings
            result = self.process_single_image(image_path, custom_max_tokens=doc_specific_tokens)

            # Restore original field list
            self.field_list = original_field_list

            if verbose:
                extracted_data = result.get('extracted_data', {})
                found_fields = sum(1 for v in extracted_data.values() if v != 'NOT_FOUND')
                print(f"✅ Extracted {found_fields}/{len(extracted_data)} fields")

            return result

        except Exception as e:
            if verbose:
                print(f"❌ Error in document-aware processing: {e}")

            # Fallback to universal processing
            return self.process_single_image(image_path)

    def _parse_document_type_response(self, response: str, detection_config: Dict) -> str:
        """
        Parse document type from model response using type mappings.

        Args:
            response: Raw model response
            detection_config: Detection configuration with type mappings

        Returns:
            Normalized document type
        """
        response_lower = response.lower().strip()

        if self.debug:
            # Use direct stdout to bypass Rich console completely for detection debug
            import sys
            sys.stdout.write(f"🔍 PARSING DEBUG - Raw response: '{response}'\n")
            sys.stdout.write(f"🔍 PARSING DEBUG - Cleaned response: '{response_lower}'\n")
            sys.stdout.flush()

        # Get type mappings from config
        type_mappings = detection_config.get("type_mappings", {})

        # Direct mapping check
        for variant, canonical in type_mappings.items():
            if variant.lower() in response_lower:
                if self.debug:
                    import sys
                    sys.stdout.write(f"✅ PARSING DEBUG - Found mapping: '{variant}' -> '{canonical}'\n")
                    sys.stdout.flush()
                return canonical

        # Fallback keyword detection
        if any(word in response_lower for word in ["receipt", "purchase", "payment"]):
            if self.debug:
                import sys
                sys.stdout.write("✅ PARSING DEBUG - Keyword match: RECEIPT\n")
                sys.stdout.flush()
            return "RECEIPT"
        elif any(word in response_lower for word in ["bank", "statement", "account"]):
            if self.debug:
                import sys
                sys.stdout.write("✅ PARSING DEBUG - Keyword match: BANK_STATEMENT\n")
                sys.stdout.flush()
            return "BANK_STATEMENT"
        elif any(word in response_lower for word in ["travel", "ticket", "boarding", "airline", "flight", "itinerary", "passenger"]):
            if self.debug:
                import sys
                sys.stdout.write("✅ PARSING DEBUG - Keyword match: TRAVEL_EXPENSE\n")
                sys.stdout.flush()
            return "TRAVEL_EXPENSE"
        elif any(word in response_lower for word in ["invoice", "bill", "tax"]):
            if self.debug:
                import sys
                sys.stdout.write("✅ PARSING DEBUG - Keyword match: INVOICE\n")
                sys.stdout.flush()
            return "INVOICE"

        # Final fallback
        fallback = detection_config.get("settings", {}).get("fallback_type", "INVOICE")
        if self.debug:
            import sys
            sys.stdout.write(f"❌ PARSING DEBUG - No matches found, using fallback: '{fallback}'\n")
            sys.stdout.flush()
        return fallback

    def load_document_image(self, image_path: str) -> Image.Image:
        """Load document image with error handling."""
        try:
            return Image.open(image_path)
        except Exception as e:
            if self.debug:
                print(f"❌ Error loading image {image_path}: {e}")
            raise

    def _resilient_generate(self, pixel_values, question, **generation_kwargs):
        """Resilient generation with OOM fallback using InternVL3 chat method."""

        # Build clean generation parameters - differentiate detection vs extraction limits
        max_tokens = generation_kwargs.get("max_new_tokens", self.generation_config["max_new_tokens"])
        is_detection = generation_kwargs.get("is_detection", False)  # New parameter to identify detection phase

        if is_detection:
            # Detection only needs a short response (just the document type)
            max_tokens = min(max_tokens, 100)  # Allow up to 100 for detection
        # Don't limit extraction tokens as aggressively - let the model complete its response
        # The recursion detection will catch any infinite loops

        # InternVL3 chat() method only accepts specific parameters
        # temperature and top_p are not valid - they cause warnings
        clean_generation_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": generation_kwargs.get(
                "do_sample", self.generation_config["do_sample"]
            ),
            "pad_token_id": self.tokenizer.eos_token_id,  # Suppress pad_token_id warnings
        }

        try:
            # Use InternVL3 chat method instead of generate
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config=clean_generation_kwargs,
                history=None,
                return_history=False
            )

            # CRITICAL: Detect infinite recursion patterns FIRST
            if self._detect_recursion_pattern(response):
                if self.debug:
                    import sys
                    sys.stdout.write(f"⚠️ RECURSION DETECTED: Truncating response at {len(response)} chars\n")
                    sys.stdout.flush()
                # For detection, we only need the document type anyway
                response = response[:200]  # Less aggressive truncation
            elif len(response) > 2000:  # Much higher safety limit for normal responses
                response = response[:2000]  # Allow longer responses for extraction

            return response

        except torch.cuda.OutOfMemoryError:
            if self.debug:
                print("⚠️ CUDA OOM during generation, attempting recovery...")

            # Clear cache and retry
            torch.cuda.empty_cache()
            emergency_cleanup(verbose=False)

            try:
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config=clean_generation_kwargs,
                    history=None,
                    return_history=False
                )

                # CRITICAL: Detect infinite recursion patterns FIRST
                if self._detect_recursion_pattern(response):
                    if self.debug:
                        import sys
                        sys.stdout.write(f"⚠️ RECURSION DETECTED (OOM recovery): Truncating response at {len(response)} chars\n")
                        sys.stdout.flush()
                    response = response[:200]  # Less aggressive truncation
                elif len(response) > 2000:  # Higher safety limit for normal responses
                    response = response[:2000]  # Allow longer responses for extraction

                return response
            except torch.cuda.OutOfMemoryError:
                if self.debug:
                    print("❌ Still OOM after cleanup, using minimal generation")

                # Instead of CPU fallback, use minimal generation
                minimal_config = {
                    "max_new_tokens": 50,  # Very minimal response
                    "temperature": 0.0,
                    "do_sample": False,
                }

                try:
                    response = self.model.chat(
                        self.tokenizer,
                        pixel_values,
                        question,
                        generation_config=minimal_config,
                        history=None,
                        return_history=False
                    )
                except Exception:
                    # Last resort - return a fallback response
                    response = "invoice" if "document" in question.lower() else "NOT_FOUND"

                # CRITICAL: Detect infinite recursion patterns FIRST
                if self._detect_recursion_pattern(response):
                    if self.debug:
                        import sys
                        sys.stdout.write(f"⚠️ RECURSION DETECTED (CPU fallback): Truncating response at {len(response)} chars\n")
                        sys.stdout.flush()
                    response = response[:200]  # Less aggressive truncation
                elif len(response) > 2000:  # Higher safety limit for normal responses
                    response = response[:2000]  # Allow longer responses for extraction

                # Skip moving quantized models (not supported with .to())
                # Quantized models can't be moved with .to() method
                if not (hasattr(self.model, 'quantization_method') or
                        hasattr(self.model, 'is_quantized') or
                        getattr(self.model.config, 'quantization_config', None) is not None):
                    # Only move non-quantized models
                    self.model = self.model.to(self.device)

                return response

        except Exception as e:
            # CRITICAL: Catch all other exceptions to prevent infinite recursion
            if self.debug:
                import sys
                sys.stdout.write(f"❌ CRITICAL: Generation failed with exception: {e}\n")
                sys.stdout.write(f"Exception type: {type(e).__name__}\n")
                sys.stdout.flush()
                import traceback
                traceback.print_exc()

            # Return empty response to trigger fallback logic (like working handler)
            return ""

    def _detect_recursion_pattern(self, response: str) -> bool:
        """Detect infinite recursion patterns in model responses."""
        if not response or len(response) < 50:
            return False

        # Check for repeated text patterns that indicate recursion
        response_lines = response.split('\n')

        # Look for lines that are repeated many times
        line_counts = {}
        for line in response_lines:
            line = line.strip()
            if len(line) > 10:  # Only count substantial lines
                line_counts[line] = line_counts.get(line, 0) + 1

        # If any line appears more than 5 times, it's likely recursion
        max_repetitions = max(line_counts.values()) if line_counts else 0
        if max_repetitions > 5:
            return True

        # Check for the specific "Answer with one of:" repetition pattern
        if "Answer with one of:" in response and response.count("Answer with one of:") > 3:
            return True

        # Check for repeated prompt instructions
        if "INVOICE" in response and response.count("INVOICE") > 10:
            return True

        return False

    def process_single_image(
        self,
        image_path: str,
        custom_prompt: Optional[str] = None,
        custom_max_tokens: Optional[int] = None,
    ) -> dict:
        """Process single image with document-aware extraction using InternVL3."""

        try:
            from pathlib import Path
            start_time = time.time()

            # Memory cleanup
            emergency_cleanup(verbose=False)

            # Use custom prompt if provided, otherwise generate from schema
            if custom_prompt:
                prompt = custom_prompt
                document_type = "CUSTOM"  # Indicate custom prompt usage
            else:
                # Get document-aware prompt using PROPER model-based detection
                detection_result = self.detect_and_classify_document(image_path, verbose=self.debug)
                detected_type = detection_result.get('document_type', 'INVOICE')

                # CRITICAL FIX: Map uppercase detection result to lowercase prompt key
                type_mapping = {
                    'INVOICE': 'invoice',
                    'RECEIPT': 'receipt',
                    'BANK_STATEMENT': 'bank_statement',  # Will be refined below
                    'TRAVEL_EXPENSE': 'travel_expense'
                }
                document_type = type_mapping.get(detected_type, 'invoice')

                # For bank statements, use vision-based structure classification
                if detected_type == 'BANK_STATEMENT':
                    try:
                        # Try different import methods to handle various execution contexts
                        try:
                            from common.vision_bank_statement_classifier import (
                                classify_bank_statement_structure_vision,
                            )
                        except ImportError:
                            import sys
                            from pathlib import Path
                            sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
                            from common.vision_bank_statement_classifier import (
                                classify_bank_statement_structure_vision,
                            )

                        if self.debug:
                            print("🔍 Running vision-based structure classification for bank statement")

                        # Pass self (the processor) which has the load_image method
                        structure_type = classify_bank_statement_structure_vision(
                            image_path,
                            model=self,  # Pass the processor itself which has load_image
                            processor=None,  # InternVL3 doesn't use separate processor
                            verbose=self.debug
                        )

                        # Map structure to specific prompt
                        document_type = f"bank_statement_{structure_type}"

                        if self.debug:
                            print(f"🏗️ Bank statement structure: {structure_type}")
                            print(f"📝 Using prompt key: {document_type}")

                    except Exception as e:
                        if self.debug:
                            print(f"⚠️ Vision classification failed: {e}")
                            print("📝 Falling back to universal prompt")
                        document_type = "universal"

                if self.debug:
                    print(f"📋 DOCUMENT DETECTION RESULT: {detection_result}")
                    print(f"🎯 DETECTED DOCUMENT TYPE: '{detected_type}' → MAPPED TO: '{document_type}'")
                    print(f"📝 LOADING EXTRACTION PROMPT FOR: '{document_type}'")

                prompt = self.get_extraction_prompt(document_type=document_type)

                # Get document-specific field list for accurate processing
                # Normalize document_type by stripping structure suffix for field list lookup
                base_doc_type = document_type.replace('_flat', '').replace('_date_grouped', '')
                document_fields = self.document_field_lists.get(base_doc_type, self.field_list)

                if self.debug:
                    print(f"📋 DOCUMENT-SPECIFIC FIELDS: {len(document_fields)} fields for '{document_type}'")
                    print(f"   Fields: {document_fields[:5]}{'...' if len(document_fields) > 5 else ''}")

            if self.debug:
                # Use direct stdout to bypass Rich console completely
                import sys

                if custom_prompt:
                    sys.stdout.write("📝 Using custom YAML prompt\n")
                    sys.stdout.write(f"🔍 CUSTOM YAML PROMPT ({len(prompt)} chars):\n")
                else:
                    sys.stdout.write(
                        f"📝 Generated prompt for {self.field_count} fields\n"
                    )
                    sys.stdout.write(
                        f"   Fields: {self.field_list[:3]}{'...' if len(self.field_list) > 3 else ''}\n"
                    )
                    sys.stdout.write(
                        f"🔍 DOCUMENT-AWARE PROMPT ({len(prompt)} chars):\n"
                    )
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.write(prompt + "\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

            # Load and preprocess image using InternVL3 pipeline
            pixel_values = self.load_image(image_path)

            # Note: load_image() already handles device placement and dtype conversion
            # based on the actual model requirements, so no additional conversion needed here

            # Prepare question for InternVL3
            question = f"<image>\n{prompt}"

            # Use custom max_tokens if provided (for YAML prompts)
            generation_config = self.generation_config.copy()
            if custom_max_tokens:
                generation_config["max_new_tokens"] = custom_max_tokens

            if self.debug:
                print(f"🖼️  Input tensor shape: {pixel_values.shape}")
                print(
                    f"💭 Generating with max_new_tokens={generation_config['max_new_tokens']}"
                )

            # Generate response using InternVL3 chat method for extraction
            generation_config['is_detection'] = False  # Enable full token limits for extraction
            response = self._resilient_generate(pixel_values, question, **generation_config)

            if self.debug:
                # Use direct stdout to bypass Rich console completely
                import sys

                sys.stdout.write(f"📄 RAW MODEL RESPONSE ({len(response)} chars):\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.write(response + "\n")
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

            # Parse response using hybrid parser (handles both JSON and plain text)
            from common.extraction_parser import hybrid_parse_response

            # Use document-specific fields if we detected a document type, otherwise use full field list
            if 'document_fields' not in locals():
                document_fields = self.field_list

            extracted_data = hybrid_parse_response(
                response, expected_fields=document_fields
            )

            # Apply ExtractionCleaner for value normalization (🧹 CLEANER CALLED output)
            cleaned_data = {}
            for field in document_fields:
                raw_value = extracted_data.get(field, "NOT_FOUND")
                if raw_value != "NOT_FOUND":
                    cleaned_value = self.cleaner.clean_field_value(field, raw_value)
                    cleaned_data[field] = cleaned_value
                else:
                    cleaned_data[field] = "NOT_FOUND"

            # Replace extracted_data with cleaned version
            extracted_data = cleaned_data

            if self.debug:
                # Use direct stdout to bypass Rich console completely
                import sys

                sys.stdout.write("📊 PARSED EXTRACTION RESULTS:\n")
                sys.stdout.write("=" * 80 + "\n")
                for field in document_fields:
                    value = extracted_data.get(field, "NOT_FOUND")
                    status = "✅" if value != "NOT_FOUND" else "❌"
                    sys.stdout.write(f'  {status} {field}: "{value}"\n')
                sys.stdout.write("=" * 80 + "\n")
                sys.stdout.flush()

                found_fields = [
                    k for k, v in extracted_data.items() if v != "NOT_FOUND"
                ]
                print(f"✅ Extracted {len(found_fields)}/{len(document_fields)} fields")
                if found_fields:
                    print(
                        f"   Found: {found_fields[:3]}{'...' if len(found_fields) > 3 else ''}"
                    )

            # Calculate metrics
            extracted_fields_count = len(
                [k for k in extracted_data.keys() if k in self.field_list]
            )
            # Use document-specific field count for accurate metrics
            document_field_count = len(document_fields) if 'document_fields' in locals() else self.field_count
            response_completeness = extracted_fields_count / document_field_count if document_field_count > 0 else 0
            content_coverage = extracted_fields_count / document_field_count if document_field_count > 0 else 0

            # Cleanup with V100 optimizations
            del pixel_values
            emergency_cleanup(verbose=self.debug)

            return {
                "image_name": Path(image_path).name,
                "extracted_data": extracted_data,
                "raw_response": response,
                "processing_time": time.time() - start_time,
                "response_completeness": response_completeness,
                "content_coverage": content_coverage,
                "extracted_fields_count": extracted_fields_count,
                "field_count": document_field_count,
                "document_type": document_type if 'document_type' in locals() else 'unknown'
            }

        except Exception as e:
            if self.debug:
                print(f"❌ Error processing {image_path}: {e}")
                import traceback

                traceback.print_exc()

            # Return error result with dynamic fields
            return {
                "image_name": Path(image_path).name,
                "extracted_data": {field: "NOT_FOUND" for field in self.field_list},
                "raw_response": f"Error: {str(e)}",
                "processing_time": 0,
                "response_completeness": 0,
                "content_coverage": 0,
                "extracted_fields_count": 0,
                "field_count": self.field_count,
            }

    def get_model_info(self) -> dict:
        """Get information about the loaded model for debugging."""
        if self.model is None:
            return {"status": "not_loaded", "model_path": self.model_path}

        return {
            "status": "loaded",
            "model_path": self.model_path,
            "device": str(self.model.device),
            "is_8b_model": self.is_8b_model,
            "field_count": self.field_count,
            "parameter_count": sum(p.numel() for p in self.model.parameters()),
        }