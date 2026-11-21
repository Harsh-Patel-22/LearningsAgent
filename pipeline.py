"""
PHASE 1 — OCR PIPELINE
Cognitive Memory OS

Multi-stage OCR system designed for handwritten personal diary extraction.
Architecture:
  1. Image preprocessing (deskew, denoise, contrast normalization)
  2. Primary OCR (PaddleOCR for handwriting)
  3. LLM vision correction pass (Claude/GPT-4V)
  4. Visual annotation extraction (underlines, stars, crossed-out text)
  5. Confidence scoring + human review queuing
"""

from __future__ import annotations
import asyncio
import base64
import hashlib
import io
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class OCRConfig:
    min_confidence_for_llm_correction: float = 0.75
    min_confidence_for_auto_accept: float = 0.90
    max_image_size_px: int = 4096
    dpi_targets: list = None
    llm_model: str = "claude-3-5-sonnet-20241022"
    enable_visual_annotation_extraction: bool = True

    def __post_init__(self):
        if self.dpi_targets is None:
            self.dpi_targets = [150, 300]


# ─────────────────────────────────────────────
# IMAGE PREPROCESSOR
# ─────────────────────────────────────────────

class ImagePreprocessor:
    """
    Prepares diary images for OCR.
    Critical for handwritten text — quality dramatically affects accuracy.
    """

    def preprocess(self, image_bytes: bytes) -> list[bytes]:
        """
        Returns list of preprocessed image variants (different enhancements).
        We try multiple and pick the one that OCR is most confident about.
        """
        try:
            from PIL import Image, ImageFilter, ImageEnhance, ImageOps
            import numpy as np
        except ImportError:
            logger.warning("PIL/numpy not available — skipping image preprocessing")
            return [image_bytes]

        img = Image.open(io.BytesIO(image_bytes))
        variants = []

        # Variant 1: Standard normalize
        v1 = self._normalize(img)
        variants.append(self._to_bytes(v1))

        # Variant 2: High contrast + sharpen (good for faded ink)
        v2 = self._high_contrast(img)
        variants.append(self._to_bytes(v2))

        # Variant 3: Adaptive threshold (good for uneven lighting)
        v3 = self._adaptive_threshold(img)
        if v3:
            variants.append(self._to_bytes(v3))

        return variants

    def _normalize(self, img):
        from PIL import ImageOps, ImageFilter
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = ImageOps.autocontrast(img, cutoff=2)
        img = self._deskew(img)
        return img

    def _high_contrast(self, img):
        from PIL import ImageEnhance
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = ImageEnhance.Contrast(img).enhance(1.8)
        img = ImageEnhance.Sharpness(img).enhance(2.0)
        return img

    def _adaptive_threshold(self, img):
        try:
            import cv2
            import numpy as np
            gray = cv2.cvtColor(
                __import__('numpy').array(img.convert("RGB")),
                cv2.COLOR_RGB2GRAY
            )
            binary = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )
            from PIL import Image
            return Image.fromarray(binary).convert("RGB")
        except ImportError:
            return None

    def _deskew(self, img):
        """Simple deskew using Hough transform via OpenCV if available."""
        try:
            import cv2
            import numpy as np
            gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            if lines is None:
                return img
            angles = []
            for line in lines[:20]:
                rho, theta = line[0]
                angle = (theta - np.pi / 2) * 180 / np.pi
                if abs(angle) < 10:
                    angles.append(angle)
            if not angles:
                return img
            median_angle = float(np.median(angles))
            if abs(median_angle) > 0.5:
                return img.rotate(-median_angle, expand=True, fillcolor="white")
            return img
        except ImportError:
            return img

    def _to_bytes(self, img, format: str = "PNG") -> bytes:
        buf = io.BytesIO()
        img.save(buf, format=format)
        return buf.getvalue()


# ─────────────────────────────────────────────
# PRIMARY OCR ENGINE (PaddleOCR)
# ─────────────────────────────────────────────

class PaddleOCREngine:
    """
    Primary OCR using PaddleOCR PP-OCRv4.
    Best open-source option for handwritten text.
    Falls back to Tesseract if PaddleOCR unavailable.
    """

    def __init__(self):
        self._ocr = None
        self._engine_type = "none"
        self._init_engine()

    def _init_engine(self):
        try:
            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                use_gpu=False,
                show_log=False
            )
            self._engine_type = "paddleocr"
            logger.info("PaddleOCR initialized successfully")
        except ImportError:
            logger.warning("PaddleOCR not available, trying Tesseract")
            try:
                import pytesseract
                pytesseract.get_tesseract_version()
                self._engine_type = "tesseract"
                logger.info("Tesseract initialized as fallback")
            except Exception:
                logger.error("No OCR engine available — OCR will be skipped")

    def extract(self, image_bytes: bytes) -> OCRResult:
        """Extract text with per-line confidence scores."""
        if self._engine_type == "paddleocr":
            return self._paddle_extract(image_bytes)
        elif self._engine_type == "tesseract":
            return self._tesseract_extract(image_bytes)
        else:
            return OCRResult(text="", confidence=0.0, engine="none", lines=[])

    def _paddle_extract(self, image_bytes: bytes) -> OCRResult:
        try:
            import numpy as np
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes))
            img_array = np.array(img)

            result = self._ocr.ocr(img_array, cls=True)
            if not result or not result[0]:
                return OCRResult(text="", confidence=0.0, engine="paddleocr", lines=[])

            lines = []
            total_conf = 0.0

            for item in result[0]:
                bbox, (text, confidence) = item
                lines.append(OCRLine(text=text, confidence=confidence, bbox=bbox))
                total_conf += confidence

            full_text = "\n".join(line.text for line in lines)
            avg_confidence = total_conf / len(lines) if lines else 0.0

            return OCRResult(
                text=full_text,
                confidence=round(avg_confidence, 4),
                engine="paddleocr",
                lines=lines
            )
        except Exception as e:
            logger.error(f"PaddleOCR extraction error: {e}")
            return OCRResult(text="", confidence=0.0, engine="paddleocr_error", lines=[])

    def _tesseract_extract(self, image_bytes: bytes) -> OCRResult:
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes))
            data = pytesseract.image_to_data(
                img,
                output_type=pytesseract.Output.DICT,
                config="--oem 3 --psm 6"
            )

            lines = []
            text_parts = []
            confidences = []

            for i, word in enumerate(data["text"]):
                word = word.strip()
                if not word:
                    continue
                conf = data["conf"][i]
                if conf == -1:
                    continue
                conf_normalized = conf / 100.0
                text_parts.append(word)
                confidences.append(conf_normalized)

            full_text = " ".join(text_parts)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            return OCRResult(
                text=full_text,
                confidence=round(avg_conf, 4),
                engine="tesseract",
                lines=lines
            )
        except Exception as e:
            logger.error(f"Tesseract extraction error: {e}")
            return OCRResult(text="", confidence=0.0, engine="tesseract_error", lines=[])


@dataclass
class OCRLine:
    text: str
    confidence: float
    bbox: Optional[list] = None


@dataclass
class OCRResult:
    text: str
    confidence: float
    engine: str
    lines: list


# ─────────────────────────────────────────────
# LLM VISION CORRECTION PASS
# ─────────────────────────────────────────────

class LLMOCRCorrector:
    """
    Uses a vision LLM to correct OCR errors in handwritten diary text.
    This is the most impactful step for quality.

    Strategy:
    - Send (original image + raw OCR) to Claude/GPT-4V
    - Model uses visual context to fix OCR errors
    - Preserve emotional language EXACTLY — never substitute synonyms
    - Extract date if visible in image
    """

    # The core correction prompt — designed specifically for personal diary content
    CORRECTION_PROMPT = """You are correcting OCR output from a handwritten personal diary page.

ORIGINAL IMAGE: [attached]
RAW OCR OUTPUT:
---
{ocr_text}
---

Your task:
1. Correct OCR errors using the image as ground truth
2. Reconstruct unclear words from surrounding context
3. PRESERVE emotional language exactly — do not substitute synonyms or soften anything
4. PRESERVE the writer's voice, punctuation style, and sentence structure
5. If a word is truly illegible, write [illegible]
6. If there are crossed-out words, write them as: ~~crossed out text~~
7. Note any margin annotations separately at the end as: [MARGIN: text]
8. Note any underlined passages as: **underlined text**
9. Note heavily emphasized passages (multiple underlines, capitals, stars) as: ***EMPHASIZED: text***
10. Handle code-mixed/transliterated GujLish. The page may contain Gujarati phrases or words written in the English script. Correct spelling/transliteration errors using the image as a guide, and ensure the transliterated Gujarati words are preserved exactly as written rather than translated or anglicized.

Output ONLY the corrected text. No commentary, no preamble."""

    DATE_EXTRACTION_PROMPT = """Look at this diary page image.
Extract the date if visible anywhere on the page (top, bottom, margin).
Return JSON: {"date": "YYYY-MM-DD or null", "date_string": "exact text found or null", "confidence": 0.0-1.0}
If no date visible, return: {"date": null, "date_string": null, "confidence": 0.1}"""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model

    async def correct(
        self,
        image_bytes: bytes,
        ocr_text: str,
        image_mime: str = "image/jpeg"
    ) -> LLMCorrectionResult:
        """Run the vision correction pass."""
        if not self.api_key or not image_bytes:
            return LLMCorrectionResult(
                corrected_text=ocr_text,
                date_extracted=None,
                date_confidence=0.1,
                visual_annotations=[],
                model_used="none"
            )

        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        # Run correction and date extraction in parallel
        corrected_text, date_info = await asyncio.gather(
            self._run_correction(image_b64, image_mime, ocr_text),
            self._extract_date(image_b64, image_mime)
        )

        annotations = self._parse_visual_annotations(corrected_text)
        clean_text = self._strip_annotation_markers(corrected_text)

        return LLMCorrectionResult(
            corrected_text=clean_text,
            raw_corrected_text=corrected_text,
            date_extracted=date_info.get("date"),
            date_string=date_info.get("date_string"),
            date_confidence=date_info.get("confidence", 0.1),
            visual_annotations=annotations,
            model_used=self.model
        )

    async def _run_correction(self, image_b64: str, image_mime: str, ocr_text: str) -> str:
        prompt = self.CORRECTION_PROMPT.format(ocr_text=ocr_text)
        return await self._call_vision_api(image_b64, image_mime, prompt)

    async def _extract_date(self, image_b64: str, image_mime: str) -> dict:
        raw = await self._call_vision_api(image_b64, image_mime, self.DATE_EXTRACTION_PROMPT)
        try:
            # Strip markdown code fences if present
            raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return {"date": None, "date_string": None, "confidence": 0.1}

    async def _call_vision_api(
        self, image_b64: str, image_mime: str, prompt: str
    ) -> str:
        """Call the Gemini OpenAI-compatible vision API."""
        async with httpx.AsyncClient(timeout=90.0) as client:
            payload = {
                "model": "gemini-3.1-flash-lite" if ("claude" in self.model or "gpt" in self.model or self.model == "gemini-3.1-flash") else self.model,
                "max_tokens": 4096,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_mime};base64,{image_b64}"
                                }
                            }
                        ]
                    }
                ]
            }
            resp = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    def _parse_visual_annotations(self, text: str) -> list[str]:
        """Extract visual annotation markers from corrected text."""
        annotations = []
        if "~~" in text:
            crossed = re.findall(r"~~(.+?)~~", text)
            annotations.extend([f"crossed_out: {c}" for c in crossed])
        if "**" in text:
            underlined = re.findall(r"\*\*(?!\*)(.+?)(?<!\*)\*\*", text)
            annotations.extend([f"underlined: {u}" for u in underlined])
        if "***" in text:
            emphasized = re.findall(r"\*\*\*EMPHASIZED: (.+?)\*\*\*", text)
            annotations.extend([f"emphasized: {e}" for e in emphasized])
        if "[MARGIN:" in text:
            margins = re.findall(r"\[MARGIN: (.+?)\]", text)
            annotations.extend([f"margin_note: {m}" for m in margins])
        return annotations

    def _strip_annotation_markers(self, text: str) -> str:
        """Remove markdown-style annotation markers for clean storage."""
        text = re.sub(r"\*\*\*EMPHASIZED: (.+?)\*\*\*", r"\1", text)
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"~~.+?~~", "", text)
        text = re.sub(r"\[MARGIN: .+?\]", "", text)
        return text.strip()


@dataclass
class LLMCorrectionResult:
    corrected_text: str
    date_extracted: Optional[str]
    date_confidence: float
    visual_annotations: list
    model_used: str
    raw_corrected_text: Optional[str] = None
    date_string: Optional[str] = None


# ─────────────────────────────────────────────
# SEMANTIC CHUNKER
# ─────────────────────────────────────────────

class SemanticChunker:
    """
    Splits diary entries at natural semantic boundaries.
    NOT fixed character/token size — that destroys emotional coherence.

    A new passage starts when:
    - Topic shifts significantly
    - Emotional register changes
    - New time or event begins
    - Writer moves from narrative to reflection
    """

    CHUNKING_PROMPT = """Split this diary entry into natural semantic passages.

A new passage starts when:
- The topic shifts significantly
- The emotional register changes
- A new time or life event begins
- The writer moves from narrative to reflection or analysis
- A new person or situation is introduced

Rules:
- Preserve ALL text exactly — do not modify or summarize
- Add [PASSAGE_BREAK] between passages
- Each passage should be 100–400 words where possible
- Very short diary entries (under 200 words) = single passage
- Minimum 2 sentences per passage unless the break is essential

DIARY ENTRY:
---
{text}
---

Output the full text with [PASSAGE_BREAK] markers inserted. Nothing else."""

    def __init__(self, api_key: str, model: str = "claude-3-5-sonnet-20241022"):
        self.api_key = api_key
        self.model = model

    async def chunk(self, text: str) -> list[str]:
        """Split text into semantic passages."""
        if len(text.split()) < 150:
            # Short entries: single passage
            return [text.strip()] if text.strip() else []

        try:
            chunks = await self._llm_chunk(text)
            if not chunks:
                # Fallback: paragraph-based splitting
                return self._paragraph_fallback(text)
            return chunks
        except Exception as e:
            logger.error(f"Semantic chunking error: {e}")
            return self._paragraph_fallback(text)

    async def _llm_chunk(self, text: str) -> list[str]:
        prompt = self.CHUNKING_PROMPT.format(text=text)
        async with httpx.AsyncClient(timeout=90.0) as client:
            payload = {
                "model": "gemini-3.1-flash-lite" if ("claude" in self.model or "gpt" in self.model or self.model == "gemini-3.1-flash") else self.model,
                "max_tokens": 4096,
                "messages": [{"role": "user", "content": prompt}]
            }
            resp = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json=payload
            )
            resp.raise_for_status()
            result = resp.json()["choices"][0]["message"]["content"]

        passages = [p.strip() for p in result.split("[PASSAGE_BREAK]")]
        return [p for p in passages if len(p.split()) > 20]

    def _paragraph_fallback(self, text: str) -> list[str]:
        """Simple paragraph-based fallback."""
        paragraphs = re.split(r"\n{2,}", text)
        chunks = []
        current = []
        current_wc = 0
        for para in paragraphs:
            wc = len(para.split())
            if current_wc + wc > 400 and current:
                chunks.append("\n\n".join(current))
                current = [para]
                current_wc = wc
            else:
                current.append(para)
                current_wc += wc
        if current:
            chunks.append("\n\n".join(current))
        return [c for c in chunks if c.strip()]


# ─────────────────────────────────────────────
# MASTER OCR ORCHESTRATOR
# ─────────────────────────────────────────────

class OCROrchestrator:
    """
    Orchestrates the full OCR pipeline for a single diary image.
    Returns a fully processed result ready for memory extraction.
    """

    def __init__(self, config: OCRConfig, api_key: str):
        self.config = config
        self.preprocessor = ImagePreprocessor()
        self.primary_ocr = PaddleOCREngine()
        self.llm_corrector = LLMOCRCorrector(api_key=api_key, model=config.llm_model)
        self.chunker = SemanticChunker(api_key=api_key, model=config.llm_model)

    async def process_image(
        self,
        image_bytes: bytes,
        source_filename: Optional[str] = None,
        hint_date: Optional[str] = None
    ) -> OCRProcessingResult:
        """
        Full pipeline:
        1. Preprocess image variants
        2. Run primary OCR on all variants, pick best
        3. LLM correction pass if below confidence threshold
        4. Semantic chunking
        5. Return fully processed result
        """
        file_hash = hashlib.sha256(image_bytes).hexdigest()
        logger.info(f"Processing image {source_filename or file_hash[:12]}")

        # Step 1: Preprocess
        image_variants = self.preprocessor.preprocess(image_bytes)
        logger.debug(f"Generated {len(image_variants)} image variants")

        # Step 2: Primary OCR on all variants
        ocr_results = [self.primary_ocr.extract(variant) for variant in image_variants]
        best_ocr = max(ocr_results, key=lambda r: r.confidence)
        logger.info(f"Best OCR confidence: {best_ocr.confidence:.3f} via {best_ocr.engine}")

        # Step 3: LLM correction if needed
        llm_result = None
        needs_review = best_ocr.confidence < self.config.min_confidence_for_auto_accept

        if best_ocr.confidence < self.config.min_confidence_for_llm_correction:
            logger.info("Running LLM vision correction pass")
            # Use original (unprocessed) image for LLM — it sees better than preprocessed
            llm_result = await self.llm_corrector.correct(
                image_bytes=image_bytes,
                ocr_text=best_ocr.text
            )
            final_text = llm_result.corrected_text
            final_date = llm_result.date_extracted
            final_date_confidence = llm_result.date_confidence
            visual_annotations = llm_result.visual_annotations
        else:
            # High-confidence OCR — still try to extract date via LLM
            logger.info("OCR confidence high — skipping full correction, extracting date only")
            final_text = best_ocr.text
            visual_annotations = []
            # Attempt date extraction even for high-confidence OCR
            try:
                image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
                date_info = await self.llm_corrector._extract_date(image_b64, "image/jpeg")
                final_date = date_info.get("date")
                final_date_confidence = date_info.get("confidence", 0.5)
            except Exception:
                final_date = hint_date
                final_date_confidence = 0.3 if hint_date else 0.1

        # Step 4: Semantic chunking
        passages = await self.chunker.chunk(final_text)
        logger.info(f"Split into {len(passages)} semantic passages")

        return OCRProcessingResult(
            file_hash=file_hash,
            source_filename=source_filename or "unknown",
            raw_ocr_text=best_ocr.text,
            ocr_confidence=best_ocr.confidence,
            ocr_engine=best_ocr.engine,
            corrected_text=final_text,
            was_llm_corrected=llm_result is not None,
            semantic_passages=passages,
            date_extracted=final_date,
            date_confidence=final_date_confidence,
            visual_annotations=visual_annotations,
            needs_human_review=needs_review,
            processing_notes=self._generate_notes(best_ocr, llm_result)
        )

    def _generate_notes(self, ocr: OCRResult, llm: Optional[LLMCorrectionResult]) -> list[str]:
        notes = []
        if ocr.confidence < 0.5:
            notes.append(f"LOW OCR confidence: {ocr.confidence:.2f} — review recommended")
        if llm and llm.date_confidence < 0.5:
            notes.append("Date could not be reliably extracted — manual review needed")
        if ocr.engine in ("paddleocr_error", "tesseract_error", "none"):
            notes.append(f"OCR engine error: {ocr.engine}")
        return notes


@dataclass
class OCRProcessingResult:
    file_hash: str
    source_filename: str
    raw_ocr_text: str
    ocr_confidence: float
    ocr_engine: str
    corrected_text: str
    was_llm_corrected: bool
    semantic_passages: list[str]
    date_extracted: Optional[str]
    date_confidence: float
    visual_annotations: list[str]
    needs_human_review: bool
    processing_notes: list[str]
