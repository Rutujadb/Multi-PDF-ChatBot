"""Generate text captions for extracted PDF images using a Gemma vision model."""

from __future__ import annotations

import base64
import logging
import mimetypes
from pathlib import Path
from typing import Optional, Tuple

from langchain_core.messages import HumanMessage

from config import (
    GOOGLE_API_KEY,
    IMAGE_CAPTION_MODEL,
    IMAGE_CAPTION_PROVIDER,
    IMAGE_CAPTION_ENABLED,
    OPENROUTER_API_KEY,
    get_image_caption_model_attempts,
)
from utils import is_rate_limit_error

logger = logging.getLogger(__name__)


_CAPTION_PROMPT = """Describe this image extracted from a PDF page for a document Q&A system.
Include visible text, chart titles, axis labels, table headers, diagram labels, and the main subject.
Be factual and concise. Do not invent content that is not visible in the image."""

_INVALID_CAPTION_MARKERS = (
    "rate limit",
    "rate-limited",
    "api key",
    "unauthorized",
    "429",
    "error:",
    "failed to",
    "error generating answer",
)


def _image_mime_type(path: Path) -> str:
    """Return a MIME type for an on-disk image file."""
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "image/png"


def _image_data_url(path: Path) -> str:
    """Encode an image file as a data URL for multimodal chat APIs."""
    mime = _image_mime_type(path)
    encoded = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _is_invalid_caption_text(text: str) -> bool:
    """Return True when caption text looks like an API failure, not a description."""
    body = (text or "").strip()
    if len(body) < 10:
        return True
    lowered = body.lower()
    return any(marker in lowered for marker in _INVALID_CAPTION_MARKERS)


def get_caption_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
):
    """Return the configured vision-capable LLM for image captioning."""
    from rag_chain import get_llm

    chosen_provider = (provider or IMAGE_CAPTION_PROVIDER).strip().lower()
    chosen_model = model or IMAGE_CAPTION_MODEL

    if chosen_provider == "openrouter" and not OPENROUTER_API_KEY:
        chosen_provider = "gemini"
    if chosen_provider == "gemini" and not GOOGLE_API_KEY and OPENROUTER_API_KEY:
        chosen_provider = "openrouter"

    logger.info("Caption LLM: provider=%s, model=%s", chosen_provider, chosen_model)
    return get_llm(llm_provider=chosen_provider, llm_model=chosen_model)


def caption_image(
    image_path: Path,
    source: str = "",
    page_label: str = "",
    provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Tuple[str, str]:
    """Describe one extracted PDF image with a vision model.

    Tries the primary caption model first, then a configured free fallback when
    rate limits or retryable failures occur.

    Args:
        image_path: Path to the saved image file.
        source: Optional PDF filename for prompt context.
        page_label: Optional 1-based page label for prompt context.
        provider: Optional provider override for the primary attempt only.
        model: Optional model override for the primary attempt only.

    Returns:
        Tuple of ``(caption_text, caption_model)``. Both are empty when captioning
        is disabled or every attempt fails.
    """
    if not IMAGE_CAPTION_ENABLED:
        return "", ""

    path = Path(image_path)
    if not path.is_file():
        logger.warning("Image file not found for captioning: %s", image_path)
        return "", ""

    logger.info("Captioning image: %s (source=%s, page=%s)", path.name, source, page_label)

    context_bits = []
    if source:
        context_bits.append(f"Source PDF: {source}")
    if page_label:
        context_bits.append(f"Page: {page_label}")
    context_line = "\n".join(context_bits)
    prompt = _CAPTION_PROMPT
    if context_line:
        prompt = f"{context_line}\n\n{_CAPTION_PROMPT}"

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": _image_data_url(path)}},
        ]
    )

    if provider or model:
        attempts = [
            (
                (provider or IMAGE_CAPTION_PROVIDER).strip().lower(),
                model or IMAGE_CAPTION_MODEL,
            )
        ]
        for prov, mod in get_image_caption_model_attempts():
            if (prov, mod) not in attempts:
                attempts.append((prov, mod))
    else:
        attempts = get_image_caption_model_attempts()

    if not attempts:
        logger.warning("No caption models configured with valid API keys")
        return "", ""

    last_error: Optional[Exception] = None
    for index, (prov, mod) in enumerate(attempts):
        is_last = index == len(attempts) - 1
        try:
            response = get_caption_llm(provider=prov, model=mod).invoke([message])
            text = getattr(response, "content", str(response)).strip()
            if _is_invalid_caption_text(text):
                if is_rate_limit_error(text) and not is_last:
                    logger.warning(
                        "Caption rate-limited on %s/%s; trying fallback model",
                        prov,
                        mod,
                    )
                    continue
                logger.warning(
                    "Invalid caption response from %s/%s: %s",
                    prov,
                    mod,
                    text[:120],
                )
                if not is_last:
                    continue
                return "", ""

            logger.info(
                "Caption generated for %s using %s/%s (%d chars)",
                path.name,
                prov,
                mod,
                len(text),
            )
            return text, mod
        except Exception as exc:
            last_error = exc
            if is_rate_limit_error(exc) and not is_last:
                logger.warning(
                    "Caption API rate-limited on %s/%s; trying fallback model",
                    prov,
                    mod,
                )
                continue
            logger.error(
                "Caption API call failed for %s on %s/%s: %s",
                path.name,
                prov,
                mod,
                exc,
                exc_info=True,
            )
            if not is_last:
                continue

    if last_error is not None:
        logger.error(
            "All caption attempts failed for %s: %s",
            path.name,
            last_error,
        )
    return "", ""
