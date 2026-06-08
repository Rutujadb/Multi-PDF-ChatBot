"""Highlighted PDF preview and download for the React dashboard API."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from pdf_storage import get_pdf_path

_PREVIEW_ZOOM = 2.0


class SourcePreviewRequest(BaseModel):
    """Payload describing a cited source to preview."""

    file: str
    page: int | str
    line: Optional[int] = None
    excerpt: str = ""
    highlight_phrases: List[str] = Field(default_factory=list)
    label: str = ""


def _search_page_text(page, text: str) -> list:
    """Search a PDF page for text, trying a few case variants."""
    import fitz

    needle = " ".join((text or "").split()).strip()
    if len(needle) < 4:
        return []

    variants = []
    seen = set()
    for candidate in (needle, needle.lower(), needle.title()):
        key = candidate.casefold()
        if key not in seen:
            seen.add(key)
            variants.append(candidate)

    rects = []
    for variant in variants:
        found = page.search_for(variant)
        if not found:
            found = page.search_for(variant, quads=True)
        for item in found:
            rect = fitz.Rect(item) if not isinstance(item, fitz.Rect) else item
            rects.append(rect)
    return rects


def _highlight_rects(page, rects: list, limit: int = 6) -> bool:
    """Add highlight annotations for the given rectangles."""
    if not rects:
        return False

    seen_boxes = set()
    added = 0
    for rect in rects[:limit]:
        box = (round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1))
        if box in seen_boxes:
            continue
        seen_boxes.add(box)
        annot = page.add_highlight_annot(rect)
        annot.set_colors(stroke=(1.0, 0.84, 0.0))
        annot.set_opacity(0.45)
        annot.update()
        added += 1
    return added > 0


def _highlight_by_line(page, line_num: int) -> bool:
    """Add a highlight annotation for a 1-based line number on the page."""
    import fitz

    if not line_num or line_num < 1:
        return False

    current_line = 0
    page_dict = page.get_text("dict")
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            current_line += 1
            if current_line == int(line_num):
                rect = fitz.Rect(line["bbox"])
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=(1.0, 0.84, 0.0))
                annot.set_opacity(0.45)
                annot.update()
                return True
    return False


def build_highlighted_page_pdf(
    pdf_path: Path,
    page_num: int,
    excerpt: str,
    line_num: Optional[int] = None,
    highlight_phrases: Optional[List[str]] = None,
) -> Optional[bytes]:
    """Return a single-page PDF with answer-aligned highlights."""
    try:
        import fitz
    except ImportError:
        return None

    try:
        doc = fitz.open(pdf_path)
        page_idx = max(0, min(int(page_num) - 1, len(doc) - 1))
        page = doc[page_idx]
        highlighted = False
        excerpt_clean = " ".join((excerpt or "").split())

        unique_phrases: List[str] = []
        seen_phrases = set()
        for phrase in highlight_phrases or []:
            cleaned = " ".join((phrase or "").split()).strip()
            key = cleaned.casefold()
            if len(cleaned) >= 4 and key not in seen_phrases:
                seen_phrases.add(key)
                unique_phrases.append(cleaned)
        unique_phrases.sort(key=len, reverse=True)

        for phrase in unique_phrases:
            rects = _search_page_text(page, phrase)
            if _highlight_rects(page, rects):
                highlighted = True

        if not highlighted and excerpt_clean:
            for fragment in unique_phrases:
                if fragment.lower() in excerpt_clean.lower():
                    rects = _search_page_text(page, fragment)
                    if _highlight_rects(page, rects):
                        highlighted = True
                        break

        if not highlighted and excerpt_clean:
            for length in (180, 140, 100, 70, 45):
                fragment = excerpt_clean[:length].strip()
                if len(fragment) < 12:
                    break
                rects = _search_page_text(page, fragment)
                if _highlight_rects(page, rects):
                    highlighted = True
                    break

        if not highlighted and excerpt_clean:
            for start in range(0, max(1, len(excerpt_clean) - 40), 40):
                fragment = excerpt_clean[start : start + 100].strip()
                if len(fragment) < 20:
                    continue
                rects = _search_page_text(page, fragment)
                if _highlight_rects(page, rects):
                    highlighted = True
                    break

        if not highlighted and line_num:
            highlighted = _highlight_by_line(page, line_num)

        single_page = fitz.open()
        single_page.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
        pdf_bytes = single_page.tobytes()
        single_page.close()
        doc.close()
        return pdf_bytes
    except Exception:
        return None


def _pdf_bytes_to_png(pdf_bytes: bytes, zoom: float = _PREVIEW_ZOOM) -> Optional[bytes]:
    """Rasterize the first page of a single-page PDF to PNG bytes."""
    try:
        import fitz
    except ImportError:
        return None

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if doc.page_count == 0:
            doc.close()
            return None
        page = doc[0]
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
        png_bytes = pix.tobytes("png")
        doc.close()
        return png_bytes
    except Exception:
        return None


def build_highlighted_page_png(source_item: Dict[str, Any]) -> Optional[bytes]:
    """Return PNG bytes for a cited source page."""
    pdf_bytes = build_highlighted_page_pdf_bytes(source_item)
    if not pdf_bytes:
        return None
    return _pdf_bytes_to_png(pdf_bytes)


def build_highlighted_page_pdf_bytes(source_item: Dict[str, Any]) -> Optional[bytes]:
    """Return highlighted single-page PDF bytes for a cited source."""
    filename = source_item.get("file", "")
    pdf_path = get_pdf_path(filename)
    if pdf_path is None:
        return None

    page = source_item.get("page", 1)
    excerpt = source_item.get("excerpt", "")
    line_num = source_item.get("line")
    highlight_phrases = source_item.get("highlight_phrases") or []

    pdf_bytes = build_highlighted_page_pdf(
        pdf_path,
        page,
        excerpt,
        line_num=line_num,
        highlight_phrases=highlight_phrases,
    )
    if pdf_bytes:
        return pdf_bytes
    return pdf_path.read_bytes()


def _safe_download_name(source_item: Dict[str, Any]) -> str:
    """Build a filesystem-safe download filename."""
    stem = Path(source_item.get("file", "source.pdf")).stem
    page = source_item.get("page", 1)
    return f"{stem}-p{page}-highlighted.pdf"


def render_source_preview_image(body: SourcePreviewRequest) -> Response:
    """Return a PNG preview of the cited PDF page."""
    payload = body.model_dump()
    png_bytes = build_highlighted_page_png(payload)
    if not png_bytes:
        raise HTTPException(
            status_code=404,
            detail="Source PDF not found. Re-upload and process the document.",
        )
    return Response(content=png_bytes, media_type="image/png")


def render_source_download(body: SourcePreviewRequest) -> Response:
    """Return the highlighted single-page PDF as a download."""
    payload = body.model_dump()
    pdf_bytes = build_highlighted_page_pdf_bytes(payload)
    if not pdf_bytes:
        raise HTTPException(
            status_code=404,
            detail="Source PDF not found. Re-upload and process the document.",
        )
    filename = _safe_download_name(payload)
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
