"""Right-side PDF source viewer with paragraph highlighting for Streamlit."""

from __future__ import annotations

import base64
import html
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
import streamlit.components.v1 as components

from pdf_storage import get_pdf_path

_PDF_PANEL_CACHE: dict[tuple, tuple[str, str, str, str]] = {}
_PREVIEW_ZOOM = 2.0

PANEL_STYLE = """
#mpdf-source-panel {
  position: fixed !important;
  top: 0 !important;
  right: 0 !important;
  width: 50vw !important;
  height: 100vh !important;
  max-width: 50vw !important;
  background: #ffffff !important;
  border-left: 1px solid #e5e7eb !important;
  box-shadow: -16px 0 48px rgba(17, 24, 39, 0.18) !important;
  display: flex !important;
  flex-direction: column !important;
  z-index: 999999 !important;
  transform: translateX(100%);
  transition: transform 0.28s ease-out;
  font-family: "Segoe UI", system-ui, sans-serif !important;
  color: #111827 !important;
  overflow: hidden !important;
}
#mpdf-source-panel.mpdf-open {
  transform: translateX(0);
}
#mpdf-source-panel .mpdf-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 16px 18px;
  border-bottom: 1px solid #e5e7eb;
  background: #f9fafb;
  flex-shrink: 0;
}
#mpdf-source-panel .mpdf-title-wrap {
  min-width: 0;
  flex: 1;
}
#mpdf-source-panel .mpdf-title {
  font-size: 15px;
  font-weight: 700;
  line-height: 1.35;
  word-break: break-word;
}
#mpdf-source-panel .mpdf-subtitle {
  margin-top: 4px;
  font-size: 12px;
  color: #6b7280;
  word-break: break-word;
}
#mpdf-source-panel .mpdf-close {
  border: none;
  background: #111827;
  color: #ffffff;
  font-size: 13px;
  font-weight: 700;
  padding: 10px 16px;
  border-radius: 8px;
  cursor: pointer;
  flex-shrink: 0;
}
#mpdf-source-panel .mpdf-close:hover {
  background: #2563eb;
}
#mpdf-source-panel .mpdf-body {
  flex: 1 1 auto;
  min-height: 0;
  background: #f3f4f6;
  position: relative;
}
#mpdf-source-panel .mpdf-body embed,
#mpdf-source-panel .mpdf-body object,
#mpdf-source-panel .mpdf-body iframe,
#mpdf-source-panel .mpdf-body img {
  display: block;
  width: 100% !important;
  height: auto !important;
  border: none !important;
  background: #ffffff;
}
#mpdf-source-panel .mpdf-preview-scroll {
  width: 100%;
  height: 100%;
  overflow: auto;
  background: #ffffff;
}
#mpdf-source-panel .mpdf-legend {
  padding: 10px 18px;
  border-top: 1px solid #e5e7eb;
  font-size: 12px;
  color: #4b5563;
  background: #fffbeb;
  flex-shrink: 0;
  word-break: break-word;
}
body.mpdf-panel-open [data-testid="stAppViewContainer"] {
  margin-right: 50vw !important;
  transition: margin-right 0.28s ease-out;
}
"""


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
    """Return a single-page PDF with the source excerpt highlighted.

    Uses PyMuPDF to locate answer-aligned phrases and the most relevant excerpt
    sentences on the page before falling back to line-based highlighting.

    Args:
        pdf_path: Path to the original uploaded PDF.
        page_num: 1-based page number shown in citations.
        excerpt: Retrieved chunk text to locate and highlight.
        line_num: Optional 1-based line number within the page.
        highlight_phrases: Answer-aligned phrases to search for first.

    Returns:
        PDF bytes for the highlighted page, or ``None`` on failure.
    """
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


def build_highlighted_page_png(
    pdf_path: Path,
    page_num: int,
    excerpt: str,
    line_num: Optional[int] = None,
    highlight_phrases: Optional[List[str]] = None,
) -> Optional[bytes]:
    """Return PNG bytes of a highlighted PDF page for browser preview."""
    pdf_bytes = build_highlighted_page_pdf(
        pdf_path,
        page_num,
        excerpt,
        line_num=line_num,
        highlight_phrases=highlight_phrases,
    )
    if not pdf_bytes:
        return None
    return _pdf_bytes_to_png(pdf_bytes)


def _prepare_panel_preview(source_item: Dict[str, Any]) -> tuple[str, str, str, str]:
    """Build base64 PNG preview data and escaped labels for the viewer panel."""
    filename = source_item.get("file", "Unknown")
    page = source_item.get("page", "?")
    label = source_item.get("label") or f"{filename} - p.{page}"
    excerpt = source_item.get("excerpt", "")
    line_num = source_item.get("line")
    highlight_phrases = source_item.get("highlight_phrases") or []

    pdf_path = get_pdf_path(filename)
    if pdf_path is None:
        return "", html.escape(str(filename)), html.escape(str(label)), html.escape(
            excerpt[:240]
        )

    phrases_key = "|".join(highlight_phrases[:5])
    cache_key = (
        filename,
        str(page),
        str(line_num),
        excerpt[:160],
        phrases_key,
        pdf_path.stat().st_mtime,
    )
    if cache_key in _PDF_PANEL_CACHE:
        return _PDF_PANEL_CACHE[cache_key]

    png_bytes = build_highlighted_page_png(
        pdf_path,
        page,
        excerpt,
        line_num=line_num,
        highlight_phrases=highlight_phrases,
    )
    if not png_bytes:
        try:
            import fitz

            doc = fitz.open(pdf_path)
            page_idx = max(0, min(int(page) - 1, len(doc) - 1))
            single = fitz.open()
            single.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
            fallback_pdf = single.tobytes()
            single.close()
            doc.close()
            png_bytes = _pdf_bytes_to_png(fallback_pdf)
        except Exception:
            png_bytes = None

    if not png_bytes:
        return "", html.escape(str(filename)), html.escape(str(label)), html.escape(
            excerpt[:240]
        )

    png_b64 = base64.b64encode(png_bytes).decode("ascii")
    result = (
        png_b64,
        html.escape(str(filename)),
        html.escape(str(label)),
        html.escape(excerpt[:240]),
    )
    _PDF_PANEL_CACHE[cache_key] = result
    return result


def _prepare_panel_pdf(source_item: Dict[str, Any]) -> tuple[str, str, str, str]:
    """Build base64 PDF data for optional download (not used for display)."""
    filename = source_item.get("file", "Unknown")
    page = source_item.get("page", "?")
    label = source_item.get("label") or f"{filename} - p.{page}"
    excerpt = source_item.get("excerpt", "")
    line_num = source_item.get("line")
    highlight_phrases = source_item.get("highlight_phrases") or []

    pdf_path = get_pdf_path(filename)
    if pdf_path is None:
        return "", html.escape(str(filename)), html.escape(str(label)), html.escape(
            excerpt[:240]
        )

    pdf_bytes = build_highlighted_page_pdf(
        pdf_path,
        page,
        excerpt,
        line_num=line_num,
        highlight_phrases=highlight_phrases,
    )
    if not pdf_bytes:
        pdf_bytes = pdf_path.read_bytes()

    pdf_b64 = base64.b64encode(pdf_bytes).decode("ascii")
    return (
        pdf_b64,
        html.escape(str(filename)),
        html.escape(str(label)),
        html.escape(excerpt[:240]),
    )


def _png_bytes_for_item(source_item: Dict[str, Any]) -> Optional[bytes]:
    """Return highlighted page PNG bytes for a source item."""
    png_b64, _, _, _ = _prepare_panel_preview(source_item)
    if png_b64:
        return base64.b64decode(png_b64)
    return None


def _pdf_bytes_for_item(source_item: Dict[str, Any]) -> Optional[bytes]:
    """Return highlighted PDF bytes for optional download."""
    pdf_b64, _, _, _ = _prepare_panel_pdf(source_item)
    if pdf_b64:
        return base64.b64decode(pdf_b64)
    pdf_path = get_pdf_path(source_item.get("file", ""))
    if pdf_path is not None:
        return pdf_path.read_bytes()
    return None


def _use_native_pdf_viewer() -> bool:
    """Use Streamlit dialog preview instead of the JS slide-in panel.

    The dialog renders a PNG via ``st.image`` because Chrome blocks nested PDF
    iframes (including ``st.pdf``) on Streamlit Community Cloud.
    """
    override = os.getenv("USE_NATIVE_PDF_VIEWER", "auto").strip().lower()
    if override in {"0", "false", "no", "off"}:
        return False
    return True


def _show_native_pdf_dialog(item: Dict[str, Any]) -> None:
    """Open the module-level Streamlit dialog for the cited PDF page."""
    _native_pdf_dialog(item)


@st.dialog("Source document", width="large")
def _native_pdf_dialog(item: Dict[str, Any]) -> None:
    """Render a highlighted page PNG inside a Chrome-safe Streamlit dialog."""
    label = item.get("label") or item.get("file", "Source")
    excerpt = (item.get("excerpt") or "").strip()

    st.markdown(f"**{label}**")
    if excerpt:
        st.caption(f"Highlighted passage: {excerpt[:280]}")

    png_bytes = _png_bytes_for_item(item)
    pdf_bytes = _pdf_bytes_for_item(item)
    if not png_bytes:
        st.warning(
            "The original PDF file is not on disk. Re-upload and process it "
            "to enable preview."
        )
    else:
        st.image(png_bytes, use_container_width=True)
        if pdf_bytes:
            st.download_button(
                "Download highlighted page (PDF)",
                data=pdf_bytes,
                file_name=f"{item.get('file', 'source')}-p{item.get('page', 1)}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    if st.button("Close preview", use_container_width=True, key="mpdf_close"):
        dismiss_pdf_viewer()
        st.rerun()


def _mount_panel_script(
    png_b64: str,
    safe_filename: str,
    safe_label: str,
    safe_excerpt: str,
    panel_token: int,
    source_key: str,
    missing_pdf: bool = False,
) -> str:
    """Return JS that mounts the panel on the parent Streamlit document."""
    panel_style_json = json.dumps(PANEL_STYLE)
    panel_token_json = json.dumps(panel_token)
    source_key_json = json.dumps(source_key)
    missing_message = (
        "The original PDF file is not on disk. Re-upload and process it to enable preview."
        if missing_pdf
        else ""
    )

    if missing_pdf:
        body_html = f"""
          <div class="mpdf-header">
            <div class="mpdf-title-wrap">
              <div class="mpdf-title">{safe_filename}</div>
              <div class="mpdf-subtitle">{safe_label}</div>
            </div>
            <button class="mpdf-close" id="mpdf-close-btn" type="button">Close</button>
          </div>
          <div class="mpdf-body" style="display:flex;align-items:center;justify-content:center;padding:24px;text-align:center;color:#4b5563;">
            {html.escape(missing_message)}
          </div>
        """
        png_b64_json = json.dumps("")
    else:
        body_html = f"""
          <div class="mpdf-header">
            <div class="mpdf-title-wrap">
              <div class="mpdf-title">{safe_filename}</div>
              <div class="mpdf-subtitle">{safe_label}</div>
            </div>
            <button class="mpdf-close" id="mpdf-close-btn" type="button">Close</button>
          </div>
          <div class="mpdf-body">
            <div id="mpdf-loading" style="position:absolute;inset:0;display:grid;place-items:center;background:#f3f4f6;color:#4b5563;font-size:14px;font-weight:600;z-index:1;">Loading preview…</div>
            <div class="mpdf-preview-scroll">
              <img id="mpdf-preview-img-{panel_token}" alt="Source page preview" />
            </div>
          </div>
          <div class="mpdf-legend">Highlighted passage: {safe_excerpt or "Opened at cited page."}</div>
        """
        png_b64_json = json.dumps(png_b64)

    return f"""
<!DOCTYPE html>
<html><head><meta charset="utf-8" /></head><body>
<script>
(function () {{
  const parentWin = window.parent;
  const parentDoc = parentWin.document;
  const panelToken = {panel_token_json};
  const sourceKey = {source_key_json};
  const pngB64 = {png_b64_json};

  const existing = parentDoc.getElementById("mpdf-source-panel");
  if (
    existing &&
    existing.getAttribute("data-mpdf-source-key") === sourceKey &&
    existing.getAttribute("data-mpdf-token") === String(panelToken)
  ) {{
    return;
  }}

  function revokeBlobUrl() {{
    if (parentWin.__mpdfBlobUrl) {{
      URL.revokeObjectURL(parentWin.__mpdfBlobUrl);
      parentWin.__mpdfBlobUrl = null;
    }}
  }}

  revokeBlobUrl();
  parentDoc.getElementById("mpdf-source-panel")?.remove();
  parentDoc.getElementById("mpdf-source-panel-style")?.remove();
  parentDoc.body.classList.remove("mpdf-panel-open");

  const style = parentDoc.createElement("style");
  style.id = "mpdf-source-panel-style";
  style.textContent = {panel_style_json};
  parentDoc.head.appendChild(style);

  const panel = parentDoc.createElement("div");
  panel.id = "mpdf-source-panel";
  panel.setAttribute("data-mpdf-token", String(panelToken));
  panel.setAttribute("data-mpdf-source-key", sourceKey);
  panel.innerHTML = `{body_html}`;
  parentDoc.body.appendChild(panel);
  parentDoc.body.classList.add("mpdf-panel-open");

  const appView = parentDoc.querySelector('[data-testid="stAppViewContainer"]');
  if (appView) appView.style.marginRight = "50vw";

  function attachPreview() {{
    if (!pngB64) return;
    const img = parentDoc.getElementById("mpdf-preview-img-" + panelToken);
    const loading = parentDoc.getElementById("mpdf-loading");
    if (!img) return;

    function hideLoading() {{
      if (loading) loading.style.display = "none";
    }}

    img.onload = hideLoading;
    img.onerror = hideLoading;
    img.src = "data:image/png;base64," + pngB64;
    parentWin.__mpdfPanelToken = panelToken;
    setTimeout(hideLoading, 1500);
  }}

  requestAnimationFrame(function () {{
    panel.classList.add("mpdf-open");
    attachPreview();
  }});

  function closePanel() {{
    panel.classList.remove("mpdf-open");
    parentDoc.body.classList.remove("mpdf-panel-open");
    if (appView) appView.style.removeProperty("margin-right");
    revokeBlobUrl();
    setTimeout(function () {{
      panel.remove();
      style.remove();
    }}, 280);
    parentWin.postMessage({{
      isStreamlitMessage: true,
      type: "streamlit:setComponentValue",
      value: "close"
    }}, "*");
  }}

  parentDoc.getElementById("mpdf-close-btn").addEventListener("click", closePanel);
  parentWin.addEventListener("keydown", function onKey(event) {{
    if (event.key === "Escape") {{
      parentWin.removeEventListener("keydown", onKey);
      closePanel();
    }}
  }});
}})();
</script>
</body></html>
"""


def _cleanup_panel_script() -> str:
    """Remove any mounted panel from the parent document."""
    return """
<!DOCTYPE html>
<html><body><script>
(function () {
  const parentWin = window.parent;
  const parentDoc = parentWin.document;
  const panel = parentDoc.getElementById("mpdf-source-panel");
  if (panel) panel.classList.remove("mpdf-open");
  parentDoc.body.classList.remove("mpdf-panel-open");
  const appView = parentDoc.querySelector('[data-testid="stAppViewContainer"]');
  if (appView) appView.style.removeProperty("margin-right");
  if (parentWin.__mpdfBlobUrl) {
    URL.revokeObjectURL(parentWin.__mpdfBlobUrl);
    parentWin.__mpdfBlobUrl = null;
  }
  parentWin.__mpdfPanelToken = null;
  parentDoc.getElementById("mpdf-source-panel")?.remove();
  parentDoc.getElementById("mpdf-source-panel-style")?.remove();
})();
</script></body></html>
"""


def dismiss_pdf_viewer() -> None:
    """Clear PDF viewer session state and invalidate any mounted panel."""
    import streamlit as st

    st.session_state.pdf_viewer = None
    st.session_state.pdf_viewer_item = None
    st.session_state.pdf_viewer_payload = None
    st.session_state.pdf_viewer_stage = "idle"
    st.session_state.pdf_viewer_token = st.session_state.get("pdf_viewer_token", 0) + 1


def request_pdf_viewer(item: Dict[str, Any]) -> None:
    """Queue a source preview with a cleanup-then-mount refresh cycle."""
    import streamlit as st

    payload_item = dict(item)
    st.session_state.pdf_viewer_item = payload_item
    st.session_state.pdf_viewer = payload_item
    st.session_state.pdf_viewer_payload = _build_payload_dict(payload_item)
    st.session_state.pdf_viewer_token = st.session_state.get("pdf_viewer_token", 0) + 1
    st.session_state.pdf_viewer_stage = "cleanup"


def _build_payload_dict(source_item: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare HTML-safe PNG panel payload for the viewer iframe."""
    png_b64, safe_filename, safe_label, safe_excerpt = _prepare_panel_preview(
        source_item
    )
    missing_pdf = not png_b64 and get_pdf_path(source_item.get("file", "")) is None
    source_key = (
        f"{source_item.get('file')}|{source_item.get('page')}|"
        f"{source_item.get('line')}|{source_item.get('excerpt', '')[:80]}"
    )
    return {
        "png_b64": png_b64,
        "safe_filename": safe_filename,
        "safe_label": safe_label,
        "safe_excerpt": safe_excerpt,
        "missing_pdf": missing_pdf,
        "source_key": source_key,
    }


def render_source_citations(source_items: List[Dict[str, Any]], message_key: str) -> None:
    """Render clickable source chips that open the PDF preview panel."""
    import streamlit as st

    if not source_items:
        return

    st.markdown("📄 **Sources:**")
    cols = st.columns(min(len(source_items), 3))
    for index, item in enumerate(source_items):
        col = cols[index % len(cols)]
        if col.button(
            item.get("label", item.get("file", "Source")),
            key=f"{message_key}_source_{index}",
            use_container_width=True,
        ):
            request_pdf_viewer(item)
            st.rerun()


def render_pdf_viewer_modal() -> None:
    """Show the right-side PDF panel when a source has been selected."""
    import streamlit as st

    stage = st.session_state.get("pdf_viewer_stage", "idle")
    item = st.session_state.get("pdf_viewer_item")

    if stage == "cleanup":
        if not _use_native_pdf_viewer():
            components.html(_cleanup_panel_script(), height=1, scrolling=False)
        st.session_state.pdf_viewer_stage = "mount"
        st.rerun()
        return

    if not item:
        if stage != "idle" and not _use_native_pdf_viewer():
            components.html(_cleanup_panel_script(), height=0, scrolling=False)
            st.session_state.pdf_viewer_stage = "idle"
        elif stage != "idle":
            st.session_state.pdf_viewer_stage = "idle"
        return

    token = st.session_state.get("pdf_viewer_token", 0)
    payload = st.session_state.get("pdf_viewer_payload")
    if payload is None:
        payload = _build_payload_dict(item)
        st.session_state.pdf_viewer_payload = payload

    if stage in ("mount", "open"):
        if _use_native_pdf_viewer():
            _show_native_pdf_dialog(item)
            st.session_state.pdf_viewer_stage = "open"
            return

        mount_html = _mount_panel_script(
            payload["png_b64"],
            payload["safe_filename"],
            payload["safe_label"],
            payload["safe_excerpt"],
            panel_token=token,
            source_key=payload["source_key"],
            missing_pdf=payload["missing_pdf"],
        )
        result = components.html(mount_html, height=0, scrolling=False)
        st.session_state.pdf_viewer_stage = "open"
        if result == "close":
            dismiss_pdf_viewer()
            st.session_state.pdf_viewer_stage = "cleanup"
            st.rerun()
