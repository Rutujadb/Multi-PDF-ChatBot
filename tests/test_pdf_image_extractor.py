"""PyTest coverage for PDF image extraction and manifest storage."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from image_rag import build_image_context_block, enrich_documents_with_image_context
from pdf_image_pipeline import build_caption_chunks_for_sources
from image_store import (
    count_images_by_source,
    get_images_for_page,
    init_db,
    insert_images,
    update_caption,
)
from pdf_image_extractor import extract_images_from_pdf


@pytest.fixture
def image_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide isolated image manifest and extraction directories."""
    db_path = tmp_path / "image_manifest.db"
    images_dir = tmp_path / "extracted_images"
    monkeypatch.setattr("config.EXTRACTED_IMAGES_DIR", images_dir)
    monkeypatch.setattr("config.IMAGE_DB_PATH", db_path)
    monkeypatch.setattr("image_store.IMAGE_DB_PATH", db_path)
    monkeypatch.setattr("config.IMAGE_EXTRACTION_ENABLED", True)
    monkeypatch.setattr("config.IMAGE_MIN_WIDTH", 10)
    monkeypatch.setattr("config.IMAGE_MIN_HEIGHT", 10)
    init_db(db_path)
    return db_path


def test_insert_and_query_images(image_db: Path) -> None:
    """Manifest rows should be retrievable by session, source, and page."""
    records = [
        {
            "image_id": "img-1",
            "session_id": "sess-a",
            "source": "report.pdf",
            "page": 0,
            "page_label": "1",
            "image_index": 0,
            "file_path": str(image_db.parent / "img.png"),
            "width": 120,
            "height": 80,
            "bytes_sha256": "abc123",
        }
    ]
    (image_db.parent / "img.png").write_bytes(b"fake")

    inserted = insert_images(records, db_path=image_db)
    assert len(inserted) == 1
    update_caption("img-1", "A bar chart of quarterly revenue.", "test-model", db_path=image_db)

    rows = get_images_for_page("sess-a", "report.pdf", 0, db_path=image_db)
    assert len(rows) == 1
    assert rows[0]["caption"] == "A bar chart of quarterly revenue."
    assert count_images_by_source("sess-a", db_path=image_db)["report.pdf"] == 1


def test_build_image_context_block(image_db: Path) -> None:
    """Image captions should render as plain text context blocks."""
    insert_images(
        [
            {
                "image_id": "img-2",
                "session_id": "sess-b",
                "source": "plan.pdf",
                "page": 2,
                "page_label": "3",
                "image_index": 0,
                "file_path": str(image_db.parent / "plan.png"),
                "width": 200,
                "height": 100,
                "bytes_sha256": "def456",
            }
        ],
        db_path=image_db,
    )
    update_caption("img-2", "Org chart with Engineering and Sales.", "test-model", db_path=image_db)

    block = build_image_context_block("sess-b", "plan.pdf", 2, page_label="3")
    assert "plan.pdf" in block
    assert "Org chart with Engineering and Sales." in block


def test_enrich_documents_with_image_context(image_db: Path) -> None:
    """Retrieved chunks should gain page-level image caption text."""
    from langchain_core.documents import Document

    insert_images(
        [
            {
                "image_id": "img-3",
                "session_id": "sess-c",
                "source": "deck.pdf",
                "page": 1,
                "page_label": "2",
                "image_index": 0,
                "file_path": str(image_db.parent / "deck.png"),
                "width": 300,
                "height": 200,
                "bytes_sha256": "ghi789",
            }
        ],
        db_path=image_db,
    )
    update_caption("img-3", "Process flow diagram.", "test-model", db_path=image_db)

    docs = [
        Document(
            page_content="Some page text.",
            metadata={"source": "deck.pdf", "page": 1, "page_label": "2"},
        )
    ]
    enriched = enrich_documents_with_image_context(docs, "sess-c")
    assert "Process flow diagram." in enriched[0].page_content
    assert enriched[0].page_content.startswith("Some page text.")


def test_build_caption_chunks_for_sources(image_db: Path) -> None:
    """Caption manifest rows should become indexable text chunks."""
    insert_images(
        [
            {
                "image_id": "img-4",
                "session_id": "sess-d",
                "source": "visual.pdf",
                "page": 0,
                "page_label": "1",
                "image_index": 0,
                "file_path": str(image_db.parent / "visual.png"),
                "width": 100,
                "height": 100,
                "bytes_sha256": "zzz999",
            }
        ],
        db_path=image_db,
    )
    update_caption("img-4", "A company logo on a white background.", "test-model", db_path=image_db)

    chunks = build_caption_chunks_for_sources("sess-d", ["visual.pdf"])
    assert len(chunks) == 1
    assert "company logo" in chunks[0].page_content


def test_extract_images_from_pdf(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """PyMuPDF extraction should persist one image file and metadata record."""
    mock_fitz = MagicMock()
    images_dir = tmp_path / "extracted_images"
    monkeypatch.setattr("config.EXTRACTED_IMAGES_DIR", images_dir)
    monkeypatch.setattr("config.IMAGE_MIN_WIDTH", 10)
    monkeypatch.setattr("config.IMAGE_MIN_HEIGHT", 10)
    monkeypatch.setitem(sys.modules, "fitz", mock_fitz)

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    mock_doc = MagicMock()
    mock_doc.__len__.return_value = 1
    mock_page = MagicMock()
    mock_page.get_images.return_value = [(7, 0, 100, 100, 8, "DeviceRGB", "", "Im1", "FlateDecode")]
    mock_doc.__getitem__.return_value = mock_page
    mock_doc.extract_image.return_value = {
        "width": 100,
        "height": 80,
        "image": b"\x89PNG\r\n\x1a\n",
        "ext": "png",
    }
    mock_fitz.open.return_value = mock_doc

    records = extract_images_from_pdf(pdf_path, "sess-x", "sample.pdf")
    assert len(records) == 1
    assert records[0]["source"] == "sample.pdf"
    assert Path(records[0]["file_path"]).is_file()
    mock_doc.close.assert_called_once()
