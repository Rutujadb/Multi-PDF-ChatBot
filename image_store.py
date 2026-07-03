"""SQLite manifest for extracted PDF images and Gemma-generated captions."""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import IMAGE_DB_PATH

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS pdf_images (
    image_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    source TEXT NOT NULL,
    page INTEGER NOT NULL,
    page_label TEXT NOT NULL,
    image_index INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    width INTEGER,
    height INTEGER,
    bytes_sha256 TEXT NOT NULL,
    caption TEXT,
    caption_model TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(session_id, source, page, image_index)
);

CREATE INDEX IF NOT EXISTS idx_pdf_images_session_source_page
    ON pdf_images(session_id, source, page);
"""


def _utc_now() -> str:
    """Return the current UTC time as an ISO8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_db_path(db_path: Optional[Path] = None) -> Path:
    """Return the database path to use for one operation."""
    return Path(db_path) if db_path is not None else Path(IMAGE_DB_PATH)


def init_db(db_path: Optional[Path] = None) -> None:
    """Create image manifest tables when they do not exist."""
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()
    logger.debug("Image manifest DB initialised at %s", path)


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open a SQLite connection and ensure schema exists."""
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    init_db(path)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a SQLite row into a plain dictionary."""
    return {key: row[key] for key in row.keys()}


def insert_images(
    records: List[Dict[str, Any]],
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Insert extracted image records, skipping duplicates already stored."""
    if not records:
        return []

    logger.info("Inserting %d image record(s) into manifest", len(records))
    created_at = _utc_now()
    inserted: List[Dict[str, Any]] = []
    with _connect(db_path) as conn:
        for record in records:
            conn.execute(
                """
                INSERT OR IGNORE INTO pdf_images (
                    image_id, session_id, source, page, page_label,
                    image_index, file_path, width, height, bytes_sha256,
                    caption, caption_model, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?)
                """,
                (
                    record["image_id"],
                    record["session_id"],
                    record["source"],
                    int(record["page"]),
                    str(record["page_label"]),
                    int(record["image_index"]),
                    record["file_path"],
                    record.get("width"),
                    record.get("height"),
                    record["bytes_sha256"],
                    created_at,
                ),
            )
            row = conn.execute(
                """
                SELECT * FROM pdf_images
                WHERE session_id = ? AND source = ? AND page = ? AND image_index = ?
                """,
                (
                    record["session_id"],
                    record["source"],
                    int(record["page"]),
                    int(record["image_index"]),
                ),
            ).fetchone()
            if row is not None:
                inserted.append(_row_to_dict(row))
        conn.commit()
    logger.info("Stored %d image record(s) in manifest", len(inserted))
    return inserted


def update_caption(
    image_id: str,
    caption: str,
    caption_model: str,
    db_path: Optional[Path] = None,
) -> None:
    """Persist a Gemma-generated caption for one image."""
    logger.debug("Updating caption for image %s (model=%s)", image_id, caption_model)
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE pdf_images
            SET caption = ?, caption_model = ?
            WHERE image_id = ?
            """,
            (caption.strip(), caption_model, image_id),
        )
        conn.commit()


def get_image(image_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return one image record by id."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM pdf_images WHERE image_id = ?",
            (image_id,),
        ).fetchone()
    return _row_to_dict(row) if row is not None else None


def get_images_for_page(
    session_id: str,
    source: str,
    page: int,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return image records for one zero-based PDF page."""
    sid = (session_id or "").strip()
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT * FROM pdf_images
            WHERE session_id = ? AND source = ? AND page = ?
            ORDER BY image_index
            """,
            (sid, source, int(page)),
        ).fetchall()
    return [_row_to_dict(row) for row in rows]


def list_images(
    session_id: str,
    source: Optional[str] = None,
    page: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """List image records for a session with optional source/page filters."""
    sid = (session_id or "").strip()
    query = "SELECT * FROM pdf_images WHERE session_id = ?"
    params: List[Any] = [sid]
    if source:
        query += " AND source = ?"
        params.append(source)
    if page is not None:
        query += " AND page = ?"
        params.append(int(page))
    query += " ORDER BY source, page, image_index"

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return [_row_to_dict(row) for row in rows]


def count_images_by_source(
    session_id: str,
    db_path: Optional[Path] = None,
) -> Dict[str, int]:
    """Return a mapping of source filename to extracted image count."""
    sid = (session_id or "").strip()
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT source, COUNT(*) AS image_count
            FROM pdf_images
            WHERE session_id = ?
            GROUP BY source
            """,
            (sid,),
        ).fetchall()
    return {row["source"]: int(row["image_count"]) for row in rows}


def delete_images_for_source(
    session_id: str,
    source: str,
    db_path: Optional[Path] = None,
) -> None:
    """Delete manifest rows and image files for one PDF source."""
    logger.info("Deleting images for source '%s' in session %s", source, session_id)
    sid = (session_id or "").strip()
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT file_path FROM pdf_images
            WHERE session_id = ? AND source = ?
            """,
            (sid, source),
        ).fetchall()
        conn.execute(
            "DELETE FROM pdf_images WHERE session_id = ? AND source = ?",
            (sid, source),
        )
        conn.commit()

    for row in rows:
        path = Path(row["file_path"])
        if path.is_file():
            path.unlink(missing_ok=True)


def delete_images_for_session(
    session_id: str,
    db_path: Optional[Path] = None,
) -> None:
    """Delete all image records and files for one session."""
    logger.info("Deleting all images for session %s", session_id)
    sid = (session_id or "").strip()
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT file_path FROM pdf_images WHERE session_id = ?",
            (sid,),
        ).fetchall()
        conn.execute("DELETE FROM pdf_images WHERE session_id = ?", (sid,))
        conn.commit()

    for row in rows:
        path = Path(row["file_path"])
        if path.is_file():
            path.unlink(missing_ok=True)
