"""SQLite-backed persistent chat memory for Multi-PDF ChatBot.

Stores conversation sessions and messages on disk so LangChain chat history
survives API or Streamlit process restarts.

Schema:
    chat_sessions — session_id, summary, created_at, updated_at
    chat_messages — id, session_id, timestamp, role, content
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from config import CHAT_DB_PATH

logger = logging.getLogger(__name__)

VALID_ROLES = frozenset({"user", "assistant", "system"})

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    summary TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session
    ON chat_messages(session_id, id);
"""


def _utc_now() -> str:
    """Return the current UTC time as an ISO8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _resolve_db_path(db_path: Optional[Path] = None) -> Path:
    """Return the database path to use for one operation."""
    return Path(db_path) if db_path is not None else Path(CHAT_DB_PATH)


def _connect(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open a SQLite connection and ensure schema exists."""
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    init_db(path)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Create chat memory tables and indexes when they do not exist."""
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(_SCHEMA_SQL)
        conn.commit()


def _validate_role(role: str) -> str:
    """Normalize and validate a message role string."""
    normalized = (role or "").strip().lower()
    if normalized not in VALID_ROLES:
        raise ValueError(
            f"Invalid role {role!r}. Expected one of: {', '.join(sorted(VALID_ROLES))}."
        )
    return normalized


def create_session(session_id: str, db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Create a chat session row, or return the existing session metadata."""
    sid = (session_id or "").strip()
    if not sid:
        raise ValueError("session_id cannot be empty.")

    logger.info("Creating chat session: %s", sid)
    now = _utc_now()
    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO chat_sessions (session_id, summary, created_at, updated_at)
            VALUES (?, NULL, ?, ?)
            ON CONFLICT(session_id) DO NOTHING
            """,
            (sid, now, now),
        )
        conn.commit()
    return get_session(sid, db_path=db_path) or {
        "session_id": sid,
        "summary": None,
        "created_at": now,
        "updated_at": now,
        "message_count": 0,
    }


def session_exists(session_id: str, db_path: Optional[Path] = None) -> bool:
    """Return True when a session row exists in the database."""
    sid = (session_id or "").strip()
    if not sid:
        return False
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM chat_sessions WHERE session_id = ?",
            (sid,),
        ).fetchone()
    return row is not None


def get_session(session_id: str, db_path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
    """Return session metadata including message count."""
    sid = (session_id or "").strip()
    if not sid:
        return None
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT
                s.session_id,
                s.summary,
                s.created_at,
                s.updated_at,
                COUNT(m.id) AS message_count
            FROM chat_sessions AS s
            LEFT JOIN chat_messages AS m ON m.session_id = s.session_id
            WHERE s.session_id = ?
            GROUP BY s.session_id, s.summary, s.created_at, s.updated_at
            """,
            (sid,),
        ).fetchone()
    if row is None:
        return None
    return {
        "session_id": row["session_id"],
        "summary": row["summary"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "message_count": int(row["message_count"]),
    }


def list_sessions(db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Return all sessions ordered by most recently updated."""
    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT
                s.session_id,
                s.summary,
                s.created_at,
                s.updated_at,
                COUNT(m.id) AS message_count
            FROM chat_sessions AS s
            LEFT JOIN chat_messages AS m ON m.session_id = s.session_id
            GROUP BY s.session_id, s.summary, s.created_at, s.updated_at
            ORDER BY s.updated_at DESC
            """
        ).fetchall()
    return [
        {
            "session_id": row["session_id"],
            "summary": row["summary"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "message_count": int(row["message_count"]),
        }
        for row in rows
    ]


def append_message(
    session_id: str,
    role: str,
    content: str,
    timestamp: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> int:
    """Append one message to a session and return the new row id."""
    sid = (session_id or "").strip()
    if not sid:
        raise ValueError("session_id cannot be empty.")

    normalized_role = _validate_role(role)
    body = (content or "").strip()
    if not body:
        raise ValueError("content cannot be empty.")

    logger.debug("Appending %s message to session %s", normalized_role, sid)
    create_session(sid, db_path=db_path)
    ts = timestamp or _utc_now()
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO chat_messages (session_id, timestamp, role, content)
            VALUES (?, ?, ?, ?)
            """,
            (sid, ts, normalized_role, body),
        )
        conn.execute(
            "UPDATE chat_sessions SET updated_at = ? WHERE session_id = ?",
            (ts, sid),
        )
        conn.commit()
        return int(cursor.lastrowid)


def get_messages(
    session_id: str,
    limit: Optional[int] = None,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Return ordered messages for one session."""
    sid = (session_id or "").strip()
    if not sid:
        return []

    query = """
        SELECT id, session_id, timestamp, role, content
        FROM chat_messages
        WHERE session_id = ?
        ORDER BY id ASC
    """
    params: List[Any] = [sid]
    if limit is not None and limit > 0:
        query += " LIMIT ?"
        params.append(limit)

    with _connect(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    return [
        {
            "id": row["id"],
            "session_id": row["session_id"],
            "timestamp": row["timestamp"],
            "role": row["role"],
            "content": row["content"],
        }
        for row in rows
    ]


def get_summary(session_id: str, db_path: Optional[Path] = None) -> Optional[str]:
    """Return the rolling summary for one session, if set."""
    session = get_session(session_id, db_path=db_path)
    if session is None:
        return None
    return session.get("summary")


def update_summary(
    session_id: str,
    summary: str,
    db_path: Optional[Path] = None,
) -> None:
    """Persist a rolling conversation summary for one session."""
    sid = (session_id or "").strip()
    if not sid:
        raise ValueError("session_id cannot be empty.")
    create_session(sid, db_path=db_path)
    now = _utc_now()
    with _connect(db_path) as conn:
        conn.execute(
            """
            UPDATE chat_sessions
            SET summary = ?, updated_at = ?
            WHERE session_id = ?
            """,
            (summary, now, sid),
        )
        conn.commit()


def clear_messages(session_id: str, db_path: Optional[Path] = None) -> int:
    """Delete all messages for a session while keeping the session row."""
    sid = (session_id or "").strip()
    if not sid:
        return 0
    logger.info("Clearing all messages for session %s", sid)
    now = _utc_now()
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM chat_messages WHERE session_id = ?",
            (sid,),
        )
        conn.execute(
            """
            UPDATE chat_sessions
            SET summary = NULL, updated_at = ?
            WHERE session_id = ?
            """,
            (now, sid),
        )
        conn.commit()
        return int(cursor.rowcount)


def delete_session(session_id: str, db_path: Optional[Path] = None) -> bool:
    """Delete a session and all of its messages."""
    sid = (session_id or "").strip()
    if not sid:
        return False
    logger.info("Deleting chat session: %s", sid)
    with _connect(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM chat_sessions WHERE session_id = ?",
            (sid,),
        )
        conn.commit()
        return cursor.rowcount > 0


def _message_to_row(message: BaseMessage) -> tuple[str, str]:
    """Map a LangChain message to database role and content."""
    if isinstance(message, HumanMessage):
        return "user", str(message.content)
    if isinstance(message, AIMessage):
        return "assistant", str(message.content)
    if isinstance(message, SystemMessage):
        return "system", str(message.content)
    message_type = getattr(message, "type", "user")
    if message_type == "human":
        return "user", str(message.content)
    if message_type == "ai":
        return "assistant", str(message.content)
    if message_type == "system":
        return "system", str(message.content)
    return "user", str(message.content)


def _row_to_message(row: Dict[str, Any]) -> BaseMessage:
    """Convert a database row into a LangChain message."""
    role = row["role"]
    content = row["content"]
    if role == "assistant":
        return AIMessage(content=content)
    if role == "system":
        return SystemMessage(content=content)
    return HumanMessage(content=content)


def messages_to_api_format(
    session_id: str,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Rebuild React/API chat bubbles from persisted memory rows."""
    ui_messages: List[Dict[str, Any]] = []
    for row in get_messages(session_id, db_path=db_path):
        role = row["role"]
        if role == "user":
            ui_messages.append({"role": "user", "text": row["content"]})
        elif role == "assistant":
            ui_messages.append(
                {
                    "role": "assistant",
                    "text": row["content"],
                    "sources": [],
                    "sources_text": "",
                }
            )
    return ui_messages


def messages_to_streamlit_format(
    session_id: str,
    db_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Rebuild Streamlit chat bubbles from persisted memory rows."""
    ui_messages: List[Dict[str, Any]] = []
    for row in get_messages(session_id, db_path=db_path):
        role = row["role"]
        if role == "user":
            ui_messages.append({"role": "user", "content": row["content"]})
        elif role == "assistant":
            ui_messages.append(
                {
                    "role": "assistant",
                    "content": row["content"],
                    "sources": "",
                    "source_items": [],
                }
            )
    return ui_messages


class SqliteChatMessageHistory(BaseChatMessageHistory):
    """LangChain chat history backed by SQLite for one session."""

    def __init__(self, session_id: str, db_path: Optional[Path] = None):
        """Attach chat history to a persistent session id."""
        super().__init__()
        self.session_id = (session_id or "").strip()
        if not self.session_id:
            raise ValueError("session_id cannot be empty.")
        self.db_path = Path(db_path) if db_path is not None else None
        create_session(self.session_id, db_path=self.db_path)

    @property
    def messages(self) -> List[BaseMessage]:
        """Return all LangChain messages for this session."""
        rows = get_messages(self.session_id, db_path=self.db_path)
        return [_row_to_message(row) for row in rows]

    def add_message(self, message: BaseMessage) -> None:
        """Persist a single LangChain message."""
        role, content = _message_to_row(message)
        append_message(
            self.session_id,
            role,
            content,
            db_path=self.db_path,
        )

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        """Persist multiple LangChain messages in order."""
        for message in messages:
            self.add_message(message)

    def clear(self) -> None:
        """Remove all persisted messages for this session."""
        clear_messages(self.session_id, db_path=self.db_path)
