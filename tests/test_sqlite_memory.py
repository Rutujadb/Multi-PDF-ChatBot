"""PyTest coverage for SQLite-backed chat memory."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlite_memory import (
    SqliteChatMessageHistory,
    append_message,
    clear_messages,
    create_session,
    delete_session,
    get_messages,
    get_session,
    get_summary,
    init_db,
    list_sessions,
    messages_to_api_format,
    session_exists,
    update_summary,
)


@pytest.fixture
def memory_db(tmp_path: Path) -> Path:
    """Provide an isolated SQLite database path for each test."""
    db_path = tmp_path / "chat_memory.db"
    init_db(db_path)
    return db_path


def test_init_db_creates_tables(memory_db: Path) -> None:
    """Schema creation should succeed on a fresh database file."""
    import sqlite3

    with sqlite3.connect(memory_db) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert "chat_sessions" in tables
    assert "chat_messages" in tables


def test_create_session_idempotent(memory_db: Path) -> None:
    """Creating the same session twice should not raise."""
    first = create_session("sess-1", db_path=memory_db)
    second = create_session("sess-1", db_path=memory_db)
    assert first["session_id"] == "sess-1"
    assert second["session_id"] == "sess-1"
    assert session_exists("sess-1", db_path=memory_db)


def test_append_and_get_messages_order(memory_db: Path) -> None:
    """Messages should be returned in insert order."""
    create_session("sess-2", db_path=memory_db)
    append_message("sess-2", "user", "Hello?", db_path=memory_db)
    append_message("sess-2", "assistant", "Hi there.", db_path=memory_db)

    rows = get_messages("sess-2", db_path=memory_db)
    assert len(rows) == 2
    assert rows[0]["role"] == "user"
    assert rows[0]["content"] == "Hello?"
    assert rows[1]["role"] == "assistant"
    assert rows[1]["content"] == "Hi there."


def test_get_session_metadata(memory_db: Path) -> None:
    """Session metadata should include message count and timestamps."""
    create_session("sess-3", db_path=memory_db)
    append_message("sess-3", "user", "Question one", db_path=memory_db)

    session = get_session("sess-3", db_path=memory_db)
    assert session is not None
    assert session["message_count"] == 1
    assert session["created_at"]
    assert session["updated_at"]


def test_update_and_get_summary(memory_db: Path) -> None:
    """Summary updates should round-trip through the session row."""
    create_session("sess-4", db_path=memory_db)
    update_summary("sess-4", "User asked about leave policy.", db_path=memory_db)
    assert get_summary("sess-4", db_path=memory_db) == "User asked about leave policy."


def test_clear_messages_keeps_session(memory_db: Path) -> None:
    """Clearing messages should keep the session row but remove turns."""
    create_session("sess-5", db_path=memory_db)
    append_message("sess-5", "user", "Hello", db_path=memory_db)
    removed = clear_messages("sess-5", db_path=memory_db)

    assert removed == 1
    assert get_messages("sess-5", db_path=memory_db) == []
    assert session_exists("sess-5", db_path=memory_db)
    assert get_summary("sess-5", db_path=memory_db) is None


def test_delete_session_removes_all(memory_db: Path) -> None:
    """Deleting a session should remove its messages too."""
    create_session("sess-6", db_path=memory_db)
    append_message("sess-6", "user", "Delete me", db_path=memory_db)

    assert delete_session("sess-6", db_path=memory_db) is True
    assert get_session("sess-6", db_path=memory_db) is None
    assert get_messages("sess-6", db_path=memory_db) == []


def test_get_messages_empty_session(memory_db: Path) -> None:
    """Unknown sessions should return an empty message list."""
    assert get_messages("missing-session", db_path=memory_db) == []


def test_append_auto_creates_session(memory_db: Path) -> None:
    """Appending to a new session id should create the session row."""
    append_message("sess-7", "user", "Auto create", db_path=memory_db)
    assert session_exists("sess-7", db_path=memory_db)


def test_invalid_role_raises(memory_db: Path) -> None:
    """Unsupported roles should be rejected."""
    create_session("sess-8", db_path=memory_db)
    with pytest.raises(ValueError):
        append_message("sess-8", "tool", "Not allowed", db_path=memory_db)


def test_list_sessions(memory_db: Path) -> None:
    """list_sessions should return all known sessions."""
    create_session("sess-a", db_path=memory_db)
    create_session("sess-b", db_path=memory_db)
    append_message("sess-b", "user", "One", db_path=memory_db)

    sessions = list_sessions(db_path=memory_db)
    ids = {item["session_id"] for item in sessions}
    assert {"sess-a", "sess-b"}.issubset(ids)


def test_langchain_history_add_and_reload(memory_db: Path) -> None:
    """A new history instance should reload messages written by another."""
    history = SqliteChatMessageHistory("sess-9", db_path=memory_db)
    history.add_messages(
        [
            HumanMessage(content="What is the leave policy?"),
            AIMessage(content="Employees receive twelve days of leave."),
        ]
    )

    reloaded = SqliteChatMessageHistory("sess-9", db_path=memory_db)
    messages = reloaded.messages
    assert len(messages) == 2
    assert isinstance(messages[0], HumanMessage)
    assert messages[0].content == "What is the leave policy?"
    assert isinstance(messages[1], AIMessage)


def test_langchain_clear(memory_db: Path) -> None:
    """clear() should remove persisted LangChain messages."""
    history = SqliteChatMessageHistory("sess-10", db_path=memory_db)
    history.add_message(HumanMessage(content="Temporary"))
    history.clear()

    reloaded = SqliteChatMessageHistory("sess-10", db_path=memory_db)
    assert reloaded.messages == []


def test_langchain_system_message(memory_db: Path) -> None:
    """System messages should persist with the system role."""
    history = SqliteChatMessageHistory("sess-11", db_path=memory_db)
    history.add_message(SystemMessage(content="You are helpful."))

    rows = get_messages("sess-11", db_path=memory_db)
    assert rows[0]["role"] == "system"


def test_messages_to_api_format(memory_db: Path) -> None:
    """API UI messages should rebuild from persisted rows."""
    append_message("sess-12", "user", "Hi", db_path=memory_db)
    append_message("sess-12", "assistant", "Hello", db_path=memory_db)

    ui_messages = messages_to_api_format("sess-12", db_path=memory_db)
    assert ui_messages == [
        {"role": "user", "text": "Hi"},
        {
            "role": "assistant",
            "text": "Hello",
            "sources": [],
            "sources_text": "",
        },
    ]


def test_persistence_new_connection(memory_db: Path) -> None:
    """Simulate a process restart by using only a fresh DB path connection."""
    append_message("restart-session", "user", "Persist this", db_path=memory_db)
    append_message(
        "restart-session",
        "assistant",
        "It should survive.",
        db_path=memory_db,
    )

    loaded = get_messages("restart-session", db_path=memory_db)
    assert len(loaded) == 2
    assert loaded[0]["content"] == "Persist this"
    assert loaded[1]["content"] == "It should survive."
