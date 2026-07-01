"""Tests for conversation-recall routing helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rag_chain import answer_from_chat_history
from sqlite_memory import SqliteChatMessageHistory
from utils import is_conversation_recall_question


def test_is_conversation_recall_question_detects_previous_question() -> None:
    """Meta questions about prior turns should bypass PDF retrieval."""
    assert is_conversation_recall_question("what is my previous question?")
    assert not is_conversation_recall_question("what is this pdf about?")


def test_answer_from_chat_history_reports_previous_question(
    tmp_path: Path,
) -> None:
    """Recall answers should use stored user turns, not PDF context."""
    db_path = tmp_path / "chat_memory.db"
    history = SqliteChatMessageHistory("recall-session", db_path=db_path)
    history.add_messages(
        [
            HumanMessage(content="what is this pdf about?"),
            AIMessage(content="It is a weekly learning plan."),
        ]
    )

    result = answer_from_chat_history(
        "what is my previous question?",
        history,
    )

    assert "what is this pdf about?" in result["answer"]
    assert result["source_documents"] == []

    reloaded = SqliteChatMessageHistory("recall-session", db_path=db_path)
    assert len(reloaded.messages) == 4
