"""Tests for image captioning with primary + fallback models."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_caption_image_uses_fallback_on_rate_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Primary rate limit should retry with the configured fallback model."""
    from image_captioner import caption_image

    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    calls: list[tuple[str, str]] = []

    def fake_get_caption_llm(provider=None, model=None):
        calls.append((provider, model))
        llm = MagicMock()
        if model == "google/gemma-3-12b-it":
            llm.invoke.side_effect = Exception("429 rate limit exceeded")
        else:
            llm.invoke.return_value = MagicMock(
                content="Bar chart showing quarterly revenue growth."
            )
        return llm

    monkeypatch.setattr("image_captioner.get_caption_llm", fake_get_caption_llm)
    monkeypatch.setattr(
        "image_captioner.get_image_caption_model_attempts",
        lambda: [
            ("openrouter", "google/gemma-3-12b-it"),
            ("openrouter", "google/gemma-3-4b-it:free"),
        ],
    )
    monkeypatch.setattr("config.IMAGE_CAPTION_ENABLED", True)

    caption, model = caption_image(image_path, source="report.pdf", page_label="2")

    assert caption == "Bar chart showing quarterly revenue growth."
    assert model == "google/gemma-3-4b-it:free"
    assert calls == [
        ("openrouter", "google/gemma-3-12b-it"),
        ("openrouter", "google/gemma-3-4b-it:free"),
    ]


def test_caption_image_returns_empty_when_all_attempts_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failed caption attempts should not return user-facing chat error text."""
    from image_captioner import caption_image

    image_path = tmp_path / "chart.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\n")

    def fake_get_caption_llm(provider=None, model=None):
        llm = MagicMock()
        llm.invoke.side_effect = Exception("429 rate limit exceeded")
        return llm

    monkeypatch.setattr("image_captioner.get_caption_llm", fake_get_caption_llm)
    monkeypatch.setattr(
        "image_captioner.get_image_caption_model_attempts",
        lambda: [("openrouter", "google/gemma-3-12b-it")],
    )
    monkeypatch.setattr("config.IMAGE_CAPTION_ENABLED", True)

    caption, model = caption_image(image_path)

    assert caption == ""
    assert model == ""
