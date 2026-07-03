"""Production entrypoint for Render and other PaaS hosts."""

from __future__ import annotations

import logging
import os
import sys

import uvicorn

logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    """Start uvicorn with the platform-assigned PORT."""
    port = int(os.environ.get("PORT", "8000"))
    logging.getLogger(__name__).info("Starting API on 0.0.0.0:%d", port)
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=True,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.getLogger(__name__).critical("Failed to start API: %s", exc, exc_info=True)
        raise
