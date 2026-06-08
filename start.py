"""Production entrypoint for Render and other PaaS hosts."""

from __future__ import annotations

import os
import sys

import uvicorn


def main() -> None:
    """Start uvicorn with the platform-assigned PORT."""
    port = int(os.environ.get("PORT", "8000"))
    print(f"Starting API on 0.0.0.0:{port}", flush=True)
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
        print(f"Failed to start API: {exc}", file=sys.stderr, flush=True)
        raise
