"""Launch the React UI (dev), FastAPI backend, and optional Streamlit classic UI.

Usage:
    python run_dev.py              # React + API
    python run_dev.py --streamlit  # React + API + Streamlit
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
REQUIREMENTS = ROOT / "requirements.txt"

# import name -> pip package name (when they differ)
PYTHON_DEPS = {
    "uvicorn": "uvicorn[standard]",
    "fastapi": "fastapi",
    "langchain_community": "langchain-community",
    "langchain_classic": "langchain-classic",
    "streamlit": "streamlit",
}


def _require(command: str, install_hint: str) -> str:
    """Return the executable path or exit with a helpful message."""
    path = shutil.which(command)
    if path:
        return path
    print(f"Missing `{command}`. {install_hint}")
    sys.exit(1)


def _missing_modules(modules: list[str]) -> list[str]:
    """Return import names that are not available in the current interpreter."""
    missing = []
    for module in modules:
        if importlib.util.find_spec(module) is None:
            missing.append(module)
    return missing


def ensure_python_deps(include_streamlit: bool = False) -> None:
    """Install requirements.txt when core backend packages are missing."""
    required = ["uvicorn", "fastapi", "langchain_community", "langchain_classic"]
    if include_streamlit:
        required.append("streamlit")

    missing = _missing_modules(required)
    if not missing:
        return

    print("Missing Python packages:", ", ".join(missing))
    if not REQUIREMENTS.exists():
        print(f"Could not find {REQUIREMENTS}. Install dependencies manually:")
        print("  python -m pip install -r requirements.txt")
        sys.exit(1)

    print("Installing Python dependencies (first run can take a few minutes)…")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS)],
        cwd=ROOT,
    )
    if result.returncode != 0:
        print("\nDependency install failed. Run manually, then retry:")
        print("  python -m pip install -r requirements.txt")
        sys.exit(result.returncode)

    still_missing = _missing_modules(required)
    if still_missing:
        print("\nSome packages are still missing after install:", ", ".join(still_missing))
        pkgs = ", ".join(PYTHON_DEPS.get(name, name) for name in still_missing)
        print(f"Try: python -m pip install {pkgs}")
        sys.exit(1)


def verify_api_import() -> None:
    """Ensure api.py is present and syntactically valid without loading models."""
    api_path = ROOT / "api.py"
    if not api_path.is_file():
        print(f"\nMissing {api_path}")
        sys.exit(1)
    try:
        compile(api_path.read_text(encoding="utf-8"), str(api_path), "exec")
    except Exception as exc:
        print("\nFailed to parse api.py:")
        print(f"  {exc}")
        sys.exit(1)


def wait_for_api(timeout_seconds: int = 360) -> None:
    """Block until the FastAPI health endpoint responds."""
    import urllib.error
    import urllib.request

    url = "http://127.0.0.1:8000/api/health"
    print(
        "Waiting for API on http://localhost:8000 "
        "(first request can take a few minutes while models load)…"
    )
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=3) as response:
                if response.status == 200:
                    print("API is ready.")
                    return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(2)
    print("\nAPI did not become ready in time. Check the API logs above for errors.")
    sys.exit(1)


def start_process(name: str, command: list[str], cwd: Path) -> subprocess.Popen:
    """Start a child process with inherited stdout/stderr."""
    print(f"Starting {name}…")
    return subprocess.Popen(command, cwd=cwd)


def main() -> None:
    """Start local development processes for the Multi-PDF ChatBot."""
    parser = argparse.ArgumentParser(description="Run the Multi-PDF ChatBot dev stack")
    parser.add_argument(
        "--streamlit",
        action="store_true",
        help="Also launch the classic Streamlit UI on http://localhost:8501",
    )
    args = parser.parse_args()

    ensure_python_deps(include_streamlit=args.streamlit)
    verify_api_import()

    npm = _require("npm", "Install Node.js from https://nodejs.org")
    if not (FRONTEND / "node_modules").exists():
        print("Installing frontend dependencies…")
        subprocess.run([npm, "install"], cwd=FRONTEND, check=True)

    process_names = ["API"]
    processes = [
        start_process(
            "API",
            [sys.executable, "-m", "uvicorn", "api:app", "--reload", "--port", "8000"],
            ROOT,
        ),
    ]

    wait_for_api()

    process_names.append("React")
    processes.append(start_process("React", [npm, "run", "dev"], FRONTEND))

    if args.streamlit:
        process_names.append("Streamlit")
        processes.append(
            start_process(
                "Streamlit",
                [
                    sys.executable,
                    "-m",
                    "streamlit",
                    "run",
                    "app.py",
                    "--server.port",
                    "8501",
                ],
                ROOT,
            )
        )

    print("\nMulti-PDF ChatBot dev stack")
    print("  Landing + dashboard : http://localhost:5173")
    print("  API                 : http://localhost:8000")
    if args.streamlit:
        print("  Streamlit classic   : http://localhost:8501")
    print("\nPress Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
            for name, proc in zip(process_names, processes):
                code = proc.poll()
                if code is not None:
                    print(f"\n{name} exited with code {code}. Shutting down…")
                    raise RuntimeError(f"{name} process exited unexpectedly.")
    except KeyboardInterrupt:
        print("\nStopping…")
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            if proc.poll() is None:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()


if __name__ == "__main__":
    main()
