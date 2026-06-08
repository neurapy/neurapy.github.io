import subprocess
import sys
import time
from pathlib import Path

# Set project root to the parent of this script's directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
API_PATH = PROJECT_ROOT / "pinnfluence/frontend/api_server.py"
STREAMLIT_PATH = PROJECT_ROOT / "pinnfluence/frontend/visualization_streamlit.py"
LOGS_DIR = PROJECT_ROOT / "logs"
API_LOG = LOGS_DIR / "api_server.log"
STREAMLIT_LOG = LOGS_DIR / "streamlit.log"

API_PORT = 8000
STREAMLIT_PORT = 8501

# Ensure logs directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def run_uvicorn():
    # Run uvicorn in the background, cwd=project root, log to file, DEBUG level
    api_log = open(API_LOG, "w")
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "pinnfluence.frontend.api_server:app",
            "--log-level",
            "debug",
            f"--port={API_PORT}",
            "--workers=4",
        ],
        cwd=PROJECT_ROOT,
        stdout=api_log,
        stderr=subprocess.STDOUT,
    )
    return proc


def run_streamlit():
    # Run streamlit in the background, cwd=project root, log to file, DEBUG level
    streamlit_log = open(STREAMLIT_LOG, "w")
    return subprocess.Popen(
        [
            "streamlit",
            "run",
            str(STREAMLIT_PATH),
            "--logger.level=debug",
            f"--server.port={STREAMLIT_PORT}",
        ],
        cwd=PROJECT_ROOT,
        stdout=streamlit_log,
        stderr=subprocess.STDOUT,
    )


def check_api_log_for_address_in_use():
    if API_LOG.exists():
        with open(API_LOG) as f:
            log_content = f.read()
            if "Address already in use" in log_content:
                print(
                    f"[ERROR] API server failed to start: Address already in use. I.e., port {API_PORT} is in use."
                )
                print(f"Check {API_LOG} for details.")
                sys.exit(1)


def main(api_port=8000, streamlit_port=8501):
    global API_PORT, STREAMLIT_PORT
    API_PORT = api_port if api_port is not None else API_PORT
    STREAMLIT_PORT = streamlit_port if streamlit_port is not None else STREAMLIT_PORT
    print(f"Project root: {PROJECT_ROOT}")
    print("Starting FastAPI server...")
    api_proc = run_uvicorn()
    time.sleep(2)  # Give the API server a moment to start
    check_api_log_for_address_in_use()

    print("Starting Streamlit app...")
    streamlit_proc = run_streamlit()

    print("Both servers started. Press Ctrl+C to stop.")

    try:
        # Wait for both processes to finish (they won't, unless killed)
        api_proc.wait()
        streamlit_proc.wait()
    except KeyboardInterrupt:
        print("Shutting down...")
        api_proc.terminate()
        streamlit_proc.terminate()


if __name__ == "__main__":
    main()
