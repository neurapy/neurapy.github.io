#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import logging
import sys
import webbrowser
from collections.abc import Sequence
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WEBDEMO_DIR = REPO_ROOT / "webdemo"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080

LOGGER = logging.getLogger("webdemo")


class WebDemoRequestHandler(SimpleHTTPRequestHandler):
    """Static file handler scoped to the generated webdemo directory."""

    server_version = "PINNfluenceWebDemo/1.0"
    extensions_map = {
        **SimpleHTTPRequestHandler.extensions_map,
        ".css": "text/css",
        ".f32": "application/octet-stream",
        ".i16": "application/octet-stream",
        ".js": "text/javascript",
        ".json": "application/json",
        ".u8": "application/octet-stream",
        ".u32": "application/octet-stream",
    }

    def list_directory(self, path: str):
        self.send_error(404, "Directory listing disabled")
        return None

    def send_head(self):
        requested_path = Path(self.translate_path(self.path)).resolve()
        root = Path(self.directory).resolve()
        if not requested_path.is_relative_to(root):
            self.send_error(404, "File not found")
            return None
        return super().send_head()

    def log_message(self, message_format: str, *args: object) -> None:
        LOGGER.debug("%s - %s", self.address_string(), message_format % args)


class WebDemoServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Serve the static PINNfluence webdemo from ./webdemo.",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Host/interface to bind. Defaults to {DEFAULT_HOST}.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"TCP port to bind. Defaults to {DEFAULT_PORT}.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
        help="Open the demo in the default browser after the server starts.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-request debug logs.",
    )
    return parser.parse_args(argv)


def validate_webdemo_dir(webdemo_dir: Path = WEBDEMO_DIR) -> None:
    if not webdemo_dir.is_dir():
        raise FileNotFoundError(f"Webdemo directory not found: {webdemo_dir}")
    if not (webdemo_dir / "index.html").is_file():
        raise FileNotFoundError(f"Webdemo entrypoint not found: {webdemo_dir / 'index.html'}")


def make_handler(webdemo_dir: Path = WEBDEMO_DIR):
    return functools.partial(WebDemoRequestHandler, directory=str(webdemo_dir))


def create_server(host: str, port: int, webdemo_dir: Path = WEBDEMO_DIR) -> WebDemoServer:
    validate_webdemo_dir(webdemo_dir)
    return WebDemoServer((host, port), make_handler(webdemo_dir))


def local_url(host: str, port: int) -> str:
    display_host = "127.0.0.1" if host in {"", "0.0.0.0", "::"} else host
    if ":" in display_host and not display_host.startswith("["):
        display_host = f"[{display_host}]"
    return f"http://{display_host}:{port}/"


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


def run_server(host: str, port: int, *, open_browser: bool = False) -> int:
    with create_server(host, port) as server:
        actual_port = server.server_address[1]
        url = local_url(host, actual_port)
        print(f"Serving webdemo: {WEBDEMO_DIR}", flush=True)
        print(f"URL: {url}", flush=True)
        print("Press Ctrl+C to stop.", flush=True)

        if open_browser:
            webbrowser.open(url)

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping webdemo server.")
        finally:
            server.shutdown()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    try:
        return run_server(args.host, args.port, open_browser=args.open_browser)
    except OSError as exc:
        print(f"Could not start webdemo server on {args.host}:{args.port}: {exc}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
