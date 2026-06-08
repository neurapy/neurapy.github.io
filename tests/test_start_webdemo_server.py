from __future__ import annotations

import threading
from contextlib import contextmanager
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

from start_webdemo_server import create_server, local_url, network_url, parse_args


@contextmanager
def running_webdemo_server():
    server = create_server("127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def read_url(url: str) -> bytes:
    with urlopen(url, timeout=5) as response:
        return response.read()


def test_parse_args_defaults_to_network_accessible_webdemo_port() -> None:
    args = parse_args([])

    assert args.host == "0.0.0.0"
    assert args.port == 8080


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", "http://127.0.0.1:8080/"),
        ("0.0.0.0", "http://127.0.0.1:8080/"),
        ("::1", "http://[::1]:8080/"),
    ],
)
def test_local_url(host: str, expected: str) -> None:
    assert local_url(host, 8080) == expected


def test_network_url_is_only_shown_for_wildcard_hosts() -> None:
    assert network_url("127.0.0.1", 8080) is None
    assert network_url("0.0.0.0", 8080) is not None


def test_server_serves_webdemo_index() -> None:
    with running_webdemo_server() as base_url:
        body = read_url(f"{base_url}/")

    assert b"PINNfluence Demo" in body


def test_server_serves_mjs_modules_as_javascript() -> None:
    with running_webdemo_server() as base_url:
        with urlopen(f"{base_url}/plotGeometry.mjs", timeout=5) as response:
            content_type = response.headers.get_content_type()
            body = response.read()

    assert content_type == "text/javascript"
    assert b"plotViewport" in body


def test_server_disables_directory_listing() -> None:
    with running_webdemo_server() as base_url:
        with pytest.raises(HTTPError) as exc_info:
            read_url(f"{base_url}/data/")

    assert exc_info.value.code == 404
