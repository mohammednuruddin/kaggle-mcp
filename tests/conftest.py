import json
import os
import secrets
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from kaggle_mcp.kaggle_api import KaggleNotebook, StartedSession
from kaggle_mcp.refs import NotebookRef


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="session")
def jupyter_server(tmp_path_factory):
    """A real Jupyter server on localhost, standing in for Kaggle's session server."""
    root = tmp_path_factory.mktemp("jupyter-root")
    port = _free_port()
    token = secrets.token_hex(16)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "jupyter_server",
            "--no-browser",
            f"--port={port}",
            "--ServerApp.ip=127.0.0.1",
            f"--IdentityProvider.token={token}",
            f"--ServerApp.root_dir={root}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "JUPYTER_RUNTIME_DIR": str(root / ".runtime")},
    )
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            req = urllib.request.Request(
                f"{base}/api/status", headers={"Authorization": f"token {token}"}
            )
            urllib.request.urlopen(req, timeout=1)
            break
        except Exception:
            time.sleep(0.3)
    else:
        proc.kill()
        pytest.fail("jupyter server did not start")
    yield {"base": base, "token": token, "url": f"{base}/lab?token={token}"}
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


NOTEBOOK = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3"}},
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": ["# Demo\n", "intro"]},
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": ["x = 21\n", "print('x set')"],
        },
        {
            "cell_type": "code",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": "x * 2",
        },
    ],
}


class FakeApi:
    """Stands in for KaggleApi: one notebook, records saves."""

    def __init__(self, server_url: str | None = None):
        self.version = 3
        self.source = json.dumps(NOTEBOOK)
        self.saves: list[dict] = []
        self.session_server_url = server_url
        self.cancelled: list[int] = []
        self.sessions_started = 0

    configured = True

    def _nb(self, ref: NotebookRef) -> KaggleNotebook:
        return KaggleNotebook(
            ref=ref,
            source=self.source,
            language="python",
            kernel_type="notebook",
            title="Demo",
            version=self.version,
            metadata={
                "id": 42,
                "is_private": True,
                "enable_internet": True,
                "dataset_data_sources": ["owner/some-data"],
                "machine_shape": "NvidiaTeslaT4",
            },
        )

    async def get_notebook(self, ref):
        return self._nb(ref)

    async def save_notebook(self, notebook, text, quick_save=True, machine_shape=None):
        self.saves.append(
            {
                "text": text,
                "quick_save": quick_save,
                "machine_shape": machine_shape,
                "metadata": notebook.metadata,
            }
        )
        self.version += 1
        self.source = text
        return {"ref": notebook.ref.ref, "url": "u", "version_number": self.version}

    async def start_session(self, notebook, machine_shape, enable_internet, timeout):
        self.sessions_started += 1
        return StartedSession(
            server_urls=[self.session_server_url] if self.session_server_url else [],
            kaggle_session_id=777,
            operation={"done": True},
        )

    async def session_status(self, ref):
        return {"status": "RUNNING", "failure_message": None}

    async def cancel_session(self, session_id):
        self.cancelled.append(session_id)
        return None


@pytest.fixture
def state_dir(tmp_path) -> Path:
    return tmp_path / "state"
