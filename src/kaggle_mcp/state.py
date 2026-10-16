"""Persisted connection so a restart reattaches to the same Kaggle kernel.

The Jupyter server URL is a credential, so the file is owner-readable only.
Working copies of pulled notebooks live next to it, one .ipynb per notebook.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

DEFAULT_STATE_DIR = Path.home() / ".kaggle-mcp"
STATE_FILE = "connection.json"


@dataclass
class ConnectionState:
    notebook_ref: str | None = None
    server_url: str | None = None
    kernel_id: str | None = None
    kaggle_session_id: int | None = None
    # URLs from an API-started session that has not been attached yet.
    candidate_server_urls: list[str] | None = None


def default_state_dir() -> Path:
    override = os.environ.get("KAGGLE_MCP_STATE_DIR")
    return Path(override).expanduser() if override else DEFAULT_STATE_DIR


def notebook_path(state_dir: Path, ref: str | None) -> Path:
    name = ref.replace("/", "__") if ref else "_scratch"
    return state_dir / "notebooks" / f"{name}.ipynb"


def load_state(state_dir: Path) -> ConnectionState:
    path = state_dir / STATE_FILE
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        return ConnectionState()
    except (OSError, ValueError) as exc:
        logging.warning("Ignoring unreadable connection state at %s: %s", path, exc)
        return ConnectionState()
    if not isinstance(raw, dict):
        return ConnectionState()
    known = {f.name for f in fields(ConnectionState)}
    state = ConnectionState(**{k: v for k, v in raw.items() if k in known})
    if not isinstance(state.kaggle_session_id, int):
        state.kaggle_session_id = None
    urls = state.candidate_server_urls
    if not isinstance(urls, list) or not all(isinstance(u, str) for u in urls):
        state.candidate_server_urls = None
    for name in ("notebook_ref", "server_url", "kernel_id"):
        if not isinstance(getattr(state, name), str):
            setattr(state, name, None)
    return state


def write_private(path: Path, text: str) -> None:
    """Atomically write ``text`` to ``path`` with 0600 permissions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
    os.replace(tmp, path)


def save_state(state_dir: Path, state: ConnectionState) -> None:
    write_private(state_dir / STATE_FILE, json.dumps(asdict(state), indent=2))
