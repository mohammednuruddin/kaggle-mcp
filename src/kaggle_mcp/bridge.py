"""The Kaggle editor as a notebook the tools can work on.

browser.py opens the notebook editor in a browser tab kaggle-mcp controls and
evaluates editor_bridge.js there; LiveNotebook turns the tool operations into
calls on that script (getNotebook, insertCell, runCell, runState, ...), so cells
the agent adds and runs appear in the editor as they happen.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Protocol

from fastmcp.exceptions import ToolError

REQUEST_TIMEOUT = float(os.environ.get("KAGGLE_MCP_EDITOR_TIMEOUT", "60"))
# How long a cell may wait for a stopped Kaggle session to start before giving up.
SESSION_START_WAIT = float(os.environ.get("KAGGLE_MCP_SESSION_START_WAIT", "300"))
POLL_INTERVAL = 0.5


class BridgeError(ToolError):
    """The editor refused a request or went away (shown to the agent as a tool error)."""


class Editor(Protocol):
    ref: str

    def describe(self) -> dict[str, Any]: ...

    async def request(
        self, method: str, params: dict | None = None, timeout: float | None = None
    ) -> Any: ...


# -- the notebook as the agent sees it -------------------------------------------


def _join(text: Any) -> str:
    if isinstance(text, list):
        return "".join(str(t) for t in text)
    return "" if text is None else str(text)


def normalize_cell(cell: dict) -> dict:
    kind = cell.get("cell_type", "code")
    out = {
        "id": cell.get("id"),
        "cell_type": kind,
        "metadata": dict(cell.get("metadata") or {}),
        "source": _join(cell.get("source")),
    }
    if kind == "code":
        out["execution_count"] = cell.get("execution_count")
        out["outputs"] = list(cell.get("outputs") or [])
    return out


class LiveNotebook:
    """The notebook open in a Kaggle editor tab, edited and run through the bridge."""

    kind = "editor"
    dirty = None  # Kaggle autosaves the editor's draft itself.

    def __init__(self, editor: Editor):
        self.editor = editor

    async def notebook(self) -> dict:
        return await self.editor.request("getNotebook")

    async def cells(self) -> list[dict]:
        nb = await self.notebook()
        return [normalize_cell(c) for c in nb.get("cells") or []]

    async def add(self, index: int | None, cell_type: str, source: str) -> str:
        reply = await self.editor.request(
            "insertCell", {"index": index, "cell_type": cell_type, "source": source}
        )
        return reply["id"]

    async def update(self, cell_id: str, source: str) -> None:
        await self.editor.request("updateCell", {"id": cell_id, "source": source})

    async def move(self, cell_id: str, index: int) -> int:
        reply = await self.editor.request("moveCell", {"id": cell_id, "index": index})
        return reply["index"]

    async def delete(self, cell_id: str) -> None:
        await self.editor.request("deleteCell", {"id": cell_id})

    async def run(self, cell_id: str, timeout: float | None = None) -> dict:
        """Run a code cell in the editor; returns {"outputs", "execution_count"}."""
        started = await self.editor.request("runCell", {"id": cell_id})
        token = started["token"]
        deadline = None if timeout is None else time.monotonic() + timeout
        queued_since = None
        while True:
            state = await self.editor.request(
                "runState", {"token": token, "id": cell_id}
            )
            if state.get("deleted"):
                raise BridgeError(f"Cell {cell_id} was deleted while it ran.")
            if state.get("done"):
                if state.get("error"):
                    raise BridgeError(f"The editor reported: {state['error']}")
                return {
                    "outputs": state.get("outputs") or [],
                    "execution_count": state.get("execution_count"),
                }
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(cell_id)
            if state.get("queued"):
                queued_since = queued_since or time.monotonic()
                if time.monotonic() - queued_since > SESSION_START_WAIT:
                    raise BridgeError(
                        f"Cell {cell_id} is still waiting for the Kaggle session to start "
                        f"after {SESSION_START_WAIT:.0f}s. Check the editor (the session "
                        "may need an accelerator or quota); the cell runs once it starts."
                    )
            else:
                queued_since = None
            await asyncio.sleep(POLL_INTERVAL)

    async def interrupt(self) -> None:
        await self.editor.request("interrupt")

    async def restart(self) -> None:
        await self.editor.request("restart")

    async def to_ipynb(self, include_outputs: bool = True) -> dict:
        nb = await self.notebook()
        cells = []
        for cell in nb.get("cells") or []:
            cell = normalize_cell(cell)
            if cell["cell_type"] == "code" and not include_outputs:
                cell["outputs"] = []
                cell["execution_count"] = None
            cells.append(cell)
        return {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": nb.get("metadata") or {},
            "cells": cells,
        }
