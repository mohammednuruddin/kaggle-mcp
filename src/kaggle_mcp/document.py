"""The notebook being edited: a local nbformat-4 working copy with cell ids.

Colab keeps the notebook in the browser and exposes cell primitives over its
bridge. Kaggle's Jupyter server only runs code, so the cells live here instead:
pulled from Kaggle, edited by the cell tools, written to disk after every
change, and pushed back as a new notebook version on save.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from kaggle_mcp.state import write_private


class CellNotFound(KeyError):
    pass


def _join(text: Any) -> str:
    if isinstance(text, list):
        return "".join(str(t) for t in text)
    return "" if text is None else str(text)


def _new_id() -> str:
    return uuid.uuid4().hex[:8]


def _normalize_cell(cell: dict, seen: set[str]) -> dict:
    kind = cell.get("cell_type", "code")
    cell_id = cell.get("id") or (cell.get("metadata") or {}).get("id")
    if not isinstance(cell_id, str) or not cell_id or cell_id in seen:
        cell_id = _new_id()
    seen.add(cell_id)
    out: dict[str, Any] = {
        "id": cell_id,
        "cell_type": kind,
        "metadata": dict(cell.get("metadata") or {}),
        "source": _join(cell.get("source")),
    }
    if kind == "code":
        out["execution_count"] = cell.get("execution_count")
        out["outputs"] = list(cell.get("outputs") or [])
    elif "attachments" in cell:
        out["attachments"] = cell["attachments"]
    return out


class NotebookDocument:
    def __init__(
        self,
        cells: list[dict] | None = None,
        metadata: dict | None = None,
        path: Path | None = None,
        ref: str | None = None,
        dirty: bool = False,
        base_version: int | None = None,
    ):
        seen: set[str] = set()
        self.cells = [_normalize_cell(c, seen) for c in cells or []]
        self.metadata = metadata or {
            "kernelspec": {
                "name": "python3",
                "display_name": "Python 3",
                "language": "python",
            },
            "language_info": {"name": "python"},
        }
        self.path = path
        self.ref = ref
        # True once the working copy differs from what was last pulled/pushed.
        self.dirty = dirty
        self.base_version = base_version

    # -- (de)serialisation -----------------------------------------------------

    @classmethod
    def from_ipynb(
        cls, data: dict | str, path: Path | None = None, ref: str | None = None
    ) -> NotebookDocument:
        if isinstance(data, str):
            data = json.loads(data) if data.strip() else {}
        if not isinstance(data, dict):
            raise ValueError("notebook JSON must be an object")
        meta = data.get("metadata") or {}
        kaggle_mcp = meta.pop("kaggle_mcp", {}) if isinstance(meta, dict) else {}
        return cls(
            cells=data.get("cells") or [],
            metadata=meta or None,
            path=path,
            ref=ref or kaggle_mcp.get("ref"),
            dirty=bool(kaggle_mcp.get("dirty", False)),
            base_version=kaggle_mcp.get("base_version"),
        )

    @classmethod
    def load(cls, path: Path) -> NotebookDocument:
        return cls.from_ipynb(path.read_text(), path=path)

    def to_ipynb(
        self, include_outputs: bool = True, include_state: bool = False
    ) -> dict:
        cells = []
        for cell in self.cells:
            entry = {k: v for k, v in cell.items() if k != "outputs"}
            if cell["cell_type"] == "code":
                entry["outputs"] = list(cell["outputs"]) if include_outputs else []
                if not include_outputs:
                    entry["execution_count"] = None
            cells.append(entry)
        metadata = dict(self.metadata)
        if include_state:
            metadata["kaggle_mcp"] = {
                "ref": self.ref,
                "dirty": self.dirty,
                "base_version": self.base_version,
            }
        return {
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": metadata,
            "cells": cells,
        }

    def save(self) -> None:
        if self.path is not None:
            write_private(
                self.path, json.dumps(self.to_ipynb(include_state=True), indent=1)
            )

    # -- cells -----------------------------------------------------------------

    def index_of(self, cell_id: str) -> int:
        for index, cell in enumerate(self.cells):
            if cell["id"] == cell_id:
                return index
        raise CellNotFound(cell_id)

    def get(self, cell_id: str) -> dict:
        return self.cells[self.index_of(cell_id)]

    def _touch(self) -> None:
        self.dirty = True
        self.save()

    def add(self, index: int | None, cell_type: str, source: str) -> str:
        if cell_type not in ("code", "markdown", "raw"):
            raise ValueError(f"unknown cell type {cell_type!r}")
        seen = {c["id"] for c in self.cells}
        cell = _normalize_cell({"cell_type": cell_type, "source": source}, seen)
        position = len(self.cells) if index is None else index
        position = max(0, min(len(self.cells), position))
        self.cells.insert(position, cell)
        self._touch()
        return cell["id"]

    def update(self, cell_id: str, source: str) -> None:
        cell = self.get(cell_id)
        cell["source"] = source
        self._touch()

    def move(self, cell_id: str, index: int) -> None:
        cell = self.cells.pop(self.index_of(cell_id))
        position = max(0, min(len(self.cells), index))
        self.cells.insert(position, cell)
        self._touch()

    def delete(self, cell_id: str) -> None:
        del self.cells[self.index_of(cell_id)]
        self._touch()

    def set_outputs(
        self, cell_id: str, outputs: list[dict], execution_count: int | None
    ) -> None:
        cell = self.get(cell_id)
        cell["outputs"] = outputs
        cell["execution_count"] = execution_count
        # Outputs alone do not make the copy differ from Kaggle's source.
        self.save()
