"""The MCP tools. Names and arguments follow the colab-mcp fork.

Cell primitives (get_cells, add_code_cell, add_text_cell, update_cell,
move_cell, delete_cell, run_code_cell) use Colab's argument names so prompts and
habits carry over. The composite tools (outline, run_cells, run_notebook,
run_code, jobs, upload/download, export) behave the same. Two differences:
cancel_job can really interrupt the running cell (Jupyter supports it), and
there are Kaggle sync tools (pull_notebook, save_notebook).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools.tool import ToolResult
from mcp.types import TextContent

from kaggle_mcp.document import CellNotFound
from kaggle_mcp.outputs import (
    result_with_images,
    strip_images,
    summarize_outputs,
    truncate,
)
from kaggle_mcp.session import KaggleSession

UPLOAD_CHUNK_BYTES = int(os.environ.get("KAGGLE_MCP_UPLOAD_CHUNK", str(1024 * 1024)))
DOWNLOAD_CHUNK_BYTES = int(
    os.environ.get("KAGGLE_MCP_DOWNLOAD_CHUNK", str(1024 * 1024))
)
MAX_JOBS = 50


# -- jobs ----------------------------------------------------------------------


@dataclass
class Job:
    id: str
    kind: str
    state: str = "running"  # running | done | failed | cancelled
    started: float = field(default_factory=time.time)
    finished: float | None = None
    total: int = 0
    current: str | None = None
    results: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    cancel_requested: bool = False
    task: asyncio.Task | None = None

    def view(self, max_chars: int) -> dict[str, Any]:
        results = []
        for r in self.results:
            r = dict(r)
            r.pop("_images", None)
            for key in ("stdout", "stderr", "result"):
                if isinstance(r.get(key), str):
                    r[key], _ = truncate(r[key], max_chars)
            results.append(r)
        return {
            "job_id": self.id,
            "kind": self.kind,
            "state": self.state,
            "started": self.started,
            "finished": self.finished,
            "elapsed_seconds": round((self.finished or time.time()) - self.started, 1),
            "total_cells": self.total,
            "completed_cells": len(self.results),
            "current_cell": self.current,
            "cancel_requested": self.cancel_requested,
            "error": self.error,
            "results": results,
        }


class Jobs:
    def __init__(self):
        self.jobs: dict[str, Job] = {}

    def register(self, job: Job) -> None:
        self.jobs[job.id] = job
        while len(self.jobs) > MAX_JOBS:
            oldest = next(iter(self.jobs))
            if self.jobs[oldest].state == "running":
                break
            del self.jobs[oldest]

    def start(self, kind: str, coro) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        job.task = asyncio.create_task(coro(job))
        self.register(job)
        return job

    def get(self, job_id: str) -> Job:
        job = self.jobs.get(job_id)
        if job is None:
            raise ToolError(f"Unknown job {job_id}")
        return job


# -- helpers -------------------------------------------------------------------


def _local(path: str) -> Path:
    return Path(path).expanduser().resolve()


def _source(cell: dict) -> str:
    return cell.get("source") or ""


def _cell_view(cell: dict, include_outputs: bool) -> dict:
    view = {
        "id": cell["id"],
        "cell_type": cell["cell_type"],
        "source": _source(cell),
    }
    if cell["cell_type"] == "code":
        view["execution_count"] = cell.get("execution_count")
        if include_outputs:
            view["outputs"] = cell.get("outputs") or []
    return view


_HASH_SNIPPET = """
import hashlib, os, json
_p = os.path.expanduser({path!r})
_h = hashlib.sha256()
with open(_p, 'rb') as _f:
    for _chunk in iter(lambda: _f.read(1 << 20), b''):
        _h.update(_chunk)
print(json.dumps({{"sha256": _h.hexdigest(), "size": os.path.getsize(_p)}}))
"""

_READ_SNIPPET = """
import base64, os
_p = os.path.expanduser({path!r})
with open(_p, 'rb') as _f:
    _f.seek({offset})
    _d = _f.read({size})
print('<<B64>>' + base64.b64encode(_d).decode() + '<<END>>')
"""


def register_tools(mcp: FastMCP, session: KaggleSession) -> Jobs:
    jobs = Jobs()

    def doc():
        return session.require_doc()

    def cell_or_error(cell_id: str) -> dict:
        try:
            return doc().get(cell_id)
        except CellNotFound:
            raise ToolError(f"Unknown cell id {cell_id}") from None

    async def snippet(code: str, max_chars: int = 0) -> dict[str, Any]:
        reply = await session.execute(code)
        return summarize_outputs(reply.get("outputs"), max_chars)

    async def run_cell(cell_id: str, max_chars: int) -> dict[str, Any]:
        cell = cell_or_error(cell_id)
        if cell["cell_type"] != "code":
            raise ToolError(f"Cell {cell_id} is a {cell['cell_type']} cell, not code")
        reply = await session.execute(_source(cell))
        outputs = reply.get("outputs") or []
        try:
            doc().set_outputs(cell_id, outputs, reply.get("execution_count"))
        except CellNotFound:
            pass  # deleted while it ran
        summary = summarize_outputs(outputs, max_chars)
        summary["cell_id"] = cell_id
        summary["execution_count"] = reply.get("execution_count")
        return summary

    async def run_sequence(
        job: Job, cell_ids: list[str], stop_on_error: bool, max_chars: int
    ) -> None:
        job.total = len(cell_ids)
        try:
            for cell_id in cell_ids:
                if job.cancel_requested:
                    job.state = "cancelled"
                    break
                job.current = cell_id
                summary = await run_cell(cell_id, max_chars)
                job.results.append(summary)
                if not summary["ok"] and job.cancel_requested:
                    job.state = "cancelled"
                    break
                if not summary["ok"] and stop_on_error:
                    job.state = "failed"
                    job.error = f"cell {cell_id} raised {summary['error']['ename']}: {summary['error']['evalue']}"
                    break
            else:
                job.state = "done" if all(r["ok"] for r in job.results) else "failed"
        except asyncio.CancelledError:
            job.state = "cancelled"
            raise
        except Exception as exc:  # the kernel dropped, or the session ended
            job.state = "failed"
            job.error = str(exc)
        finally:
            job.current = None
            job.finished = time.time()

    def select_cells(
        cell_ids: list[str] | None, start_index: int | None, end_index: int | None
    ) -> list[str]:
        cells = doc().cells
        if cell_ids:
            known = {c["id"] for c in cells}
            missing = [c for c in cell_ids if c not in known]
            if missing:
                raise ToolError(f"Unknown cell ids: {missing}")
            return list(cell_ids)
        start = 0 if start_index is None else max(0, start_index)
        end = len(cells) if end_index is None else min(len(cells), end_index)
        return [c["id"] for c in cells[start:end] if c["cell_type"] == "code"]

    # -- connection ------------------------------------------------------------

    @mcp.tool
    async def open_kaggle_connection(
        notebook_url: str | None = None,
        server_url: str | None = None,
        accelerator: str | None = None,
        enable_internet: bool | None = None,
        start_session: bool = True,
        kernel_id: str | None = None,
        new_kernel: bool = False,
        refresh: bool = False,
    ) -> dict:
        """Connect to a Kaggle notebook and a live kernel to run its code on.

        notebook_url: a kaggle.com/code/<owner>/<slug> link (with or without /edit) or
        owner/slug. Its cells are pulled into a local working copy for the cell tools.
        server_url: the 'VS Code compatible URL' from Run > Kaggle Jupyter Server in the
        Kaggle editor. Without it, the saved server from last time is reused, else (if
        start_session) a session is started through the Kaggle API with `accelerator`
        (t4, l4, a100, tpu, or a Kaggle machine shape id). If none of that yields a server,
        the notebook opens in the browser and the result says what the user must do.
        A running kernel on the server is reused (so variables carry over) unless
        new_kernel=True or kernel_id names another one. refresh=True re-pulls the
        notebook even if the working copy is current.
        """
        return await session.open(
            notebook_url=notebook_url,
            server_url=server_url,
            accelerator=accelerator,
            enable_internet=enable_internet,
            start_session=start_session,
            kernel_id=kernel_id,
            new_kernel=new_kernel,
            refresh=refresh,
        )

    @mcp.tool
    async def kaggle_connection_status() -> dict:
        """Which notebook is open, whether the kernel is alive, the runtime (GPU/TPU, inputs), unsaved edits, and the Kaggle session state. Opens nothing."""
        return await session.status()

    @mcp.tool
    async def disconnect_kaggle(
        shutdown_kernel: bool = False, stop_session: bool = False
    ) -> dict:
        """Detach from the kernel. shutdown_kernel kills it (losing its state); stop_session also stops an API-started Kaggle session to free accelerator quota."""
        return await session.disconnect(shutdown_kernel, stop_session)

    # -- cell primitives (Colab names) -----------------------------------------

    @mcp.tool
    async def get_cells(includeOutputs: bool = False) -> dict:
        """All cells of the open notebook with ids, type and source (and raw outputs if includeOutputs)."""
        return {"cells": [_cell_view(c, includeOutputs) for c in doc().cells]}

    @mcp.tool
    async def add_code_cell(
        code: str, cellIndex: int | None = None, language: str = "python"
    ) -> dict:
        """Insert a code cell at cellIndex (default: end). Returns newCellId."""
        return {"newCellId": doc().add(cellIndex, "code", code)}

    @mcp.tool
    async def add_text_cell(content: str, cellIndex: int | None = None) -> dict:
        """Insert a markdown cell at cellIndex (default: end). Returns newCellId."""
        return {"newCellId": doc().add(cellIndex, "markdown", content)}

    @mcp.tool
    async def update_cell(cellId: str, content: str) -> dict:
        """Replace a cell's source."""
        cell_or_error(cellId)
        doc().update(cellId, content)
        return {"cellId": cellId, "updated": True}

    @mcp.tool
    async def move_cell(cellId: str, cellIndex: int) -> dict:
        """Move a cell to cellIndex."""
        cell_or_error(cellId)
        doc().move(cellId, cellIndex)
        return {"cellId": cellId, "cellIndex": doc().index_of(cellId)}

    @mcp.tool
    async def delete_cell(cellId: str) -> dict:
        """Delete a cell."""
        cell_or_error(cellId)
        doc().delete(cellId)
        return {"cellId": cellId, "deleted": True}

    @mcp.tool
    async def run_code_cell(cellId: str, max_output_chars: int = 8000) -> ToolResult:
        """Run one code cell on the Kaggle kernel and store its outputs in the notebook."""
        summary = await run_cell(cellId, max_output_chars)
        summary, images = strip_images(summary)
        return result_with_images(summary, images)

    # -- composite tools -------------------------------------------------------

    @mcp.tool
    async def get_notebook_outline() -> dict:
        """Cells as index, id, type, first source line and whether they have outputs or errors. Cheaper than get_cells."""
        cells = doc().cells
        outline = []
        for index, cell in enumerate(cells):
            source = _source(cell)
            first = source.strip().splitlines()[0] if source.strip() else ""
            outputs = cell.get("outputs") or []
            outline.append(
                {
                    "index": index,
                    "id": cell["id"],
                    "type": cell["cell_type"],
                    "first_line": first[:120],
                    "lines": source.count("\n") + (1 if source else 0),
                    "has_outputs": bool(outputs),
                    "has_error": any(o.get("output_type") == "error" for o in outputs),
                }
            )
        return {
            "cell_count": len(cells),
            "unsaved_changes": doc().dirty,
            "cells": outline,
        }

    async def run_cells_impl(
        cell_ids: list[str] | None,
        start_index: int | None,
        end_index: int | None,
        stop_on_error: bool,
        background: bool,
        max_output_chars: int,
    ) -> dict:
        ids = select_cells(cell_ids, start_index, end_index)
        if not ids:
            raise ToolError("No code cells matched")
        if background:
            job = jobs.start(
                "run_cells",
                lambda j: run_sequence(j, ids, stop_on_error, max_output_chars),
            )
            return {
                "job_id": job.id,
                "state": job.state,
                "total_cells": len(ids),
                "poll_with": "job_status",
            }
        job = Job(id=uuid.uuid4().hex[:12], kind="run_cells")
        jobs.register(job)
        await run_sequence(job, ids, stop_on_error, max_output_chars)
        return job.view(max_output_chars)

    @mcp.tool
    async def run_cells(
        cell_ids: list[str] | None = None,
        start_index: int | None = None,
        end_index: int | None = None,
        stop_on_error: bool = True,
        background: bool = False,
        max_output_chars: int = 4000,
    ) -> dict:
        """Run code cells in order: by id list, by index range (end exclusive), or all.

        For anything that may take more than a minute set background=True and poll
        job_status. cancel_job interrupts the running cell.
        """
        return await run_cells_impl(
            cell_ids,
            start_index,
            end_index,
            stop_on_error,
            background,
            max_output_chars,
        )

    @mcp.tool
    async def run_notebook(
        stop_on_error: bool = True,
        background: bool = True,
        max_output_chars: int = 2000,
    ) -> dict:
        """Run every code cell from the top. Background by default: poll job_status."""
        return await run_cells_impl(
            None, None, None, stop_on_error, background, max_output_chars
        )

    @mcp.tool
    async def run_code(
        code: str,
        keep_cell: bool = False,
        background: bool = False,
        timeout_seconds: float | None = None,
        max_output_chars: int = 8000,
    ) -> ToolResult:
        """Run ad-hoc Python on the Kaggle kernel without touching the notebook.

        keep_cell=True appends it to the notebook as a cell instead. Plots come back
        as image content. background=True returns a job id.
        """
        if not code.strip():
            raise ToolError("code is empty")

        async def once() -> dict:
            if keep_cell:
                cell_id = doc().add(None, "code", code)
                return await run_cell(cell_id, max_output_chars)
            reply = await session.execute(code, timeout=timeout_seconds)
            summary = summarize_outputs(reply.get("outputs"), max_output_chars)
            summary["cell_id"] = None
            summary["execution_count"] = reply.get("execution_count")
            return summary

        if background:

            async def body(job: Job):
                job.total = 1
                try:
                    summary = await once()
                    job.results.append(summary)
                    if summary["ok"]:
                        job.state = "done"
                    elif job.cancel_requested:
                        job.state = "cancelled"
                    else:
                        job.state = "failed"
                        job.error = (
                            f"{summary['error']['ename']}: {summary['error']['evalue']}"
                        )
                except Exception as exc:
                    job.state = "failed"
                    job.error = str(exc)
                finally:
                    job.finished = time.time()

            job = jobs.start("run_code", body)
            payload = {"job_id": job.id, "state": "running", "poll_with": "job_status"}
            return ToolResult(
                content=[TextContent(type="text", text=json.dumps(payload))],
                structured_content=payload,
            )
        try:
            summary = await once()
        except TimeoutError:
            raise ToolError(
                f"No reply within {timeout_seconds}s; the code may still be running. "
                "interrupt_kernel stops it."
            ) from None
        summary, images = strip_images(summary)
        return result_with_images(summary, images)

    @mcp.tool
    async def job_status(job_id: str, max_output_chars: int = 2000) -> dict:
        """Progress and per-cell results of a background run."""
        return jobs.get(job_id).view(max_output_chars)

    @mcp.tool
    async def list_jobs() -> dict:
        """All known background runs, newest last."""
        return {
            "jobs": [job.view(200) | {"results": None} for job in jobs.jobs.values()]
        }

    @mcp.tool
    async def cancel_job(job_id: str, interrupt: bool = True) -> dict:
        """Stop a background run. interrupt=True also interrupts the cell running now (KeyboardInterrupt in the kernel)."""
        job = jobs.get(job_id)
        if job.state != "running":
            return {
                "job_id": job.id,
                "state": job.state,
                "note": "job already finished",
            }
        job.cancel_requested = True
        interrupted = False
        if interrupt and session.kernel is not None:
            await session.kernel.interrupt()
            interrupted = True
        return {
            "job_id": job.id,
            "state": job.state,
            "cancel_requested": True,
            "interrupted": interrupted,
        }

    @mcp.tool
    async def interrupt_kernel() -> dict:
        """Send KeyboardInterrupt to whatever is running on the kernel."""
        kernel = await session.ensure_kernel()
        await kernel.interrupt()
        return {"interrupted": True, "kernel_id": kernel.kernel_id}

    @mcp.tool
    async def restart_kernel() -> dict:
        """Restart the kernel, clearing all variables. Files on disk are kept."""
        kernel = await session.ensure_kernel()
        await kernel.restart()
        return {"restarted": True, "kernel_id": kernel.kernel_id}

    # -- files -----------------------------------------------------------------

    async def remote_hash(remote_path: str) -> dict:
        summary = await snippet(_HASH_SNIPPET.format(path=remote_path), 2000)
        if not summary["ok"]:
            raise ToolError(
                f"Could not hash {remote_path} on the kernel: {summary['error']['evalue']}"
            )
        try:
            return json.loads(summary["stdout"].strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise ToolError(
                f"Unexpected hash output: {summary['stdout'][:200]}"
            ) from exc

    @mcp.tool
    async def upload_file(
        local_path: str, remote_path: str, overwrite: bool = True
    ) -> dict:
        """Copy a local file onto the Kaggle machine (e.g. /kaggle/working/data.csv), verified by SHA-256."""
        src = _local(local_path)
        if not src.is_file():
            raise ToolError(f"Local file not found: {src}")
        data = src.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if not overwrite:
            probe = await snippet(
                f"import os; print(os.path.exists(os.path.expanduser({remote_path!r})))",
                200,
            )
            if probe["ok"] and probe["stdout"].strip() == "True":
                raise ToolError(f"{remote_path} exists on Kaggle and overwrite=False")
        chunks = [
            data[i : i + UPLOAD_CHUNK_BYTES]
            for i in range(0, len(data), UPLOAD_CHUNK_BYTES)
        ] or [b""]
        for index, chunk in enumerate(chunks):
            mode = "wb" if index == 0 else "ab"
            code = (
                "import base64, os\n"
                f"_p = os.path.expanduser({remote_path!r})\n"
                "os.makedirs(os.path.dirname(_p) or '.', exist_ok=True)\n"
                f"with open(_p, {mode!r}) as _f:\n"
                f"    _f.write(base64.b64decode({base64.b64encode(chunk).decode()!r}))\n"
            )
            summary = await snippet(code, 500)
            if not summary["ok"]:
                raise ToolError(
                    f"Upload chunk {index + 1}/{len(chunks)} failed: {summary['error']['evalue']}"
                )
        remote = await remote_hash(remote_path)
        if remote["sha256"] != digest:
            raise ToolError(
                f"Upload verification failed: local sha256 {digest} != remote {remote['sha256']}"
            )
        return {
            "remote_path": remote_path,
            "bytes": len(data),
            "sha256": digest,
            "chunks": len(chunks),
        }

    @mcp.tool
    async def download_file(
        remote_path: str, local_path: str, overwrite: bool = False
    ) -> dict:
        """Copy a file from the Kaggle machine to this one, verified by SHA-256."""
        dst = _local(local_path)
        if dst.exists() and not overwrite:
            raise ToolError(f"{dst} exists locally and overwrite=False")
        remote = await remote_hash(remote_path)
        size = int(remote["size"])
        buffer = bytearray()
        offset = 0
        chunks = 0
        while offset < size or chunks == 0:
            summary = await snippet(
                _READ_SNIPPET.format(
                    path=remote_path, offset=offset, size=DOWNLOAD_CHUNK_BYTES
                )
            )
            if not summary["ok"]:
                raise ToolError(
                    f"Download failed at offset {offset}: {summary['error']['evalue']}"
                )
            text = summary["stdout"]
            start, end = text.find("<<B64>>"), text.rfind("<<END>>")
            if start < 0 or end < 0:
                raise ToolError(
                    f"Download chunk at offset {offset} came back truncated; lower KAGGLE_MCP_DOWNLOAD_CHUNK"
                )
            piece = base64.b64decode(text[start + 7 : end].strip())
            chunks += 1
            if not piece:
                if size > 0:
                    raise ToolError(f"Empty chunk at offset {offset}")
                break
            buffer.extend(piece)
            offset += len(piece)
        digest = hashlib.sha256(bytes(buffer)).hexdigest()
        if digest != remote["sha256"]:
            raise ToolError(
                f"Download verification failed: remote sha256 {remote['sha256']} != local {digest}"
            )
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(bytes(buffer))
        return {
            "local_path": str(dst),
            "bytes": len(buffer),
            "sha256": digest,
            "chunks": chunks,
        }

    @mcp.tool
    async def export_notebook(local_path: str, include_outputs: bool = True) -> dict:
        """Save the open notebook as an .ipynb file on this machine."""
        dst = _local(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(doc().to_ipynb(include_outputs), indent=1))
        return {
            "local_path": str(dst),
            "cell_count": len(doc().cells),
            "bytes": dst.stat().st_size,
        }

    # -- Kaggle sync -----------------------------------------------------------

    @mcp.tool
    async def pull_notebook(discard_local_changes: bool = False) -> dict:
        """Replace the working copy with the notebook's latest version on Kaggle. Refuses if there are unsaved edits unless discard_local_changes."""
        return await session.pull(discard_local_changes)

    @mcp.tool
    async def save_notebook(
        quick_save: bool = True,
        include_outputs: bool = False,
        accelerator: str | None = None,
    ) -> dict:
        """Save the working copy to Kaggle as a new notebook version.

        quick_save=True (default) saves without running. quick_save=False is Kaggle's
        'Save & Run All': it runs the whole notebook in a fresh batch session (using
        quota; `accelerator` picks the machine). The notebook's data sources, privacy
        and internet settings are kept. Outputs are stripped unless include_outputs.
        """
        logging.info("Saving %s to Kaggle (quick_save=%s)", session.ref, quick_save)
        return await session.save(quick_save, include_outputs, accelerator)

    return jobs
