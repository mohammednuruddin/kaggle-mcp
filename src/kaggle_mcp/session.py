"""Connection lifecycle: which notebook, which Jupyter server, which kernel.

Opening a notebook does three things:

1. pull its source through the Kaggle API into a local working copy (kept if
   it has unsaved edits);
2. find a Jupyter server for it: the URL the user pasted, the saved one from
   last time, or a new session started through the Kaggle API;
3. attach to a kernel there, reusing a running one so state carries over.

If no server can be found, the notebook editor is opened in the browser so
the user can start "Run > Kaggle Jupyter Server" and hand back the URL.

By default, though, opening a notebook opens its editor in a browser kaggle-mcp
controls (browser.py: the user's Edge/Chrome, or its own window) and the cell
tools work on that live editor: cells the agent adds and runs appear in the tab,
and no Jupyter server URL is needed. use_browser=False (or no browser) keeps the
local-working-copy mode.
"""

from __future__ import annotations

import json
import logging
import os
import webbrowser
from pathlib import Path
from typing import Any
from collections.abc import Callable

from fastmcp.exceptions import ToolError

from kaggle_mcp.bridge import BridgeError, LiveNotebook
from kaggle_mcp.browser import (
    BrowserUnavailable,
    EditorTab,
    KaggleBrowser,
    SignInRequired,
)
from kaggle_mcp.document import CellNotFound, NotebookDocument
from kaggle_mcp.kaggle_api import (
    KaggleApi,
    KaggleApiError,
    KaggleNotebook,
    SessionAlreadyRunning,
    credentials_source,
    resolve_accelerator,
    url_shape,
)
from kaggle_mcp.kernel import KernelUnavailable, RemoteKernel
from kaggle_mcp.refs import (
    KAGGLE,
    NotebookRef,
    ServerEndpoint,
    UnsupportedNotebookUrl,
    UnsupportedServerUrl,
    parse_notebook_ref,
    parse_server_url,
    redact_url,
)
from kaggle_mcp.state import (
    ConnectionState,
    default_state_dir,
    load_state,
    notebook_path,
    save_state,
)

# How long to wait for a session started through the Kaggle API.
SESSION_START_TIMEOUT = float(os.environ.get("KAGGLE_MCP_SESSION_TIMEOUT", "300"))
# Reattach to the saved server/kernel when a tool runs with no live kernel.
AUTO_RECONNECT = os.environ.get("KAGGLE_MCP_AUTO_RECONNECT", "1") not in (
    "0",
    "false",
    "no",
)

# How long to wait for the Kaggle editor to load (it starts a session first).
EDITOR_WAIT = float(os.environ.get("KAGGLE_MCP_EDITOR_WAIT", "120"))

SERVER_INFO: dict = {"log_dir": None}

_PROBE = r"""
import json, os, platform, subprocess
_i = {"python": platform.python_version(), "cwd": os.getcwd(), "on_kaggle": os.path.isdir("/kaggle")}
try:
    _i["gpus"] = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
except Exception:
    _i["gpus"] = []
_i["tpu"] = bool(os.environ.get("TPU_ACCELERATOR_TYPE") or os.environ.get("TPU_NAME") or os.path.exists("/dev/accel0"))
_i["kaggle_input"] = sorted(os.listdir("/kaggle/input"))[:50] if os.path.isdir("/kaggle/input") else None
print(json.dumps(_i))
"""

EDITOR_NOTE = (
    "The notebook is open in {where}: cells you add, edit and run appear in its editor "
    "as they happen, and code runs on that editor's Kaggle session (the accelerator is "
    "picked in the editor's settings panel)."
)

MANUAL_STEPS = (
    "In the Kaggle notebook tab that just opened: Run > Kaggle Jupyter Server, pick the "
    "accelerator you want, start it, and copy the 'VS Code compatible URL'. Then call "
    "open_kaggle_connection again with server_url set to that URL (ask the user to paste it)."
)


class KaggleSession:
    def __init__(
        self,
        state_dir: Path | None = None,
        ephemeral: bool = False,
        api: KaggleApi | None = None,
        kernel_factory: Callable[[ServerEndpoint], RemoteKernel] = RemoteKernel,
        open_browser: Callable[[str], Any] = webbrowser.open_new,
        browser: KaggleBrowser | None = None,
    ):
        self.state_dir = state_dir or default_state_dir()
        self.ephemeral = ephemeral
        self.api = api or KaggleApi()
        self.kernel_factory = kernel_factory
        self.open_browser = open_browser
        self.browser = browser
        self.state = ConnectionState() if ephemeral else load_state(self.state_dir)
        self.ref: NotebookRef | None = (
            _maybe_ref(self.state.notebook_ref) if self.state.notebook_ref else None
        )
        self.notebook: KaggleNotebook | None = None
        self.doc: NotebookDocument | None = None
        self.kernel: RemoteKernel | None = None
        self.environment: dict | None = None
        self.last_session_attempt: dict | None = None

    # -- persistence -----------------------------------------------------------

    def _persist(self) -> None:
        if self.ephemeral:
            return
        try:
            save_state(self.state_dir, self.state)
        except OSError as exc:
            logging.warning("Could not persist state to %s: %s", self.state_dir, exc)

    def _doc_path(self, ref: NotebookRef | None) -> Path | None:
        if self.ephemeral:
            return None
        return notebook_path(self.state_dir, ref.ref if ref else None)

    # -- which notebook the cell tools work on ----------------------------------

    @property
    def editor_mode(self) -> bool:
        return (
            self.browser is not None
            and self.ref is not None
            and self.state.mode == "editor"
        )

    def editor(self) -> EditorTab | None:
        """The connected editor tab for the open notebook, if any (no I/O)."""
        if not self.editor_mode:
            return None
        return self.browser.editor_for(self.ref.ref)

    async def editor_tab(self) -> EditorTab | None:
        """The editor tab in editor mode, reopened if the tab or browser was closed."""
        if not self.editor_mode:
            return None
        tab = self.browser.editor_for(self.ref.ref)
        if tab is not None:
            return tab
        try:
            return await self.browser.open_editor(self.ref, EDITOR_WAIT)
        except ToolError as exc:
            raise ToolError(
                f"The Kaggle editor tab for {self.ref.ref} is gone and could not be "
                f"reopened: {exc}"
            ) from exc

    async def notebook_view(self) -> LiveNotebook | LocalNotebook:
        tab = await self.editor_tab()
        return LiveNotebook(tab) if tab is not None else LocalNotebook(self)

    # -- notebook source -------------------------------------------------------

    async def _load_doc(self, ref: NotebookRef | None, refresh: bool) -> list[str]:
        """Load the working copy for ``ref``; returns notes for the agent."""
        notes: list[str] = []
        path = self._doc_path(ref)
        local = None
        if path is not None and path.is_file():
            try:
                local = NotebookDocument.load(path)
            except (OSError, ValueError) as exc:
                notes.append(f"Ignored unreadable working copy {path}: {exc}")

        if ref is None:
            self.notebook = None
            self.doc = local or NotebookDocument(path=path)
            self.doc.save()
            notes.append(
                "No notebook_url given: using a local scratch notebook. "
                "save_notebook is unavailable until a Kaggle notebook is opened."
            )
            return notes

        if not self.api.configured:
            self.notebook = None
            self.doc = local or NotebookDocument(path=path, ref=ref.ref)
            self.doc.save()
            notes.append(
                "No Kaggle API credentials, so the notebook's cells were not pulled and it "
                "cannot be saved back. Add a token at ~/.kaggle/access_token to enable that."
            )
            return notes

        try:
            remote = await self.api.get_notebook(ref)
        except KaggleApiError as exc:
            raise ToolError(f"Could not pull {ref.ref}: {exc}") from exc
        if remote.kernel_type != "notebook":
            raise ToolError(
                f"{ref.ref} is a {remote.kernel_type}, not a notebook; only notebooks are supported."
            )
        self.notebook = remote

        if local is not None and not refresh:
            if local.dirty:
                self.doc = local
                if remote.version != local.base_version:
                    notes.append(
                        f"Kept your unsaved local edits (based on version {local.base_version}); "
                        f"Kaggle now has version {remote.version}. pull_notebook(discard_local_changes=True) "
                        "takes Kaggle's copy instead."
                    )
                else:
                    notes.append("Kept unsaved local edits from the previous session.")
                return notes
            if remote.version == local.base_version:
                self.doc = local
                return notes

        doc = NotebookDocument.from_ipynb(remote.source or "{}", path=path, ref=ref.ref)
        doc.dirty = False
        doc.base_version = remote.version
        doc.save()
        self.doc = doc
        notes.append(
            f"Pulled '{remote.title}' (version {remote.version}, {len(doc.cells)} cells) from Kaggle."
        )
        return notes

    # -- servers and kernels ---------------------------------------------------

    async def _start_api_session(
        self, accelerator: str | None, enable_internet: bool | None, notes: list[str]
    ) -> list[ServerEndpoint]:
        if self.notebook is None:
            return []
        shape = resolve_accelerator(accelerator)
        try:
            started = await self.api.start_session(
                self.notebook, shape, enable_internet, SESSION_START_TIMEOUT
            )
        except SessionAlreadyRunning as exc:
            self.last_session_attempt = {"ok": False, "error": str(exc)}
            notes.append(str(exc))
            return []
        except KaggleApiError as exc:
            logging.warning("API session start failed: %s", exc)
            self.last_session_attempt = {"ok": False, "error": str(exc)}
            notes.append(f"Could not start a session through the Kaggle API: {exc}")
            return []
        self.last_session_attempt = {
            "ok": bool(started.server_urls),
            "kaggle_session_id": started.kaggle_session_id,
            "operation_done": started.operation.get("done"),
            "server_urls": [url_shape(u) for u in started.server_urls],
        }
        if not started.server_urls:
            # A session we cannot use still burns quota: stop it.
            if started.kaggle_session_id is not None:
                try:
                    await self.api.cancel_session(started.kaggle_session_id)
                    self.last_session_attempt["cancelled"] = True
                except KaggleApiError as exc:
                    logging.warning("Could not cancel unusable session: %s", exc)
            logging.warning(
                "Kaggle session operation gave no Jupyter URL; response keys: %s",
                list((started.operation.get("response") or {}).keys())
                if isinstance(started.operation.get("response"), dict)
                else type(started.operation.get("response")).__name__,
            )
            notes.append(
                "Starting a session through the Kaggle API did not return a Jupyter URL."
            )
            return []
        self.state.kaggle_session_id = started.kaggle_session_id
        # Kept until one attaches: Kaggle refuses a second session for the
        # notebook (409), so a retry must reuse these rather than start again.
        self.state.candidate_server_urls = list(started.server_urls)
        self._persist()
        endpoints = []
        for url in started.server_urls:
            try:
                endpoints.append(parse_server_url(url))
            except UnsupportedServerUrl:
                continue
        return endpoints

    async def _attach_first(
        self,
        endpoints: list[ServerEndpoint],
        kernel_id: str | None,
        new_kernel: bool,
        errors: list[str],
    ) -> dict | None:
        for endpoint in endpoints:
            try:
                info = await self._attach(endpoint, kernel_id, new_kernel)
            except KernelUnavailable as exc:
                logging.info("Attach to %s failed: %s", url_shape(endpoint.url), exc)
                errors.append(f"{url_shape(endpoint.url)}: {exc}")
                continue
            self.state.candidate_server_urls = None
            self._persist()
            return info
        return None

    async def _attach(
        self,
        endpoint: ServerEndpoint,
        kernel_id: str | None,
        new_kernel: bool,
    ) -> dict:
        if self.kernel is not None:
            await self.kernel.close()
            self.kernel = None
        kernel = self.kernel_factory(endpoint)
        info = await kernel.connect(kernel_id=kernel_id, new_kernel=new_kernel)
        self.kernel = kernel
        self.state.server_url = endpoint.url
        self.state.kernel_id = kernel.kernel_id
        self._persist()
        try:
            reply = await kernel.execute(_PROBE, timeout=60)
            text = "".join(
                o.get("text", "")
                for o in reply.get("outputs", [])
                if o.get("output_type") == "stream" and o.get("name") == "stdout"
            )
            self.environment = json.loads(text.strip().splitlines()[-1])
        except Exception as exc:
            logging.info("Environment probe failed: %s", exc)
            self.environment = None
        return info

    async def open(
        self,
        notebook_url: str | None = None,
        server_url: str | None = None,
        accelerator: str | None = None,
        enable_internet: bool | None = None,
        start_session: bool = True,
        kernel_id: str | None = None,
        new_kernel: bool = False,
        refresh: bool = False,
        use_browser: bool = True,
    ) -> dict:
        try:
            ref = parse_notebook_ref(notebook_url) if notebook_url else self.ref
        except UnsupportedNotebookUrl as exc:
            raise ToolError(str(exc)) from exc
        endpoint = None
        if server_url:
            try:
                endpoint = parse_server_url(server_url)
            except UnsupportedServerUrl as exc:
                raise ToolError(str(exc)) from exc

        browser = self.browser if use_browser else None
        # Nothing named: adopt the notebook already open in the browser's editor.
        if ref is None and endpoint is None and browser is not None:
            refs = sorted(set(await browser.open_editor_refs()))
            if len(refs) > 1:
                raise ToolError(
                    f"Several notebooks are open in the browser's editor ({', '.join(refs)}); "
                    "pass notebook_url to pick one."
                )
            if refs:
                ref = _maybe_ref(refs[0])

        switching = ref != self.ref
        if switching:
            if self.kernel is not None:
                await self.kernel.close()
                self.kernel = None
            if not endpoint:
                # A saved server belongs to the previous notebook's session.
                self.state.server_url = None
                self.state.kernel_id = None
                self.state.kaggle_session_id = None
                self.state.candidate_server_urls = None
        self.ref = ref
        self.state.notebook_ref = ref.ref if ref else None
        self._persist()

        browser_note = None
        if browser is not None and ref is not None:
            try:
                tab = await browser.open_editor(ref, EDITOR_WAIT)
            except SignInRequired as exc:
                self.state.mode = "editor"
                self._persist()
                return self._result(False, [str(exc)], sign_in_required=True)
            except BrowserUnavailable as exc:
                browser_note = f"{exc} Using the local working copy instead."
            else:
                self.state.mode = "editor"
                self._persist()
                return await self._open_editor(tab, endpoint, kernel_id, new_kernel)
        self.state.mode = "local"
        self._persist()

        notes = await self._load_doc(ref, refresh or False)
        if browser_note:
            notes.append(browser_note)

        # Already attached to the requested server: keep the kernel.
        if (
            self.kernel is not None
            and (endpoint is None or endpoint.url == self.kernel.endpoint.url)
            and not new_kernel
            and (kernel_id is None or kernel_id == self.kernel.kernel_id)
            and await self.kernel.is_alive()
        ):
            return self._result(True, notes + ["Already connected."])

        source = "server_url"
        if endpoint is None and self.state.server_url:
            endpoint = parse_server_url(self.state.server_url)
            source = "saved"
            kernel_id = kernel_id or self.state.kernel_id
        attach_error = None
        if endpoint is not None:
            try:
                info = await self._attach(endpoint, kernel_id, new_kernel)
                return self._result(True, notes, attach=info, server_source=source)
            except KernelUnavailable as exc:
                attach_error = str(exc)
                if source == "server_url":
                    raise ToolError(
                        f"{exc}. Check the URL is the current 'VS Code compatible URL' from "
                        "Run > Kaggle Jupyter Server and that the session is still running."
                    ) from exc
                notes.append(f"The saved Kaggle session is gone ({exc}).")
                self.state.server_url = None
                self.state.kernel_id = None
                self._persist()

        errors: list[str] = []
        if self.state.candidate_server_urls:
            pending = []
            for url in self.state.candidate_server_urls:
                try:
                    pending.append(parse_server_url(url))
                except UnsupportedServerUrl:
                    continue
            info = await self._attach_first(pending, None, new_kernel, errors)
            if info is not None:
                return self._result(True, notes, attach=info, server_source="api")
            notes.append(
                "Could not attach to the session the Kaggle API started earlier."
            )
            self.state.candidate_server_urls = None
            self._persist()

        if start_session and ref is not None and self.api.configured:
            endpoints = await self._start_api_session(
                accelerator, enable_internet, notes
            )
            info = await self._attach_first(endpoints, None, new_kernel, errors)
            if info is not None:
                return self._result(True, notes, attach=info, server_source="api")
            if endpoints:
                notes.append("Started a Kaggle session but could not attach to it.")
        if errors:
            attach_error = "; ".join(errors)

        target = ref.edit_url if ref else f"{KAGGLE}/code"
        self.open_browser(target)
        notes.append(MANUAL_STEPS)
        result = self._result(False, notes, open_url=target)
        if attach_error:
            result["last_error"] = attach_error
        return result

    async def _open_editor(
        self,
        editor: EditorTab,
        endpoint: ServerEndpoint | None,
        kernel_id: str | None,
        new_kernel: bool,
    ) -> dict:
        notes = [EDITOR_NOTE.format(where=editor.describe()["browser"])]
        extra: dict[str, Any] = {}
        if endpoint is not None:
            try:
                extra["attach"] = await self._attach(endpoint, kernel_id, new_kernel)
                notes.append(
                    "Also attached to the Jupyter server; file transfers use it so they "
                    "do not show up as cells."
                )
            except KernelUnavailable as exc:
                notes.append(
                    f"Could not attach to the Jupyter server as well ({exc}); file "
                    "transfers will run as temporary cells in the editor."
                )
        try:
            cells = await LiveNotebook(editor).cells()
        except BridgeError as exc:
            raise ToolError(f"The Kaggle editor tab did not respond: {exc}") from exc
        return self._result(True, notes, cell_count=len(cells), **extra)

    async def ensure_kernel(self) -> RemoteKernel:
        if self.kernel is not None and self.kernel.connected:
            return self.kernel
        if AUTO_RECONNECT and self.state.server_url:
            try:
                await self._attach(
                    parse_server_url(self.state.server_url), self.state.kernel_id, False
                )
                return self.kernel
            except KernelUnavailable as exc:
                raise ToolError(
                    f"The Kaggle session is no longer reachable ({exc}). Sessions stop when idle "
                    "or at their time limit: call open_kaggle_connection to start or attach to a new one."
                ) from exc
        raise ToolError(
            "No Kaggle kernel is connected. Call open_kaggle_connection with notebook_url "
            "(and server_url if you have one) first."
        )

    async def execute(self, code: str, timeout: float | None = None) -> dict:
        kernel = await self.ensure_kernel()
        try:
            return await kernel.execute(code, timeout=timeout)
        except KernelUnavailable as exc:
            await kernel.close()
            self.kernel = None
            raise ToolError(f"{exc}. The next call will try to reattach.") from exc

    def require_doc(self) -> NotebookDocument:
        if self.doc is None:
            path = self._doc_path(self.ref)
            if path is not None and path.is_file():
                self.doc = NotebookDocument.load(path)
            else:
                raise ToolError(
                    "No notebook is open. Call open_kaggle_connection with notebook_url first."
                )
        return self.doc

    # -- Kaggle sync -----------------------------------------------------------

    async def pull(self, discard_local_changes: bool) -> dict:
        if self.editor_mode:
            raise ToolError(
                "The notebook is open in the Kaggle editor, which already holds its "
                "current version; pull_notebook only refreshes the local working copy."
            )
        doc = self.require_doc()
        if self.ref is None:
            raise ToolError("The scratch notebook has no Kaggle copy to pull.")
        if doc.dirty and not discard_local_changes:
            raise ToolError(
                "The working copy has unsaved edits. save_notebook first, or pass "
                "discard_local_changes=True to replace them with Kaggle's version."
            )
        notes = await self._load_doc(self.ref, refresh=True)
        return {"notes": notes, "cell_count": len(self.doc.cells)}

    async def save(
        self, quick_save: bool, include_outputs: bool, accelerator: str | None
    ) -> dict:
        editor = await self.editor_tab()
        doc = None if editor is not None else self.require_doc()
        if self.ref is None:
            raise ToolError("The scratch notebook has no Kaggle notebook to save to.")
        if self.notebook is None:
            if not self.api.configured:
                raise ToolError(
                    "Saving needs Kaggle API credentials (~/.kaggle/access_token)."
                )
            self.notebook = await self.api.get_notebook(self.ref)
        if editor is not None:
            try:
                content = await LiveNotebook(editor).to_ipynb(include_outputs)
            except BridgeError as exc:
                raise ToolError(str(exc)) from exc
        else:
            content = doc.to_ipynb(include_outputs=include_outputs)
        text = json.dumps(content)
        try:
            saved = await self.api.save_notebook(
                self.notebook,
                text,
                quick_save=quick_save,
                machine_shape=resolve_accelerator(accelerator),
            )
        except KaggleApiError as exc:
            raise ToolError(str(exc)) from exc
        version = saved.get("version_number")
        if doc is not None:
            doc.dirty = False
            doc.base_version = version or doc.base_version
            doc.save()
            version = doc.base_version
        self.notebook.version = version or self.notebook.version
        saved["quick_save"] = quick_save
        return saved

    # -- status ----------------------------------------------------------------

    def _result(self, connected: bool, notes: list[str], **extra) -> dict:
        payload = {"connected": connected, **self.describe(), "notes": notes}
        payload.update(extra)
        return payload

    def describe(self) -> dict:
        editor = self.editor()
        # In editor mode the tab holds the notebook; the local copy is not used.
        doc = None if self.editor_mode else self.doc
        return {
            "mode": "editor" if self.editor_mode else "local",
            "editor": editor.describe() if editor is not None else None,
            "notebook": self.ref.ref if self.ref else None,
            "notebook_url": self.ref.url if self.ref else None,
            "kaggle_version": doc.base_version if doc else None,
            "unsaved_changes": doc.dirty if doc else None,
            "cell_count": len(doc.cells) if doc else None,
            "working_copy": str(doc.path) if doc and doc.path else None,
            "server_url": redact_url(self.state.server_url),
            "kernel_id": self.kernel.kernel_id if self.kernel else self.state.kernel_id,
            "environment": self.environment,
        }

    async def status(self) -> dict:
        alive = await self.kernel.is_alive() if self.kernel else False
        payload = {
            "connected": alive or self.editor() is not None,
            "kernel_alive": alive,
            **self.describe(),
            "browser": (
                {
                    "attached": self.browser.running,
                    "which": self.browser.kind,
                    "name": self.browser.name,
                    "own_profile_dir": str(self.browser.profile_dir),
                }
                if self.browser
                else None
            ),
            "credentials": credentials_source(),
            "last_api_session_attempt": self.last_session_attempt,
            "auto_reconnect": AUTO_RECONNECT,
            "state_dir": None if self.ephemeral else str(self.state_dir),
            "log_dir": SERVER_INFO.get("log_dir"),
            "server_pid": os.getpid(),
        }
        if self.ref is not None and self.api.configured:
            try:
                payload["kaggle_session"] = await self.api.session_status(self.ref)
            except KaggleApiError as exc:
                payload["kaggle_session"] = {"error": str(exc)}
        return payload

    async def disconnect(self, shutdown_kernel: bool, stop_session: bool) -> dict:
        result: dict[str, Any] = {"disconnected": self.kernel is not None}
        if self.kernel is not None:
            await self.kernel.close(shutdown_kernel=shutdown_kernel)
            self.kernel = None
        if stop_session:
            if self.state.kaggle_session_id is None:
                result["stop_session"] = (
                    "No API-started session id is known; stop the session from the Kaggle "
                    "editor (Run > Stop session)."
                )
            else:
                try:
                    error = await self.api.cancel_session(self.state.kaggle_session_id)
                    result["stop_session"] = error or "cancel requested"
                except KaggleApiError as exc:
                    result["stop_session"] = str(exc)
        self.state.server_url = None
        self.state.kernel_id = None
        self.state.kaggle_session_id = None
        self.state.candidate_server_urls = None
        self._persist()
        return result

    async def close(self) -> None:
        if self.kernel is not None:
            await self.kernel.close()
            self.kernel = None


class LocalNotebook:
    """The local working copy, run on the attached kernel (the mode without an editor tab).

    Same interface as bridge.LiveNotebook, so the tools do not care which one
    they are working on.
    """

    kind = "local"

    def __init__(self, session: KaggleSession):
        self.session = session

    @property
    def doc(self) -> NotebookDocument:
        return self.session.require_doc()

    @property
    def dirty(self) -> bool:
        return self.doc.dirty

    async def cells(self) -> list[dict]:
        return self.doc.cells

    async def add(self, index: int | None, cell_type: str, source: str) -> str:
        return self.doc.add(index, cell_type, source)

    def _cell(self, cell_id: str) -> dict:
        try:
            return self.doc.get(cell_id)
        except CellNotFound:
            raise ToolError(f"Unknown cell id {cell_id}") from None

    async def update(self, cell_id: str, source: str) -> None:
        self._cell(cell_id)
        self.doc.update(cell_id, source)

    async def move(self, cell_id: str, index: int) -> int:
        self._cell(cell_id)
        self.doc.move(cell_id, index)
        return self.doc.index_of(cell_id)

    async def delete(self, cell_id: str) -> None:
        self._cell(cell_id)
        self.doc.delete(cell_id)

    async def run(self, cell_id: str, timeout: float | None = None) -> dict:
        source = self._cell(cell_id).get("source") or ""
        reply = await self.session.execute(source, timeout=timeout)
        outputs = reply.get("outputs") or []
        try:
            self.doc.set_outputs(cell_id, outputs, reply.get("execution_count"))
        except CellNotFound:
            pass  # deleted while it ran
        return {"outputs": outputs, "execution_count": reply.get("execution_count")}

    async def interrupt(self) -> None:
        kernel = await self.session.ensure_kernel()
        await kernel.interrupt()

    async def restart(self) -> None:
        kernel = await self.session.ensure_kernel()
        await kernel.restart()

    async def to_ipynb(self, include_outputs: bool = True) -> dict:
        return self.doc.to_ipynb(include_outputs)


def _maybe_ref(text: str) -> NotebookRef | None:
    try:
        return parse_notebook_ref(text)
    except UnsupportedNotebookUrl:
        return None
