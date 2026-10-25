"""Editor mode: the cell tools working on the Kaggle editor in the controlled Chrome.

FakeTab stands in for an EditorTab (a Kaggle editor page with editor_bridge.js in
it) and FakeBrowser for KaggleBrowser. The last test drives a real headless
Chrome to check the launcher, the DevTools client and the script injection.
"""

import asyncio
import contextlib
import copy
import io
import json
from pathlib import Path
import uuid

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from conftest import NOTEBOOK, FakeApi
from kaggle_mcp import bridge as bridge_mod
from kaggle_mcp import build_server
from kaggle_mcp.bridge import BridgeError
from kaggle_mcp.browser import (
    BrowserUnavailable,
    CdpPage,
    EditorTab,
    KaggleBrowser,
    SignInRequired,
    find_browser,
)
from kaggle_mcp.session import KaggleSession

NB = "https://www.kaggle.com/code/owner/demo-notebook/edit"
BUSY = "A cell is running in the Kaggle editor."


class FakeTab:
    """A Kaggle editor page: a notebook in memory, code run with exec in one namespace.

    Mirrors editor_bridge.js: same methods, same id handling and busy rules.
    """

    def __init__(self, ref: str = "owner/demo-notebook"):
        self.ref = ref
        self.open = True
        self.nb = copy.deepcopy(NOTEBOOK)
        self.namespace: dict = {}
        self.count = 0
        self.runs: dict[str, dict] = {}
        self.running = False

    def describe(self):
        return {"notebook": self.ref, "browser": "the test browser"}

    async def request(self, method, params=None, timeout=None):
        if not self.open:
            raise BridgeError("The Kaggle editor tab was closed.")
        try:
            return await getattr(self, "m_" + method)(**(params or {}))
        except BridgeError:
            raise
        except Exception as exc:
            raise BridgeError(str(exc)) from exc

    def _ids(self):
        if self.nb["nbformat_minor"] < 5 or any(
            "id" not in c for c in self.nb["cells"]
        ):
            if not self.running:
                self.nb["nbformat_minor"] = 5
                for cell in self.nb["cells"]:
                    cell.setdefault("id", uuid.uuid4().hex)
        return self.nb

    def _index(self, cell_id):
        for i, c in enumerate(self.nb["cells"]):
            if c.get("id") == cell_id:
                return i
        raise KeyError(f"Unknown cell id {cell_id}")

    def _idle(self):
        if self.running:
            raise RuntimeError(BUSY + " wait for it to finish or interrupt it first.")

    async def m_ready(self):
        return {"ready": True, "ref": self.ref}

    async def m_getNotebook(self):
        return copy.deepcopy(self._ids())

    async def m_insertCell(self, index=None, cell_type="code", source=""):
        self._ids()
        cells = self.nb["cells"]
        at = len(cells) if index is None else max(0, min(len(cells), index))
        cell = {
            "cell_type": cell_type,
            "id": uuid.uuid4().hex,
            "metadata": {},
            "source": source,
        }
        if cell_type == "code":
            cell.update(execution_count=None, outputs=[])
        cells.insert(at, cell)
        return {"id": cell["id"], "index": at}

    async def m_updateCell(self, id, source):
        self._idle()
        self.nb["cells"][self._index(id)]["source"] = source
        return {"id": id}

    async def m_moveCell(self, id, index):
        self._idle()
        cell = self.nb["cells"].pop(self._index(id))
        at = max(0, min(len(self.nb["cells"]), index))
        self.nb["cells"].insert(at, cell)
        return {"id": id, "index": at}

    async def m_deleteCell(self, id):
        self._idle()
        del self.nb["cells"][self._index(id)]
        return {"id": id}

    async def m_runCell(self, id):
        if self.running:
            raise RuntimeError("still running")
        cell = self.nb["cells"][self._index(id)]
        token = uuid.uuid4().hex
        self.runs[token] = {"done": False}
        self.running = True
        asyncio.create_task(self._execute(token, cell))
        return {"token": token, "index": self._index(id)}

    async def _execute(self, token, cell):
        source = cell["source"]
        source = "".join(source) if isinstance(source, list) else source
        cell["execution_count"] = None
        cell["outputs"] = []
        await asyncio.sleep(0.05)
        buf = io.StringIO()
        outputs = []
        try:
            lines = source.strip().splitlines()
            body, last = lines[:-1], lines[-1] if lines else ""
            if "time.sleep" in source:
                await asyncio.sleep(1.0)
            with contextlib.redirect_stdout(buf):
                exec("\n".join(body), self.namespace)
                try:
                    value = eval(last, self.namespace)
                except SyntaxError:
                    exec(last, self.namespace)
                    value = None
            if buf.getvalue():
                outputs.append(
                    {"output_type": "stream", "name": "stdout", "text": buf.getvalue()}
                )
            if value is not None:
                outputs.append(
                    {
                        "output_type": "execute_result",
                        "data": {"text/plain": repr(value)},
                        "metadata": {},
                    }
                )
        except Exception as exc:
            outputs.append(
                {
                    "output_type": "error",
                    "ename": type(exc).__name__,
                    "evalue": str(exc),
                    "traceback": [],
                }
            )
        self.count += 1
        cell["outputs"] = outputs
        cell["execution_count"] = self.count
        self.running = False
        self.runs[token]["done"] = True

    async def m_runState(self, token, id):
        try:
            cell = self.nb["cells"][self._index(id)]
        except KeyError:
            cell = None
        run = self.runs.get(token, {"done": False})
        return {
            "done": run["done"],
            "error": None,
            "deleted": cell is None,
            "execution_count": cell and cell.get("execution_count"),
            "outputs": (cell or {}).get("outputs", []),
            "kernel": "busy" if self.running else "idle",
        }

    async def m_interrupt(self):
        return None


class StoppedSessionTab(FakeTab):
    """Kaggle with the session off: a run starts it, and the cell runs some time later."""

    def __init__(self, ref="owner/demo-notebook", start_delay=0.8, never=False):
        super().__init__(ref)
        self.start_delay = start_delay
        self.never = never

    async def m_runCell(self, id):
        reply = await super().m_runCell(id)
        self.runs[reply["token"]]["queued"] = True
        return reply

    async def _execute(self, token, cell):
        if self.never:
            return  # the session never comes up
        await asyncio.sleep(self.start_delay)
        await super()._execute(token, cell)

    async def m_runState(self, token, id):
        state = await super().m_runState(token, id)
        state["queued"] = not state["done"] and self.runs.get(token, {}).get(
            "queued", False
        )
        return state


class FakeBrowser:
    def __init__(self):
        self.tabs: dict[str, FakeTab] = {}
        self.pages: dict[str, FakeTab] = {}  # what open_editor finds or creates
        self.opened: list[str] = []
        self.sign_in = False
        self.unavailable = False
        self.running = True
        self.kind = "own"
        self.name = "Test Browser"
        self.profile_dir = Path("/tmp/kaggle-mcp-test-profile")

    def editor_for(self, ref):
        tabs = [t for r, t in self.tabs.items() if t.open and (ref is None or r == ref)]
        return tabs[-1] if tabs else None

    async def open_editor_refs(self):
        return [r for r, t in self.tabs.items() if t.open]

    async def open_editor(self, ref, timeout):
        if self.unavailable:
            raise BrowserUnavailable("Could not find Chrome.")
        if self.sign_in:
            raise SignInRequired("Sign in to Kaggle in the kaggle-mcp Chrome window.")
        self.opened.append(ref.ref)
        tab = self.pages.setdefault(ref.ref, FakeTab(ref.ref))
        tab.open = True
        self.tabs[ref.ref] = tab
        return tab

    async def close(self):
        pass


def make(state_dir, browser, api=None, opened=None):
    session = KaggleSession(
        state_dir=state_dir,
        api=api or FakeApi(),
        open_browser=(opened.append if opened is not None else lambda url: None),
        browser=browser,
    )
    return session, build_server(session)


async def call(client, name, **args):
    result = await client.call_tool(name, args)
    if result.structured_content is not None:
        sc = result.structured_content
        return sc.get("result", sc) if set(sc) == {"result"} else sc
    return json.loads(result.content[0].text)


async def test_cells_live_in_the_editor(state_dir):
    browser = FakeBrowser()
    api = FakeApi()
    opened: list[str] = []
    session, mcp = make(state_dir, browser, api, opened)
    async with Client(mcp) as client:
        result = await call(client, "open_kaggle_connection", notebook_url=NB)
        assert result["connected"] is True
        assert result["mode"] == "editor"
        assert result["editor"]["notebook"] == "owner/demo-notebook"
        assert result["cell_count"] == 3
        assert browser.opened == ["owner/demo-notebook"]
        assert opened == []  # no manual-steps tab in the user's browser
        editor = browser.tabs["owner/demo-notebook"]

        outline = await call(client, "get_notebook_outline")
        assert outline["mode"] == "editor"
        assert [c["type"] for c in outline["cells"]] == ["markdown", "code", "code"]
        assert all(c["id"] for c in outline["cells"])  # the 4.4 notebook got ids

        run = await call(client, "run_cells")
        assert run["state"] == "done"
        assert run["results"][0]["stdout"] == "x set\n"
        assert run["results"][1]["result"] == "42"

        new = (await call(client, "add_code_cell", code="y = x + 1\ny", cellIndex=1))[
            "newCellId"
        ]
        assert editor.nb["cells"][1]["id"] == new
        ran = await call(client, "run_code_cell", cellId=new)
        assert ran["result"] == "22" and ran["cell_id"] == new
        assert editor.nb["cells"][1]["outputs"][0]["data"]["text/plain"] == "22"

        await call(client, "update_cell", cellId=new, content="x - 1")
        assert (await call(client, "run_code_cell", cellId=new))["result"] == "20"
        moved = await call(client, "move_cell", cellId=new, cellIndex=0)
        assert moved["cellIndex"] == 0 and editor.nb["cells"][0]["id"] == new
        await call(client, "delete_cell", cellId=new)
        assert new not in [c["id"] for c in editor.nb["cells"]]

        # run_code shows up as a cell while it runs, then is removed
        before = len(editor.nb["cells"])
        adhoc = await call(client, "run_code", code="print('adhoc')\n7 * 6")
        assert adhoc["stdout"] == "adhoc\n" and adhoc["result"] == "42"
        assert adhoc["cell_id"] is None and len(editor.nb["cells"]) == before
        kept = await call(client, "run_code", code="'kept'", keep_cell=True)
        assert editor.nb["cells"][-1]["id"] == kept["cell_id"]

        with pytest.raises(ToolError, match="Unknown cell id"):
            await call(client, "run_code_cell", cellId="nope")
        with pytest.raises(ToolError, match="already holds"):
            await call(client, "pull_notebook")

        saved = await call(client, "save_notebook")
        pushed = json.loads(api.saves[-1]["text"])
        assert "".join(pushed["cells"][-1]["source"]) == "'kept'"
        assert pushed["cells"][-1]["outputs"] == []  # outputs stripped by default
        assert saved["version_number"] == 4

        status = await call(client, "kaggle_connection_status")
        assert status["connected"] is True and status["mode"] == "editor"


async def test_busy_editor_refuses_rewrites(state_dir):
    session, mcp = make(state_dir, FakeBrowser())
    async with Client(mcp) as client:
        await call(client, "open_kaggle_connection", notebook_url=NB)
        slow = (
            await call(
                client, "add_code_cell", code="import time\ntime.sleep(1)\n'slow'"
            )
        )["newCellId"]
        job = await call(client, "run_cells", cell_ids=[slow], background=True)
        await asyncio.sleep(0.3)
        with pytest.raises(ToolError, match="A cell is running"):
            await call(client, "update_cell", cellId=slow, content="1")
        for _ in range(50):
            status = await call(client, "job_status", job_id=job["job_id"])
            if status["state"] != "running":
                break
            await asyncio.sleep(0.1)
        assert status["state"] == "done"
        assert status["results"][0]["result"] == "'slow'"


async def test_sign_in_then_connect(state_dir):
    browser = FakeBrowser()
    browser.sign_in = True
    session, mcp = make(state_dir, browser)
    async with Client(mcp) as client:
        first = await call(client, "open_kaggle_connection", notebook_url=NB)
        assert first["connected"] is False and first["sign_in_required"] is True
        assert "Sign in" in first["notes"][0]
        browser.sign_in = False  # the user signed in
        second = await call(client, "open_kaggle_connection", notebook_url=NB)
        assert second["connected"] is True and second["mode"] == "editor"


async def test_closed_tab_is_reopened(state_dir):
    browser = FakeBrowser()
    session, mcp = make(state_dir, browser)
    async with Client(mcp) as client:
        await call(client, "open_kaggle_connection", notebook_url=NB)
        browser.tabs["owner/demo-notebook"].open = False  # user closed the tab
        cells = await call(client, "get_cells")
        assert len(cells["cells"]) == 3
        assert browser.opened == ["owner/demo-notebook", "owner/demo-notebook"]


async def test_adopts_notebook_open_in_the_window(state_dir):
    browser = FakeBrowser()
    await browser.open_editor(
        type("Ref", (), {"ref": "owner/demo-notebook"})(), timeout=1
    )
    session, mcp = make(state_dir, browser)
    async with Client(mcp) as client:
        result = await call(client, "open_kaggle_connection")
    assert result["mode"] == "editor" and result["notebook"] == "owner/demo-notebook"


async def test_without_chrome_or_opted_out_uses_local_copy(state_dir):
    browser = FakeBrowser()
    browser.unavailable = True
    opened: list[str] = []
    session, mcp = make(state_dir, browser, FakeApi(), opened)
    async with Client(mcp) as client:
        result = await call(
            client, "open_kaggle_connection", notebook_url=NB, start_session=False
        )
        assert result["mode"] == "local" and result["connected"] is False
        assert any("Could not find Chrome" in n for n in result["notes"])
        assert opened == ["https://www.kaggle.com/code/owner/demo-notebook/edit"]
        assert len((await call(client, "get_cells"))["cells"]) == 3

        browser.unavailable = False
        local = await call(
            client,
            "open_kaggle_connection",
            notebook_url=NB,
            start_session=False,
            use_browser=False,
        )
        assert local["mode"] == "local" and browser.opened == []


@pytest.mark.skipif(find_browser() is None, reason="no Chromium browser installed")
async def test_real_chrome_runs_the_bridge(tmp_path):
    # prefer="own": never attach to the developer's everyday Chrome from a test.
    browser = KaggleBrowser(tmp_path / "profile", headless=True, prefer="own")
    try:
        await browser.ensure_running()
        assert browser.running and browser.kind == "own"
        created = await browser.conn.send("Target.createTarget", {"url": "about:blank"})
        page = await CdpPage.attach(browser.conn, created["targetId"])
        tab = EditorTab(page, "owner/demo", browser.where)
        # First call installs editor_bridge.js; off an editor page it reports so.
        state = await tab.request("ready")
        assert state["ready"] is False and state["ref"] is None
        assert state["reason"] == "not an editor page" and state["notFound"] is False
        with pytest.raises(BridgeError, match="Unknown method nope"):
            await tab.request("nope")
        assert await browser.open_editor_refs() == []
        # Another kaggle-mcp process on the same profile reattaches, not relaunches.
        again = KaggleBrowser(tmp_path / "profile", headless=True, prefer="own")
        assert await again.attach_existing() and again.kind == "own"
        await again.close()
        await page.detach()
        assert page.closed
    finally:
        await browser.shutdown()


async def test_run_waits_for_a_stopped_session_to_start(state_dir, monkeypatch):
    browser = FakeBrowser()
    browser.pages["owner/demo-notebook"] = StoppedSessionTab()
    session, mcp = make(state_dir, browser)
    async with Client(mcp) as client:
        await call(client, "open_kaggle_connection", notebook_url=NB)
        new = (await call(client, "add_code_cell", code="6 * 7"))["newCellId"]
        ran = await call(client, "run_code_cell", cellId=new)
        assert ran["result"] == "42" and ran["execution_count"] == 1

        monkeypatch.setattr(bridge_mod, "SESSION_START_WAIT", 0.3)
        browser.tabs["owner/demo-notebook"].never = True
        stuck = (await call(client, "add_code_cell", code="1"))["newCellId"]
        with pytest.raises(ToolError, match="waiting for the Kaggle session to start"):
            await call(client, "run_code_cell", cellId=stuck)
