"""Drive the MCP server through a real MCP client against a real Jupyter server."""

import asyncio
import json
import os

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from conftest import FakeApi
from kaggle_mcp import build_server, tools
from kaggle_mcp.session import KaggleSession

NB = "https://www.kaggle.com/code/owner/demo-notebook/edit"


def make(state_dir, api, opened=None):
    session = KaggleSession(
        state_dir=state_dir,
        api=api,
        open_browser=(opened.append if opened is not None else lambda url: None),
    )
    return session, build_server(session)


async def call(client, name, **args):
    result = await client.call_tool(name, args)
    if result.structured_content is not None:
        sc = result.structured_content
        return sc.get("result", sc) if set(sc) == {"result"} else sc
    return json.loads(result.content[0].text)


async def test_open_run_edit_save(jupyter_server, state_dir):
    api = FakeApi()
    session, mcp = make(state_dir, api)
    async with Client(mcp) as client:
        opened = await call(
            client,
            "open_kaggle_connection",
            notebook_url=NB,
            server_url=jupyter_server["url"],
        )
        assert opened["connected"] is True
        assert opened["notebook"] == "owner/demo-notebook"
        assert opened["cell_count"] == 3
        assert opened["environment"]["python"]
        assert opened["server_url"].endswith("?token=***")
        assert jupyter_server["token"] not in json.dumps(opened)

        outline = await call(client, "get_notebook_outline")
        assert [c["type"] for c in outline["cells"]] == ["markdown", "code", "code"]
        assert outline["unsaved_changes"] is False

        run = await call(client, "run_cells")
        assert run["state"] == "done"
        assert run["results"][0]["stdout"] == "x set\n"
        assert run["results"][1]["result"] == "42"

        cells = await call(client, "get_cells", includeOutputs=True)
        assert cells["cells"][2]["outputs"][0]["data"]["text/plain"] == "42"

        new = await call(client, "add_code_cell", code="1/0")
        failing = await call(client, "run_code_cell", cellId=new["newCellId"])
        assert (
            failing["ok"] is False and failing["error"]["ename"] == "ZeroDivisionError"
        )
        await call(client, "update_cell", cellId=new["newCellId"], content="y = x + 1")
        await call(client, "move_cell", cellId=new["newCellId"], cellIndex=0)

        adhoc = await call(client, "run_code", code="print(x + 100)")
        assert adhoc["stdout"] == "121\n"
        assert (await call(client, "get_notebook_outline"))["cell_count"] == 4

        saved = await call(client, "save_notebook")
        assert saved["version_number"] == 4 and saved["quick_save"] is True
        pushed = json.loads(api.saves[0]["text"])
        assert pushed["cells"][0]["source"] == "y = x + 1"
        assert all(c.get("outputs", []) == [] for c in pushed["cells"])
        assert api.saves[0]["metadata"]["dataset_data_sources"] == ["owner/some-data"]
        status = await call(client, "kaggle_connection_status")
        assert status["unsaved_changes"] is False
        assert status["kaggle_version"] == 4
        assert status["kaggle_session"]["status"] == "RUNNING"
    await session.disconnect(shutdown_kernel=True, stop_session=False)


async def test_restart_reattaches_same_kernel(jupyter_server, state_dir):
    api = FakeApi()
    session, mcp = make(state_dir, api)
    async with Client(mcp) as client:
        await call(
            client,
            "open_kaggle_connection",
            notebook_url=NB,
            server_url=jupyter_server["url"],
        )
        await call(client, "run_code", code="survivor = 'still here'")
        kernel_id = session.kernel.kernel_id
        await call(client, "add_text_cell", content="unsaved note")
    await session.close()

    # A fresh server process: same state dir, no explicit open.
    session2, mcp2 = make(state_dir, api)
    async with Client(mcp2) as client:
        result = await call(client, "run_code", code="print(survivor)")
        assert result["stdout"] == "still here\n"
        assert session2.kernel.kernel_id == kernel_id
        outline = await call(client, "get_notebook_outline")
        assert outline["unsaved_changes"] is True
        assert outline["cells"][-1]["first_line"] == "unsaved note"

        # Re-opening keeps the unsaved edits rather than re-pulling.
        reopened = await call(client, "open_kaggle_connection", notebook_url=NB)
        assert reopened["unsaved_changes"] is True
        with pytest.raises(ToolError, match="unsaved edits"):
            await client.call_tool("pull_notebook", {})
        pulled = await call(client, "pull_notebook", discard_local_changes=True)
        assert pulled["cell_count"] == 3
    await session2.disconnect(shutdown_kernel=True, stop_session=False)


async def test_cancel_interrupts_running_cell(jupyter_server, state_dir):
    session, mcp = make(state_dir, FakeApi())
    async with Client(mcp) as client:
        await call(
            client,
            "open_kaggle_connection",
            notebook_url=NB,
            server_url=jupyter_server["url"],
            new_kernel=True,
        )
        job = await call(
            client, "run_code", code="import time\ntime.sleep(60)", background=True
        )
        await asyncio.sleep(1.5)
        cancelled = await call(client, "cancel_job", job_id=job["job_id"])
        assert cancelled["interrupted"] is True
        for _ in range(50):
            status = await call(client, "job_status", job_id=job["job_id"])
            if status["state"] != "running":
                break
            await asyncio.sleep(0.2)
        assert status["state"] == "cancelled"
        assert status["results"][0]["error"]["ename"] == "KeyboardInterrupt"
        # The kernel is still usable afterwards.
        after = await call(client, "run_code", code="print('alive')")
        assert after["stdout"] == "alive\n"
    await session.disconnect(shutdown_kernel=True, stop_session=False)


async def test_upload_download_roundtrip(
    jupyter_server, state_dir, tmp_path, monkeypatch
):
    monkeypatch.setattr(tools, "UPLOAD_CHUNK_BYTES", 100_000)
    monkeypatch.setattr(tools, "DOWNLOAD_CHUNK_BYTES", 70_000)
    session, mcp = make(state_dir, FakeApi())
    src = tmp_path / "blob.bin"
    src.write_bytes(os.urandom(250_000))
    remote = str(tmp_path / "remote" / "blob.bin")
    async with Client(mcp) as client:
        await call(
            client,
            "open_kaggle_connection",
            notebook_url=NB,
            server_url=jupyter_server["url"],
        )
        up = await call(client, "upload_file", local_path=str(src), remote_path=remote)
        assert up["chunks"] == 3
        down = await call(
            client,
            "download_file",
            remote_path=remote,
            local_path=str(tmp_path / "back.bin"),
        )
        assert down["sha256"] == up["sha256"] and down["chunks"] == 4
        assert (tmp_path / "back.bin").read_bytes() == src.read_bytes()
        out = await call(
            client, "export_notebook", local_path=str(tmp_path / "export.ipynb")
        )
        assert json.loads((tmp_path / "export.ipynb").read_text())["nbformat"] == 4
        assert out["cell_count"] == 3
    await session.disconnect(shutdown_kernel=True, stop_session=False)


async def test_api_session_without_url_falls_back_to_browser(state_dir):
    api = FakeApi(server_url=None)
    opened: list[str] = []
    session, mcp = make(state_dir, api, opened)
    async with Client(mcp) as client:
        result = await call(
            client, "open_kaggle_connection", notebook_url=NB, accelerator="t4"
        )
        assert result["connected"] is False
        assert opened == ["https://www.kaggle.com/code/owner/demo-notebook/edit"]
        assert api.cancelled == [777]  # the unusable session is stopped
        assert "Kaggle Jupyter Server" in result["notes"][-1]
        # Cells are still available for editing before a kernel exists.
        assert (await call(client, "get_notebook_outline"))["cell_count"] == 3
        with pytest.raises(ToolError, match="No Kaggle kernel is connected"):
            await client.call_tool("run_code", {"code": "1"})


async def test_api_session_with_url_attaches(jupyter_server, state_dir):
    api = FakeApi(server_url=jupyter_server["url"])
    session, mcp = make(state_dir, api)
    async with Client(mcp) as client:
        result = await call(
            client, "open_kaggle_connection", notebook_url=NB, accelerator="t4"
        )
        assert result["connected"] is True and result["server_source"] == "api"
        assert session.state.kaggle_session_id == 777
        stopped = await call(
            client, "disconnect_kaggle", shutdown_kernel=True, stop_session=True
        )
        assert stopped["stop_session"] == "cancel requested"
        assert api.cancelled == [777]


async def test_unattachable_api_urls_are_retried_not_restarted(
    jupyter_server, state_dir
):
    api = FakeApi(server_url="https://127.0.0.1:9/nothing-here?token=x")
    opened: list[str] = []
    session, mcp = make(state_dir, api, opened)
    async with Client(mcp) as client:
        first = await call(client, "open_kaggle_connection", notebook_url=NB)
        assert first["connected"] is False and api.sessions_started == 1
        assert session.state.candidate_server_urls  # remembered for the retry
        # The retry tries the remembered URLs, then (as they fail) starts again.
        api.session_server_url = jupyter_server["url"]
        second = await call(client, "open_kaggle_connection", notebook_url=NB)
        assert second["connected"] is True and api.sessions_started == 2
        assert session.state.candidate_server_urls is None
    await session.disconnect(shutdown_kernel=True, stop_session=False)
