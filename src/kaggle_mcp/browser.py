"""The browser kaggle-mcp works in, with the Kaggle notebook editor open in it.

Colab's page ships its own bridge for colab-mcp to connect to; Kaggle's does not.
So kaggle-mcp drives a Chromium browser (Edge, Chrome, Brave, Chromium) over the
DevTools protocol: it opens the notebook's editor in a tab, evaluates
editor_bridge.js in the page, and calls it. Cell edits and runs then happen in
the editor the user is watching.

Which browser:

* the user's own browser, when they have turned on remote debugging at
  edge://inspect/#remote-debugging (chrome://inspect/... in Chrome; version
  144+). Their Kaggle sign-in is used as is; the browser asks them to allow the
  connection when kaggle-mcp attaches.
* otherwise a window of their default Chromium browser that kaggle-mcp starts
  with its own profile in ``<state dir>/chrome-profile`` (browsers refuse remote
  debugging on the default profile). The user signs in to Kaggle there once. It
  keeps running when kaggle-mcp exits, and the next start reattaches to it.

KAGGLE_MCP_BROWSER=user or =own forces one of the two; KAGGLE_MCP_CHROME picks
the executable for the own window.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import json
import logging
import os
import plistlib
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastmcp.exceptions import ToolError
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed, InvalidHandshake

from kaggle_mcp.bridge import REQUEST_TIMEOUT, BridgeError
from kaggle_mcp.refs import NotebookRef

BRIDGE_JS = (Path(__file__).parent / "editor_bridge.js").read_text()
_MISSING = "__kaggle_mcp_bridge_missing__"
_EDIT_PATH = re.compile(r"^https://(?:www\.)?kaggle\.com/code/([^/?#]+)/([^/?#]+)/edit")
LAUNCH_TIMEOUT = 30.0
# Attaching to the user's browser waits for them to click Allow.
APPROVAL_TIMEOUT = float(os.environ.get("KAGGLE_MCP_APPROVAL_TIMEOUT", "120"))


@dataclass(frozen=True)
class Browser:
    name: str
    scheme: str  # edge://, chrome://, brave:// for the inspect page
    user_data_dir: Path
    executables: tuple[str, ...]
    # How the OS names it as the default browser: macOS bundle id, Windows
    # ProgId prefix, or Linux .desktop file.
    default_ids: tuple[str, ...]

    @property
    def enable_hint(self) -> str:
        return (
            f"To use your everyday {self.name} instead (your Kaggle sign-in, no extra "
            f"window), open {self.scheme}://inspect/#remote-debugging in it and turn "
            "remote debugging on."
        )


_HOME = Path.home()
_LOCAL = Path(os.environ.get("LOCALAPPDATA", _HOME / "AppData" / "Local"))
_MAC = _HOME / "Library/Application Support"
_PROGRAMS = Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
_PROGRAMS_X86 = Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))

_BROWSERS = {
    "darwin": [
        Browser(
            "Microsoft Edge",
            "edge",
            _MAC / "Microsoft Edge",
            ("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",),
            ("com.microsoft.edgemac",),
        ),
        Browser(
            "Google Chrome",
            "chrome",
            _MAC / "Google/Chrome",
            ("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",),
            ("com.google.chrome",),
        ),
        Browser(
            "Brave",
            "brave",
            _MAC / "BraveSoftware/Brave-Browser",
            ("/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",),
            ("com.brave.browser",),
        ),
        Browser(
            "Chromium",
            "chrome",
            _MAC / "Chromium",
            ("/Applications/Chromium.app/Contents/MacOS/Chromium",),
            ("org.chromium.chromium",),
        ),
    ],
    "win32": [
        Browser(
            "Microsoft Edge",
            "edge",
            _LOCAL / "Microsoft/Edge/User Data",
            (
                str(_PROGRAMS_X86 / "Microsoft/Edge/Application/msedge.exe"),
                str(_PROGRAMS / "Microsoft/Edge/Application/msedge.exe"),
            ),
            ("MSEdgeHTM",),
        ),
        Browser(
            "Google Chrome",
            "chrome",
            _LOCAL / "Google/Chrome/User Data",
            (
                str(_PROGRAMS / "Google/Chrome/Application/chrome.exe"),
                str(_PROGRAMS_X86 / "Google/Chrome/Application/chrome.exe"),
                str(_LOCAL / "Google/Chrome/Application/chrome.exe"),
            ),
            ("ChromeHTML",),
        ),
        Browser(
            "Brave",
            "brave",
            _LOCAL / "BraveSoftware/Brave-Browser/User Data",
            (str(_PROGRAMS / "BraveSoftware/Brave-Browser/Application/brave.exe"),),
            ("BraveHTML",),
        ),
    ],
    "linux": [
        Browser(
            "Microsoft Edge",
            "edge",
            _HOME / ".config/microsoft-edge",
            ("microsoft-edge", "microsoft-edge-stable"),
            ("microsoft-edge.desktop",),
        ),
        Browser(
            "Google Chrome",
            "chrome",
            _HOME / ".config/google-chrome",
            ("google-chrome", "google-chrome-stable"),
            ("google-chrome.desktop",),
        ),
        Browser(
            "Brave",
            "brave",
            _HOME / ".config/BraveSoftware/Brave-Browser",
            ("brave-browser",),
            ("brave-browser.desktop",),
        ),
        Browser(
            "Chromium",
            "chrome",
            _HOME / ".config/chromium",
            ("chromium", "chromium-browser"),
            ("chromium.desktop", "chromium-browser.desktop"),
        ),
    ],
}


class BrowserUnavailable(ToolError):
    """No browser to use, or it would not start or let kaggle-mcp connect."""


class SignInRequired(ToolError):
    """The browser kaggle-mcp uses is not signed in to Kaggle."""


def _default_browser_id() -> str | None:
    """The OS's default web browser, as a bundle id / ProgId / .desktop name."""
    try:
        if sys.platform == "darwin":
            plist = _HOME / (
                "Library/Preferences/com.apple.LaunchServices/"
                "com.apple.launchservices.secure.plist"
            )
            for handler in plistlib.loads(plist.read_bytes()).get("LSHandlers", []):
                if handler.get("LSHandlerURLScheme") == "https":
                    return handler.get("LSHandlerRoleAll")
        elif sys.platform == "win32":
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations"
                r"\https\UserChoice",
            )
            return winreg.QueryValueEx(key, "ProgId")[0]
        else:
            out = subprocess.run(
                ["xdg-settings", "get", "default-web-browser"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return out.stdout.strip() or None
    except Exception as exc:  # no default set, or an unreadable setting
        logging.debug("Could not read the default browser: %s", exc)
    return None


def known_browsers() -> list[Browser]:
    """Chromium browsers for this OS, the user's default browser first."""
    browsers = list(_BROWSERS.get(sys.platform, []))
    default = (_default_browser_id() or "").lower()
    if default:
        browsers.sort(
            key=lambda b: not any(default.startswith(i.lower()) for i in b.default_ids)
        )
    return browsers


def _executable(browser: Browser) -> str | None:
    for candidate in browser.executables:
        if Path(candidate).exists():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    return None


def find_browser() -> tuple[str, Browser | None] | None:
    """The executable for kaggle-mcp's own window: the override, else the default browser."""
    override = os.environ.get("KAGGLE_MCP_CHROME")
    if override:
        path = override if Path(override).exists() else shutil.which(override)
        return (path, None) if path else None
    for browser in known_browsers():
        path = _executable(browser)
        if path:
            return path, browser
    return None


def read_active_port(user_data_dir: Path) -> tuple[int, str] | None:
    """A browser writes its DevTools port and endpoint path here while it listens."""
    try:
        lines = (user_data_dir / "DevToolsActivePort").read_text().split()
        return int(lines[0]), lines[1]
    except (OSError, ValueError, IndexError):
        return None


def editor_ref(url: str) -> str | None:
    m = _EDIT_PATH.match(url or "")
    return f"{m.group(1)}/{m.group(2)}" if m else None


class CdpConnection:
    """The browser-level DevTools websocket; pages are reached through sessions on it."""

    def __init__(self, ws: ClientConnection):
        self.ws = ws
        self._ids = itertools.count(1)
        self._pending: dict[int, asyncio.Future] = {}
        self.detached: set[str] = set()
        self._reader = asyncio.create_task(self._read())

    @classmethod
    async def open(cls, port: int, path: str, timeout: float) -> CdpConnection:
        # No Origin header: Chrome rejects DevTools clients that send one.
        ws = await connect(
            f"ws://127.0.0.1:{port}{path}", max_size=None, open_timeout=timeout
        )
        return cls(ws)

    @property
    def closed(self) -> bool:
        return self._reader.done()

    async def _read(self) -> None:
        try:
            async for raw in self.ws:
                msg = json.loads(raw)
                if msg.get("method") == "Target.detachedFromTarget":
                    self.detached.add(msg.get("params", {}).get("sessionId"))
                    continue
                future = self._pending.pop(msg.get("id"), None)
                if future is not None and not future.done():
                    future.set_result(msg)
        except (ConnectionClosed, ValueError):
            pass
        finally:
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(BridgeError("Chrome closed the connection."))
            self._pending.clear()

    async def send(
        self,
        method: str,
        params: dict | None = None,
        session_id: str | None = None,
        timeout: float | None = None,
    ) -> dict:
        if self.closed:
            raise BridgeError("The connection to Chrome is closed.")
        if session_id is not None and session_id in self.detached:
            raise BridgeError("The Kaggle editor tab was closed.")
        msg_id = next(self._ids)
        message: dict[str, Any] = {
            "id": msg_id,
            "method": method,
            "params": params or {},
        }
        if session_id is not None:
            message["sessionId"] = session_id
        future = asyncio.get_running_loop().create_future()
        self._pending[msg_id] = future
        try:
            await self.ws.send(json.dumps(message))
            reply = await asyncio.wait_for(future, timeout or REQUEST_TIMEOUT)
        except TimeoutError:
            raise BridgeError(f"Chrome did not answer {method} in time.") from None
        except ConnectionClosed:
            raise BridgeError("Chrome closed the connection.") from None
        finally:
            self._pending.pop(msg_id, None)
        if "error" in reply:
            raise BridgeError(
                f"Chrome refused {method}: {reply['error'].get('message')}"
            )
        return reply.get("result") or {}

    async def close(self) -> None:
        await self.ws.close()
        await asyncio.gather(self._reader, return_exceptions=True)


class CdpPage:
    """One tab, through a flattened DevTools session on the browser connection."""

    def __init__(self, conn: CdpConnection, target_id: str, session_id: str):
        self.conn = conn
        self.target_id = target_id
        self.session_id = session_id

    @classmethod
    async def attach(cls, conn: CdpConnection, target_id: str) -> CdpPage:
        result = await conn.send(
            "Target.attachToTarget", {"targetId": target_id, "flatten": True}
        )
        return cls(conn, target_id, result["sessionId"])

    @property
    def closed(self) -> bool:
        return self.conn.closed or self.session_id in self.conn.detached

    async def send(
        self, method: str, params: dict | None = None, timeout: float | None = None
    ) -> dict:
        return await self.conn.send(method, params, self.session_id, timeout)

    async def evaluate(self, expression: str, timeout: float | None = None) -> Any:
        result = await self.send(
            "Runtime.evaluate",
            {"expression": expression, "awaitPromise": True, "returnByValue": True},
            timeout,
        )
        details = result.get("exceptionDetails")
        if details:
            exc = details.get("exception") or {}
            text = exc.get("description") or details.get("text") or "error"
            first = text.strip().splitlines()[0]
            raise BridgeError(re.sub(r"^(Uncaught )?Error: ", "", first))
        return (result.get("result") or {}).get("value")

    async def navigate(self, url: str) -> None:
        await self.send("Page.navigate", {"url": url})

    async def detach(self) -> None:
        if not self.closed:
            with contextlib.suppress(BridgeError):
                await self.conn.send(
                    "Target.detachFromTarget", {"sessionId": self.session_id}
                )


class EditorTab:
    """A Kaggle editor tab; ``request`` calls a method of editor_bridge.js in it."""

    def __init__(self, page: CdpPage, ref: str, where: str):
        self.page = page
        self.ref = ref
        self.where = where  # "your Microsoft Edge", "kaggle-mcp's Chrome window", ...
        self.connected_at = time.time()

    @property
    def open(self) -> bool:
        return not self.page.closed

    def describe(self) -> dict[str, Any]:
        return {
            "notebook": self.ref,
            "browser": self.where,
            "connected_seconds": round(time.time() - self.connected_at),
        }

    async def request(
        self, method: str, params: dict | None = None, timeout: float | None = None
    ) -> Any:
        call = (
            f"(window.__kaggleMcpBridge ? window.__kaggleMcpBridge.call("
            f"{json.dumps(method)}, {json.dumps(params or {})}) : {json.dumps(_MISSING)})"
        )
        value = await self.page.evaluate(call, timeout)
        if value == _MISSING:
            # First call on this page load: install the bridge, then retry.
            await self.page.evaluate(BRIDGE_JS, timeout)
            value = await self.page.evaluate(call, timeout)
        return value


class KaggleBrowser:
    def __init__(
        self,
        profile_dir: Path,
        chrome: str | None = None,
        headless: bool = False,
        prefer: str | None = None,
    ):
        self.profile_dir = profile_dir
        self.chrome = chrome
        self.headless = headless
        # "auto": the user's browser if remote debugging is on, else our own window.
        self.prefer = (prefer or os.environ.get("KAGGLE_MCP_BROWSER") or "auto").lower()
        self.conn: CdpConnection | None = None
        self.kind: str | None = None  # "user" | "own"
        self.name: str | None = None  # "Microsoft Edge", "Google Chrome", ...
        self.tabs: dict[str, EditorTab] = {}

    @property
    def running(self) -> bool:
        return self.conn is not None and not self.conn.closed

    @property
    def where(self) -> str:
        name = self.name or "browser"
        return f"your {name}" if self.kind == "user" else f"kaggle-mcp's {name} window"

    # -- connecting --------------------------------------------------------------

    async def _connect(
        self, active: tuple[int, str], kind: str, name: str | None, timeout: float
    ) -> bool:
        try:
            self.conn = await CdpConnection.open(*active, timeout=timeout)
        except (OSError, InvalidHandshake, TimeoutError) as exc:
            logging.info(
                "Could not attach to %s (%s) on %s: %s", name, kind, active[0], exc
            )
            self.conn = None
            if kind == "user" and not isinstance(exc, ConnectionRefusedError):
                raise BrowserUnavailable(
                    f"Your {name} did not let kaggle-mcp connect. When it asks whether to "
                    "allow remote debugging, click Allow, then try again."
                ) from exc
            return False  # a stale DevToolsActivePort from a browser that has quit
        self.kind, self.name = kind, name
        self.tabs.clear()
        logging.info("Attached to %s", self.where)
        return True

    async def attach_existing(
        self, *, approval_timeout: float = APPROVAL_TIMEOUT
    ) -> bool:
        """Attach to a browser that is already running; never launches one."""
        if self.running:
            return True
        if self.prefer in ("auto", "user"):
            for browser in known_browsers():
                active = read_active_port(browser.user_data_dir)
                if active and await self._connect(
                    active, "user", browser.name, approval_timeout
                ):
                    return True
        if self.prefer in ("auto", "own"):
            active = read_active_port(self.profile_dir)
            if active and await self._connect(active, "own", self._own_name(), 10):
                return True
        return False

    def _own_name(self) -> str:
        found = None if self.chrome else find_browser()
        return found[1].name if found and found[1] else "browser"

    def enable_hint(self) -> str:
        browsers = known_browsers()
        return browsers[0].enable_hint if browsers else ""

    async def ensure_running(self) -> None:
        if await self.attach_existing():
            return
        if self.prefer == "user":
            raise BrowserUnavailable(
                "KAGGLE_MCP_BROWSER=user, but remote debugging is not on in your browser. "
                + self.enable_hint()
            )
        found = (self.chrome, None) if self.chrome else find_browser()
        if not found or not found[0]:
            raise BrowserUnavailable(
                "Could not find Microsoft Edge, Google Chrome, Brave or Chromium. Install "
                "one, or set KAGGLE_MCP_CHROME to its executable."
            )
        executable, browser = found
        name = browser.name if browser else Path(executable).stem
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        (self.profile_dir / "DevToolsActivePort").unlink(missing_ok=True)
        args = [
            executable,
            f"--user-data-dir={self.profile_dir}",
            "--remote-debugging-port=0",
            "--no-first-run",
            "--no-default-browser-check",
        ]
        if self.headless:
            args.append("--headless=new")
        args.append("about:blank")
        logging.info("Launching %s for kaggle-mcp: %s", name, executable)
        try:
            subprocess.Popen(
                args,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise BrowserUnavailable(
                f"Could not start {name} ({executable}): {exc}"
            ) from exc
        deadline = time.monotonic() + LAUNCH_TIMEOUT
        while time.monotonic() < deadline:
            active = read_active_port(self.profile_dir)
            if active and await self._connect(active, "own", name, 10):
                return
            await asyncio.sleep(0.25)
        raise BrowserUnavailable(f"{name} started but its DevTools port never came up.")

    async def pages(self) -> list[dict]:
        result = await self.conn.send("Target.getTargets")
        return [t for t in result.get("targetInfos", []) if t.get("type") == "page"]

    # -- editor tabs ---------------------------------------------------------------

    def editor_for(self, ref: str | None) -> EditorTab | None:
        """An attached tab for ``ref`` (or any editor tab if ref is None); no I/O."""
        tabs = [t for t in self.tabs.values() if t.open]
        if ref is not None:
            tabs = [t for t in tabs if t.ref.lower() == ref.lower()]
        return max(tabs, key=lambda t: t.connected_at) if tabs else None

    async def open_editor_refs(self) -> list[str]:
        """Notebooks open in editor tabs of a browser that is already running."""
        try:
            if not await self.attach_existing():
                return []
        except BrowserUnavailable:
            return []
        return [
            r for r in (editor_ref(p.get("url", "")) for p in await self.pages()) if r
        ]

    async def open_editor(self, ref: NotebookRef, timeout: float) -> EditorTab:
        """Find or open the notebook's editor tab and wait until it can be driven."""
        tab = self.editor_for(ref.ref)
        if tab is not None:
            return tab
        await self.ensure_running()
        target_id = None
        for page in await self.pages():
            if (editor_ref(page.get("url", "")) or "").lower() == ref.ref.lower():
                target_id = page["targetId"]
                break
        if target_id is None:
            created = await self.conn.send("Target.createTarget", {"url": ref.edit_url})
            target_id = created["targetId"]
        with contextlib.suppress(BridgeError):
            await self.conn.send("Target.activateTarget", {"targetId": target_id})
        page = await CdpPage.attach(self.conn, target_id)
        tab = EditorTab(page, ref.ref, self.where)
        try:
            await self._wait_ready(tab, ref, timeout)
        except BaseException:
            await page.detach()
            raise
        self.tabs[target_id] = tab
        return tab

    async def _wait_ready(
        self, tab: EditorTab, ref: NotebookRef, timeout: float
    ) -> None:
        deadline = time.monotonic() + timeout
        last: Any = None
        settled = 0  # consecutive polls showing a loaded page that is not the editor
        while time.monotonic() < deadline:
            try:
                state = await tab.request("ready", timeout=10)
            except BridgeError as exc:
                if tab.page.closed:
                    raise BridgeError("The Kaggle editor tab was closed.") from exc
                state = {"error": str(exc)}  # the page is navigating; try again
            if state and state.get("ready"):
                return
            last = state
            off_editor = bool(
                state and state.get("loaded") and state.get("ref") is None
            )
            settled = settled + 1 if off_editor else 0
            # Kaggle renders client-side: give the page a few seconds to settle.
            if off_editor and (state.get("notFound") or settled >= 4):
                await self._not_editor(tab, ref, state)
            await asyncio.sleep(1)
        raise BridgeError(
            f"The Kaggle editor for {ref.ref} did not become ready within {timeout:.0f}s "
            f"(last state: {last}). It may still be starting its session; call "
            "open_kaggle_connection again shortly."
        )

    async def _not_editor(self, tab: EditorTab, ref: NotebookRef, state: dict) -> None:
        """The tab finished loading a Kaggle page that is not the editor."""
        path = state.get("path") or ""
        if path.startswith("/account/login") or state.get("notFound"):
            # A private notebook looks missing to a signed-out browser.
            login = (
                f"https://www.kaggle.com/account/login?returnUrl={quote(ref.edit_url)}"
            )
            if not path.startswith("/account/login"):
                await tab.page.navigate(login)
            hint = "" if self.kind == "user" else f" {self.enable_hint()}"
            raise SignInRequired(
                f"Kaggle did not open {ref.ref} in {self.where}: it is not signed in to Kaggle "
                "(or the notebook does not exist). Kaggle's sign-in page is open there now; "
                "ask the user to sign in (once, it is remembered) and then call "
                f"open_kaggle_connection again.{hint}"
            )
        raise BridgeError(
            f"Kaggle opened {ref.ref} read-only instead of in the editor ({path}). Only "
            "your own notebooks can be edited; for someone else's, use Copy & Edit on "
            "Kaggle and open the copy."
        )

    async def close(self) -> None:
        """Drop the DevTools connection; the browser itself keeps running."""
        for tab in self.tabs.values():
            await tab.page.detach()
        self.tabs.clear()
        if self.conn is not None:
            await self.conn.close()
            self.conn = None

    async def shutdown(self) -> None:
        """Quit the browser if kaggle-mcp started it (never the user's own)."""
        if self.running and self.kind == "own":
            with contextlib.suppress(BridgeError):
                await self.conn.send("Browser.close", timeout=5)
        await self.close()
