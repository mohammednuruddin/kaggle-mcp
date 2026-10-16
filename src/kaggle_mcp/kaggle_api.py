"""The Kaggle public API calls this server needs, via kagglesdk.

Credentials are resolved the way the kaggle CLI does it: ``KAGGLE_API_TOKEN``
(a KGAT token or a path to one), ``~/.kaggle/access_token``, or the legacy
``~/.kaggle/kaggle.json``. Every call is synchronous in kagglesdk, so each one
runs in a worker thread.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit

from kaggle_mcp.refs import NotebookRef

# Accelerator ids accepted by Kaggle as machine_shape (Sep 2026 list).
ACCELERATORS = {
    "none": None,
    "cpu": None,
    "t4": "NvidiaTeslaT4",
    "gpu": "NvidiaTeslaT4",
    "l4": "NvidiaL4",
    "a100": "NvidiaTeslaA100",
    "tpu": "TpuV5E8",
    "v5e8": "TpuV5E8",
    "v6e8": "TpuV6E8",
}


class KaggleApiError(RuntimeError):
    pass


class SessionAlreadyRunning(KaggleApiError):
    pass


def resolve_accelerator(value: str | None) -> str | None:
    """Map a friendly name (t4, l4, a100, tpu) or a raw machine shape to a shape id."""
    if value is None or not value.strip():
        return None
    key = value.strip().lower().replace("-", "").replace(" ", "")
    if key in ACCELERATORS:
        return ACCELERATORS[key]
    return value.strip()


def credentials_source() -> str | None:
    """Where kagglesdk will find credentials, without reading the secret itself."""
    if os.environ.get("KAGGLE_API_TOKEN"):
        return "KAGGLE_API_TOKEN"
    home = Path.home() / ".kaggle"
    for name in ("access_token", "access_token.txt", "kaggle.json"):
        if (home / name).is_file():
            return f"~/.kaggle/{name}"
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return "KAGGLE_USERNAME/KAGGLE_KEY"
    return None


@dataclass
class KaggleNotebook:
    ref: NotebookRef
    source: str
    language: str
    kernel_type: str
    title: str
    version: int | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class StartedSession:
    server_urls: list[str]
    kaggle_session_id: int | None
    operation: dict[str, Any]

    @property
    def server_url(self) -> str | None:
        return self.server_urls[0] if self.server_urls else None


def _plain(obj: Any) -> Any:
    """kagglesdk objects and protobuf-ish payloads to plain JSON-able values."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    to_dict = getattr(type(obj), "to_dict", None)
    if callable(to_dict):
        try:
            return _plain(to_dict(obj))
        except Exception:
            pass
    return str(obj)


_URL_HINTS = ("token=", "jupyter", "proxy", "kkb", "kaggle.net")


def find_server_url(payload: Any) -> str | None:
    """The Jupyter server URL in an operation response, if there is one.

    The response shape of CreateKernelSession is not documented, so this walks
    the payload for http(s) strings and prefers ones that look like a Jupyter
    proxy over plain kaggle.com page links.
    """
    found: list[tuple[int, str]] = []

    def walk(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, str(k).lower())
        elif isinstance(value, list):
            for v in value:
                walk(v, key)
        elif isinstance(value, str) and value.startswith(("http://", "https://")):
            text = value.lower()
            score = sum(hint in text for hint in _URL_HINTS)
            score += 2 if any(h in key for h in ("jupyter", "server", "proxy")) else 0
            if text.startswith(
                ("https://www.kaggle.com/code", "https://kaggle.com/code")
            ):
                score -= 3
            found.append((score, value))

    walk(payload)
    found.sort(key=lambda item: item[0], reverse=True)
    return found[0][1] if found and found[0][0] > 0 else None


def server_url_candidates(payload: Any) -> list[str]:
    """Jupyter URLs to try, best first.

    CreateKernelSession answers with ``jupyterUrl`` and ``tokenizedJupyterUrl``
    (seen Oct 2026). The tokenized one can be a bare proxy host plus
    ``?token=``, which is not itself a Jupyter root, so the plain URL with that
    token added comes first and the tokenized one is kept as a fallback.
    """
    response = payload.get("response") if isinstance(payload, dict) else None
    candidates: list[str] = []
    if isinstance(response, dict):
        plain = response.get("jupyterUrl") or response.get("jupyter_url")
        tokenized = response.get("tokenizedJupyterUrl") or response.get(
            "tokenized_jupyter_url"
        )
        token = None
        if isinstance(tokenized, str):
            query = dict(parse_qsl(urlsplit(tokenized).query))
            token = query.get("token")
        if isinstance(plain, str) and plain.startswith("http"):
            if token and "token=" not in plain:
                sep = "&" if "?" in plain else "?"
                candidates.append(f"{plain}{sep}{urlencode({'token': token})}")
            candidates.append(plain)
        if isinstance(tokenized, str) and tokenized.startswith("http"):
            candidates.append(tokenized)
    fallback = find_server_url(payload)
    if fallback:
        candidates.append(fallback)
    return list(dict.fromkeys(candidates))


def url_shape(url: str | None) -> str | None:
    """A URL with every secret-looking part replaced, for logs."""
    if not url:
        return url
    parts = urlsplit(url)
    path = re.sub(r"(?<=/)[A-Za-z0-9._~-]{20,}(?=/|$)", "<secret>", parts.path)
    keys = [k for k, _ in parse_qsl(parts.query, keep_blank_values=True)]
    return f"{parts.scheme}://{parts.netloc}{path}" + (f" query={keys}" if keys else "")


def find_session_id(payload: Any) -> int | None:
    def walk(value: Any) -> int | None:
        if isinstance(value, dict):
            for k, v in value.items():
                key = str(k).lower().replace("_", "")
                if key in ("kernelsessionid", "sessionid") and str(v).isdigit():
                    return int(v)
                hit = walk(v)
                if hit is not None:
                    return hit
        elif isinstance(value, list):
            for v in value:
                hit = walk(v)
                if hit is not None:
                    return hit
        return None

    return walk(payload)


class KaggleApi:
    """Thin async wrapper over the kagglesdk calls used by the tools."""

    def __init__(self, client_factory=None):
        self._client_factory = client_factory

    def _client(self):
        if self._client_factory is not None:
            return self._client_factory()
        from kagglesdk import KaggleClient

        return KaggleClient()

    @property
    def configured(self) -> bool:
        return self._client_factory is not None or credentials_source() is not None

    def _require(self) -> None:
        if not self.configured:
            raise KaggleApiError(
                "No Kaggle credentials found. Create a token at kaggle.com/settings "
                "(API > Generate New Token) and save it to ~/.kaggle/access_token, or set "
                "KAGGLE_API_TOKEN in the MCP server's env."
            )

    async def _run(self, fn):
        self._require()
        try:
            return await asyncio.to_thread(fn)
        except KaggleApiError:
            raise
        except Exception as exc:
            raise KaggleApiError(f"Kaggle API call failed: {exc}") from exc

    async def get_notebook(self, ref: NotebookRef) -> KaggleNotebook:
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest

        def call():
            with self._client() as kaggle:
                request = ApiGetKernelRequest()
                request.user_name = ref.owner
                request.kernel_slug = ref.slug
                return kaggle.kernels.kernels_api_client.get_kernel(request)

        response = await self._run(call)
        blob, meta = response.blob, response.metadata
        if blob is None or meta is None:
            raise KaggleApiError(f"Kaggle returned no source for {ref.ref}")
        return KaggleNotebook(
            ref=ref,
            source=blob.source or "",
            language=(blob.language or meta.language or "python").lower(),
            kernel_type=(blob.kernel_type or meta.kernel_type or "notebook").lower(),
            title=meta.title or ref.slug,
            version=meta.current_version_number,
            metadata={
                "id": meta.id,
                "title": meta.title,
                "is_private": meta.is_private,
                "enable_gpu": meta.enable_gpu,
                "enable_tpu": meta.enable_tpu,
                "enable_internet": meta.enable_internet,
                "dataset_data_sources": list(meta.dataset_data_sources or []),
                "kernel_data_sources": list(meta.kernel_data_sources or []),
                "competition_data_sources": list(meta.competition_data_sources or []),
                "model_data_sources": list(meta.model_data_sources or []),
                "category_ids": list(meta.category_ids or []),
                "docker_image": meta.docker_image,
                "machine_shape": meta.machine_shape,
            },
        )

    async def save_notebook(
        self,
        notebook: KaggleNotebook,
        text: str,
        quick_save: bool = True,
        machine_shape: str | None = None,
    ) -> dict[str, Any]:
        """Push ``text`` as a new version, keeping the notebook's settings and data sources."""
        from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
        from kagglesdk.kernels.types.kernels_enums import KernelExecutionType

        meta = notebook.metadata

        def call():
            with self._client() as kaggle:
                request = ApiSaveKernelRequest()
                if meta.get("id"):
                    request.id = meta["id"]
                request.slug = notebook.ref.ref
                request.text = text
                request.language = notebook.language
                request.kernel_type = notebook.kernel_type
                if meta.get("is_private") is not None:
                    request.is_private = meta["is_private"]
                if meta.get("enable_internet") is not None:
                    request.enable_internet = meta["enable_internet"]
                request.dataset_data_sources = meta.get("dataset_data_sources") or []
                request.kernel_data_sources = meta.get("kernel_data_sources") or []
                request.competition_data_sources = (
                    meta.get("competition_data_sources") or []
                )
                request.model_data_sources = meta.get("model_data_sources") or []
                request.category_ids = meta.get("category_ids") or []
                if meta.get("docker_image"):
                    request.docker_image = meta["docker_image"]
                shape = machine_shape or meta.get("machine_shape")
                if shape:
                    request.machine_shape = shape
                request.kernel_execution_type = (
                    KernelExecutionType.QUICK_SAVE
                    if quick_save
                    else KernelExecutionType.SAVE_AND_RUN_ALL
                )
                return kaggle.kernels.kernels_api_client.save_kernel(request)

        response = await self._run(call)
        if getattr(response, "error", None):
            raise KaggleApiError(f"Kaggle refused the save: {response.error}")
        return {
            "ref": response.ref,
            "url": response.url,
            "version_number": response.version_number,
            "invalid_dataset_sources": list(response.invalid_dataset_sources or []),
        }

    async def start_session(
        self,
        notebook: KaggleNotebook,
        machine_shape: str | None,
        enable_internet: bool | None,
        timeout: float,
    ) -> StartedSession:
        """Start an interactive session and wait for its operation to finish."""
        from kagglesdk.common.types.operations_service import GetOperationRequest
        from kagglesdk.kernels.types.kernels_api_service import (
            ApiCreateKernelSessionRequest,
        )

        def create():
            with self._client() as kaggle:
                request = ApiCreateKernelSessionRequest()
                request.slug = notebook.ref.ref
                request.language = notebook.language
                request.kernel_type = notebook.kernel_type
                if machine_shape:
                    request.machine_shape = machine_shape
                if enable_internet is not None:
                    request.enable_internet = enable_internet
                if notebook.metadata.get("docker_image"):
                    request.docker_image = notebook.metadata["docker_image"]
                return kaggle.kernels.kernels_api_client.create_kernel_session(request)

        try:
            operation = await self._run(create)
        except KaggleApiError as exc:
            if "409" in str(exc):
                raise SessionAlreadyRunning(
                    f"Kaggle already has a session running for {notebook.ref.ref}. Attach to it "
                    "with server_url (Run > Kaggle Jupyter Server in the editor shows its URL), "
                    "or stop it in the editor and try again."
                ) from exc
            raise
        deadline = time.monotonic() + timeout
        delay = 2.0
        while not operation.done and operation.name and time.monotonic() < deadline:
            await asyncio.sleep(delay)
            delay = min(delay * 1.5, 10.0)

            def poll(name=operation.name):
                with self._client() as kaggle:
                    request = GetOperationRequest()
                    request.name = name
                    return kaggle.common.operations_client.get_operation(request)

            operation = await self._run(poll)
        payload = {
            "name": operation.name,
            "done": operation.done,
            "metadata": _plain(operation.metadata),
            "response": _plain(operation.response),
            "error": _plain(operation.error),
        }
        candidates = server_url_candidates(payload)
        logging.info(
            "CreateKernelSession operation: done=%s keys=%s candidates=%s",
            operation.done,
            sorted((payload["response"] or {}).keys())
            if isinstance(payload["response"], dict)
            else type(payload["response"]).__name__,
            [url_shape(u) for u in candidates],
        )
        if operation.error is not None and getattr(operation.error, "message", ""):
            raise KaggleApiError(
                f"Kaggle could not start a session: {operation.error.message}"
            )
        return StartedSession(
            server_urls=candidates,
            kaggle_session_id=find_session_id(payload),
            operation=payload,
        )

    async def session_status(self, ref: NotebookRef) -> dict[str, Any]:
        from kagglesdk.kernels.types.kernels_api_service import (
            ApiGetKernelSessionStatusRequest,
        )

        def call():
            with self._client() as kaggle:
                request = ApiGetKernelSessionStatusRequest()
                request.user_name = ref.owner
                request.kernel_slug = ref.slug
                return kaggle.kernels.kernels_api_client.get_kernel_session_status(
                    request
                )

        response = await self._run(call)
        status = response.status
        return {
            "status": getattr(status, "name", str(status)),
            "failure_message": response.failure_message or None,
        }

    async def cancel_session(self, kaggle_session_id: int) -> str | None:
        from kagglesdk.kernels.types.kernels_api_service import (
            ApiCancelKernelSessionRequest,
        )

        def call():
            with self._client() as kaggle:
                request = ApiCancelKernelSessionRequest()
                request.kernel_session_id = kaggle_session_id
                return kaggle.kernels.kernels_api_client.cancel_kernel_session(request)

        response = await self._run(call)
        return response.error_message or None
