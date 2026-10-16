"""A kernel on the Kaggle session's Jupyter server.

jupyter-kernel-client speaks the standard Jupyter REST + websocket protocol
(what VS Code uses with Kaggle's "VS Code compatible URL"). It is synchronous,
so calls run in worker threads; executions are serialised with a lock because
a kernel runs one request at a time anyway.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from kaggle_mcp.refs import ServerEndpoint


class KernelUnavailable(RuntimeError):
    pass


def _kernel_client_cls():
    from jupyter_kernel_client import KernelClient

    return KernelClient


def _quiet(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # cleanup paths: log, never raise
        logging.debug("ignored during kernel cleanup: %s", exc)


class RemoteKernel:
    def __init__(self, endpoint: ServerEndpoint, client_cls=None):
        self.endpoint = endpoint
        self._client_cls = client_cls or _kernel_client_cls()
        self._client = None
        self.kernel_id: str | None = None
        self.started_here = False
        self._lock = asyncio.Lock()

    def _new_client(self, kernel_id: str | None = None):
        return self._client_cls(
            server_url=self.endpoint.base_url,
            token=self.endpoint.token,
            kernel_id=kernel_id,
        )

    async def list_kernels(self) -> list[dict[str, Any]]:
        def call():
            probe = self._new_client()
            try:
                return probe.list_kernels()
            finally:
                # Never started a kernel, so there is nothing to shut down.
                probe._own_kernel = False

        try:
            return await asyncio.to_thread(call)
        except Exception as exc:
            raise KernelUnavailable(
                f"Could not reach the Jupyter server at {self.endpoint.redacted}: {exc}"
            ) from exc

    async def connect(
        self,
        kernel_id: str | None = None,
        new_kernel: bool = False,
        kernel_name: str = "python3",
    ) -> dict[str, Any]:
        """Attach to ``kernel_id``, else the most recently active kernel, else start one."""
        kernels = await self.list_kernels()
        known = {k.get("id") for k in kernels}
        target = None
        if kernel_id and kernel_id in known:
            target = kernel_id
        elif kernel_id:
            logging.info("Kernel %s is gone; choosing another", kernel_id)
        if target is None and not new_kernel and kernels:
            latest = max(kernels, key=lambda k: k.get("last_activity") or "")
            target = latest.get("id")

        def attach():
            client = self._new_client(target)
            client.start(name=kernel_name)
            # The client shuts down kernels it started when garbage collected;
            # the kernel must outlive this process so a restart can reattach.
            client._own_kernel = False
            return client

        try:
            self._client = await asyncio.to_thread(attach)
        except Exception as exc:
            raise KernelUnavailable(
                f"Could not open a kernel on {self.endpoint.redacted}: {exc}"
            ) from exc
        self.kernel_id = self._client.id
        self.started_here = target is None
        return {
            "kernel_id": self.kernel_id,
            "attached_to_existing": not self.started_here,
            "kernels_on_server": len(kernels),
        }

    @property
    def connected(self) -> bool:
        return self._client is not None

    async def is_alive(self) -> bool:
        if self._client is None:
            return False
        try:
            return bool(await asyncio.to_thread(self._client.is_alive))
        except Exception:
            return False

    async def execute(self, code: str, timeout: float | None = None) -> dict[str, Any]:
        """Run ``code``; returns {"execution_count", "status", "outputs"} (nbformat outputs)."""
        if self._client is None:
            raise KernelUnavailable("No kernel is connected")
        async with self._lock:
            client = self._client
            try:
                return await asyncio.to_thread(
                    client.execute, code, timeout=timeout, stop_on_error=False
                )
            except TimeoutError:
                raise
            except Exception as exc:
                raise KernelUnavailable(
                    f"Lost the kernel during execution: {exc}"
                ) from exc

    async def interrupt(self) -> None:
        if self._client is None:
            raise KernelUnavailable("No kernel is connected")
        # Goes over REST, so it does not wait for the execution lock.
        await asyncio.to_thread(self._client.interrupt)

    async def restart(self) -> None:
        if self._client is None:
            raise KernelUnavailable("No kernel is connected")
        await asyncio.to_thread(self._client.restart)

    async def close(self, shutdown_kernel: bool = False) -> None:
        client, self._client = self._client, None
        if client is None:
            return
        # Detach from the client's own finaliser so dropping it never kills the kernel.
        client._own_kernel = False
        await asyncio.to_thread(_quiet, client.stop, shutdown_kernel=shutdown_kernel)
