"""Collapse nbformat outputs into what an agent needs to read.

Same shape as the colab-mcp fork's composite tools return, so agents see
identical results from either server: stdout, stderr, result, error, and
plots as image content.
"""

from __future__ import annotations

import json
import re
from typing import Any

from fastmcp.tools.tool import ToolResult
from mcp.types import ImageContent, TextContent

MAX_IMAGES_PER_RESULT = 4
MAX_IMAGE_BYTES = 2 * 1024 * 1024

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def join_text(text: Any) -> str:
    if isinstance(text, list):
        return "".join(str(t) for t in text)
    return "" if text is None else str(text)


def plain(text: Any) -> str:
    return _ANSI.sub("", join_text(text))


def truncate(text: str, limit: int) -> tuple[str, bool]:
    if limit <= 0 or len(text) <= limit:
        return text, False
    head = limit // 2
    tail = limit - head
    return (
        f"{text[:head]}\n... [{len(text) - limit} chars truncated] ...\n{text[-tail:]}",
        True,
    )


def summarize_outputs(
    outputs: list[dict] | None, max_chars: int = 4000
) -> dict[str, Any]:
    """stdout/stderr/result/error plus image blobs (under ``_images``)."""
    stdout, stderr, results, images = [], [], [], []
    error = None
    for out in outputs or []:
        kind = out.get("output_type")
        if kind == "stream":
            (stderr if out.get("name") == "stderr" else stdout).append(
                join_text(out.get("text"))
            )
        elif kind in ("execute_result", "display_data"):
            data = out.get("data") or {}
            for mime, blob in data.items():
                if mime.startswith("image/") and len(images) < MAX_IMAGES_PER_RESULT:
                    raw = join_text(blob).strip()
                    if len(raw) * 3 // 4 <= MAX_IMAGE_BYTES:
                        images.append({"mime": mime, "base64": raw})
            if "text/plain" in data:
                results.append(join_text(data["text/plain"]))
        elif kind == "error":
            error = {
                "ename": out.get("ename"),
                "evalue": out.get("evalue"),
                "traceback": truncate(
                    "\n".join(plain(line) for line in out.get("traceback") or []),
                    max_chars,
                )[0],
            }
    out_text, t1 = truncate("".join(stdout), max_chars)
    err_text, t2 = truncate("".join(stderr), max_chars)
    res_text, t3 = truncate("\n".join(results), max_chars)
    return {
        "ok": error is None,
        "stdout": out_text,
        "stderr": err_text,
        "result": res_text,
        "error": error,
        "image_count": len(images),
        "truncated": t1 or t2 or t3,
        "_images": images,
    }


def strip_images(summary: dict[str, Any]) -> tuple[dict[str, Any], list[dict]]:
    images = summary.pop("_images", [])
    return summary, images


def result_with_images(payload: dict[str, Any], images: list[dict]) -> ToolResult:
    content: list = [TextContent(type="text", text=json.dumps(payload))]
    for image in images:
        content.append(
            ImageContent(type="image", data=image["base64"], mimeType=image["mime"])
        )
    return ToolResult(content=content, structured_content=payload)
