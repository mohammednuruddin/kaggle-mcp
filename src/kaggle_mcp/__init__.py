import argparse
import asyncio
import datetime
import logging
import sys
import tempfile
from pathlib import Path

from fastmcp import FastMCP

from kaggle_mcp.browser import KaggleBrowser
from kaggle_mcp.session import SERVER_INFO, KaggleSession
from kaggle_mcp.state import default_state_dir
from kaggle_mcp.tools import register_tools

INSTRUCTIONS = (
    "Connects to a Kaggle notebook. open_kaggle_connection opens the notebook's editor in a "
    "browser kaggle-mcp controls, and the cell tools work in that editor: cells you add "
    "and run appear there as they run, on the notebook's live Kaggle session (GPU/TPU). "
    "Start with open_kaggle_connection."
)


def init_logger(logdir: str) -> None:
    log_filename = datetime.datetime.now().strftime(
        f"{logdir}/kaggle-mcp.%Y-%m-%d_%H-%M-%S.log"
    )
    logging.basicConfig(
        format="%(asctime)s %(levelname)s:%(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
        filename=log_filename,
        level=logging.INFO,
    )


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="KaggleMCP is an MCP server that lets you work in a live Kaggle notebook."
    )
    parser.add_argument(
        "-l",
        "--log",
        help="directory for logfiles (default: a new temp directory)",
        default=None,
    )
    parser.add_argument(
        "--state-dir",
        help="directory for the saved connection and notebook working copies "
        "(default: ~/.kaggle-mcp, or KAGGLE_MCP_STATE_DIR)",
        default=None,
    )
    parser.add_argument(
        "--no-browser",
        help="never open the Kaggle editor in a browser; always use the local working copy",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ephemeral",
        help="do not persist the connection or working copies",
        action="store_true",
        default=False,
    )
    return parser.parse_args(argv)


def build_server(session: KaggleSession) -> FastMCP:
    mcp = FastMCP(name="KaggleMCP", instructions=INSTRUCTIONS)
    register_tools(mcp, session)
    return mcp


async def main_async() -> None:
    args = parse_args(sys.argv[1:])
    logdir = args.log or tempfile.mkdtemp(prefix="kaggle-mcp-logs-")
    init_logger(logdir)
    SERVER_INFO["log_dir"] = logdir
    state_dir = (
        Path(args.state_dir).expanduser() if args.state_dir else default_state_dir()
    )
    # Chrome is only launched when a notebook is opened, not at startup.
    browser = None if args.no_browser else KaggleBrowser(state_dir / "chrome-profile")
    session = KaggleSession(
        state_dir=state_dir, ephemeral=args.ephemeral, browser=browser
    )
    mcp = build_server(session)
    try:
        await mcp.run_async()
    finally:
        await session.close()
        if browser is not None:
            await browser.close()


def main() -> None:
    asyncio.run(main_async())
