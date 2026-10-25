# kaggle-mcp

An MCP server that connects your local agent to a **live Kaggle notebook**: open an
existing notebook, add and edit cells, and run them on Kaggle's GPU/TPU kernel, all in
the Kaggle editor you are watching. It mirrors the tool surface of the local `colab-mcp`
fork so the same habits work on both.

## How it works

Colab's page runs an MCP bridge that `colab-mcp` connects to. Kaggle has no such bridge,
so kaggle-mcp drives the Kaggle editor in your browser itself, over the DevTools protocol:

- `open_kaggle_connection(notebook_url=...)` opens (or reuses) the notebook's editor tab.
- kaggle-mcp evaluates a small script (`src/kaggle_mcp/editor_bridge.js`) in that page.
  Kaggle's editor is JupyterLab in an iframe, controlled by the page through an internal
  RPC client; the script finds that client and calls it.
- The cell tools then work on the editor itself: cells you add appear in it, runs show
  their output live there, and the code runs on that editor's Kaggle session. Kaggle
  autosaves the editor's draft as usual.

### Which browser

Any Chromium browser works: Microsoft Edge, Google Chrome, Brave or Chromium.

- **Your everyday browser (recommended).** In Edge open `edge://inspect/#remote-debugging`
  (Chrome: `chrome://inspect/#remote-debugging`; version 144+) and turn remote debugging on.
  kaggle-mcp then uses your browser and your Kaggle sign-in. The browser asks you to allow
  the connection when kaggle-mcp attaches and shows a banner while it is connected. While
  remote debugging is on, any local program can ask to control your browser, so turn it
  off when you do not need it.
- **Otherwise, its own window.** kaggle-mcp starts your default Chromium browser with a
  separate profile (`~/.kaggle-mcp/chrome-profile`; browsers refuse remote debugging on
  your normal profile). Sign in to Kaggle there once and it is remembered. The window
  stays open when kaggle-mcp exits, and the next start reuses it.

Only notebooks you can edit open in the editor; for someone else's, use *Copy & Edit*
on Kaggle first. If a tab or the browser is closed, the next cell tool reopens it.

### Without a browser: the local working copy

`open_kaggle_connection(..., use_browser=False)` (or `--no-browser`) uses the original
mode instead. Kaggle's notebook editor can expose the session as a standard
**Jupyter server** (*Run → Kaggle Jupyter Server*, the "VS Code compatible URL"):

| Piece | Where it lives |
| --- | --- |
| Notebook cells | Pulled through the Kaggle API into a local working copy (`~/.kaggle-mcp/notebooks/<owner>__<slug>.ipynb`), edited by the cell tools |
| Code execution | A kernel on the Kaggle session's Jupyter server (REST + websocket, via `jupyter-kernel-client`) |
| Saving | `save_notebook` pushes the working copy back through the Kaggle API as a new version |

In this mode the browser editor and the working copy are separate documents. Edits made
here appear on Kaggle after `save_notebook`; edits made in the browser appear here after
`pull_notebook`. They share the kernel when both are attached to the same session.

## Setup

1. Register the server (Claude Code, user scope):

   ```bash
   claude mcp add -s user kaggle-mcp -- uv run --project /path/to/kaggle-mcp kaggle-mcp
   ```

2. Turn on remote debugging in your browser (see [Which browser](#which-browser)), or let
   kaggle-mcp open its own window and sign in to Kaggle there once.

3. Optional: a Kaggle API token, needed for `save_notebook` (new versions, *Save & Run All*)
   and for the local-working-copy mode. kaggle.com → Settings → API → *Generate New Token*:

   ```bash
   mkdir -p ~/.kaggle && pbpaste > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
   ```

   (`KAGGLE_API_TOKEN` in the server's `env`, or a legacy `~/.kaggle/kaggle.json`, also work.)

## Connecting to a notebook

`open_kaggle_connection(notebook_url=..., server_url=None, accelerator=None, use_browser=True)`

- `notebook_url`: `kaggle.com/code/<owner>/<slug>` (with or without `/edit`) or `owner/slug`.
  With no `notebook_url`, a notebook already open in the browser's editor is used.
- By default the editor opens in the browser (above). If that browser is not signed in to
  Kaggle, the result has `sign_in_required` and the sign-in page is open in it; sign in and
  call again. The accelerator is chosen in the editor's settings panel.
- If you also pass `server_url`, kaggle-mcp attaches to that Jupyter server too and uses it
  for file transfers, so they do not show up as temporary cells.
- With `use_browser=False`, the server is chosen in this order:
  1. `server_url` if you pass one: the *VS Code compatible URL* from **Run → Kaggle Jupyter Server**;
  2. the saved server from last time (reconnects to the same kernel, so variables survive an MCP restart);
  3. a session started through the Kaggle API with `accelerator` (`t4`, `l4`, `a100`, `tpu`, or a machine-shape id);
  4. otherwise the notebook editor opens in your browser and the tool returns the steps: start
     *Kaggle Jupyter Server*, copy the URL, and call again with `server_url`.
- A running kernel on the server is reused unless `new_kernel=True` or `kernel_id` names another.
- The result includes the runtime: GPUs (`nvidia-smi -L`), TPU presence, and `/kaggle/input` contents.

Kaggle's session URLs look like `https://kkb-production.jupyter-proxy.kaggle.net/k/<session>/<JWE>/proxy`:
the credential is the path segment, so treat the whole URL as a secret. Status output and logs redact it.

The API session path (3) uses `CreateKernelSession`, whose response format is not documented
(observed Oct 2026: `jupyterUrl` and `tokenizedJupyterUrl`; both are tried).
If no Jupyter URL is found in the response, the session is cancelled so it does not use quota,
the response is logged, and the browser path (4) is used instead.

## Tools

While a cell is running in the editor, edits that rewrite the notebook (`update_cell`,
`move_cell`, `delete_cell`) are refused, because they would cut off its output; adding a
code cell still works.

| Tool | What it does |
| --- | --- |
| `open_kaggle_connection` | Open a notebook in the browser editor, or pull it and attach to a kernel (see above). |
| `kaggle_connection_status` | Notebook, kernel liveness, runtime, unsaved edits, Kaggle session state, log dir. |
| `disconnect_kaggle` | Detach; `shutdown_kernel` kills it, `stop_session` stops an API-started session. |
| `get_cells`, `add_code_cell`, `add_text_cell`, `update_cell`, `move_cell`, `delete_cell`, `run_code_cell` | Colab's cell primitives, with Colab's argument names. |
| `get_notebook_outline` | Index, id, type, first line, output and error flags per cell. |
| `run_cells`, `run_notebook` | Run by id list or index range, or all; `background=True` returns a job id. |
| `run_code` | Run ad-hoc Python. In the editor it runs as a visible cell at the end that is removed afterwards; `keep_cell=True` keeps it. Plots come back as images. |
| `job_status`, `list_jobs`, `cancel_job` | Background runs. Unlike Colab, `cancel_job` **interrupts the running cell**. |
| `interrupt_kernel`, `restart_kernel` | Direct kernel control. |
| `upload_file`, `download_file` | Copy files to/from the Kaggle machine (e.g. `/kaggle/working`), SHA-256 verified. |
| `export_notebook` | Write the notebook (the editor's, or the working copy) as `.ipynb` locally. |
| `pull_notebook` | Local-copy mode: replace the working copy with Kaggle's latest version (refuses over unsaved edits unless told). |
| `save_notebook` | New version on Kaggle. `quick_save=True` (default) saves without running; `False` is *Save & Run All*. Data sources, privacy and internet settings are preserved; outputs are stripped unless `include_outputs`. |

## Configuration

| Setting | Default | Meaning |
| --- | --- | --- |
| `--state-dir` / `KAGGLE_MCP_STATE_DIR` | `~/.kaggle-mcp` | Saved connection (0600, contains the session URL) and working copies |
| `--ephemeral` | off | Persist nothing |
| `-l/--log` | new temp dir | Log directory (shown by `kaggle_connection_status`) |
| `KAGGLE_MCP_AUTO_RECONNECT` | `1` | Reattach to the saved server/kernel when a tool runs with none connected |
| `KAGGLE_MCP_SESSION_TIMEOUT` | `300` | Seconds to wait for an API-started session |
| `KAGGLE_MCP_UPLOAD_CHUNK`, `KAGGLE_MCP_DOWNLOAD_CHUNK` | 1 MiB | File transfer chunk sizes |
| `--no-browser` | off | Never open the editor in a browser; always use the local working copy |
| `KAGGLE_MCP_BROWSER` | `auto` | `user`: only your own browser (remote debugging on); `own`: only kaggle-mcp's window; `auto`: yours if available, else its own |
| `KAGGLE_MCP_CHROME` | default browser | Executable for kaggle-mcp's own window |
| `KAGGLE_MCP_EDITOR_WAIT` | `120` | Seconds to wait for the editor to load (it starts a Kaggle session first) |
| `KAGGLE_MCP_APPROVAL_TIMEOUT` | `120` | Seconds to wait for you to allow the connection in your browser |
| `KAGGLE_MCP_EDITOR_TIMEOUT` | `60` | Seconds to wait for one editor call |

## Development

```bash
uv sync
uv run pytest
```

The end-to-end tests start a real local Jupyter server and drive the MCP tools through
an MCP client, with the Kaggle API faked.
