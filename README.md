# kaggle-mcp

An MCP server that connects your local agent to a **live Kaggle notebook**: open an
existing notebook, edit its cells, run them on Kaggle's GPU/TPU kernel, and save the
result back as a new version. It mirrors the tool surface of the local `colab-mcp` fork
so the same habits work on both.

## How it works

Colab's page runs an MCP bridge that `colab-mcp` connects to. Kaggle has no such bridge,
but its notebook editor can expose the session as a standard **Jupyter server**
(*Run → Kaggle Jupyter Server*, the "VS Code compatible URL"). This server uses that:

| Piece | Where it lives |
| --- | --- |
| Notebook cells | Pulled through the Kaggle API into a local working copy (`~/.kaggle-mcp/notebooks/<owner>__<slug>.ipynb`), edited by the cell tools |
| Code execution | A kernel on the Kaggle session's Jupyter server (REST + websocket, via `jupyter-kernel-client`) |
| Saving | `save_notebook` pushes the working copy back through the Kaggle API as a new version |

The browser editor and this working copy are separate documents. Edits made here appear
on Kaggle after `save_notebook`; edits made in the browser appear here after `pull_notebook`.
They share the kernel when both are attached to the same session.

## Setup

1. Create a Kaggle API token: kaggle.com → Settings → API → *Generate New Token*. Save it:

   ```bash
   mkdir -p ~/.kaggle && pbpaste > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token
   ```

   (`KAGGLE_API_TOKEN` in the server's `env`, or a legacy `~/.kaggle/kaggle.json`, also work.)
   Without a token you can still run code on a session URL you paste, but cells are not
   pulled and nothing can be saved.

2. Register the server (Claude Code, user scope):

   ```bash
   claude mcp add -s user kaggle-mcp -- ~/.local/bin/uv run --project ~/Documents/kaggle-mcp kaggle-mcp
   ```

## Connecting to a notebook

`open_kaggle_connection(notebook_url=..., server_url=None, accelerator=None)`

- `notebook_url`: `kaggle.com/code/<owner>/<slug>` (with or without `/edit`) or `owner/slug`.
- The server is chosen in this order:
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

| Tool | What it does |
| --- | --- |
| `open_kaggle_connection` | Pull a notebook and attach to a kernel (see above). |
| `kaggle_connection_status` | Notebook, kernel liveness, runtime, unsaved edits, Kaggle session state, log dir. |
| `disconnect_kaggle` | Detach; `shutdown_kernel` kills it, `stop_session` stops an API-started session. |
| `get_cells`, `add_code_cell`, `add_text_cell`, `update_cell`, `move_cell`, `delete_cell`, `run_code_cell` | Colab's cell primitives, with Colab's argument names. |
| `get_notebook_outline` | Index, id, type, first line, output and error flags per cell. |
| `run_cells`, `run_notebook` | Run by id list or index range, or all; `background=True` returns a job id. |
| `run_code` | Run ad-hoc Python without touching the notebook (`keep_cell=True` appends it). Plots come back as images. |
| `job_status`, `list_jobs`, `cancel_job` | Background runs. Unlike Colab, `cancel_job` **interrupts the running cell**. |
| `interrupt_kernel`, `restart_kernel` | Direct kernel control. |
| `upload_file`, `download_file` | Copy files to/from the Kaggle machine (e.g. `/kaggle/working`), SHA-256 verified. |
| `export_notebook` | Write the working copy as `.ipynb` locally. |
| `pull_notebook` | Replace the working copy with Kaggle's latest version (refuses over unsaved edits unless told). |
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

## Development

```bash
uv sync
uv run pytest
```

The end-to-end tests start a real local Jupyter server and drive the MCP tools through
an MCP client, with the Kaggle API faked.
