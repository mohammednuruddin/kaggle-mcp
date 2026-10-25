// kaggle-mcp editor bridge.
//
// kaggle-mcp evaluates this in the Kaggle notebook editor page
// (kaggle.com/code/<owner>/<slug>/edit) of the Chrome window it controls, then
// calls window.__kaggleMcpBridge.call(method, params) through the DevTools
// protocol, so the agent's cell edits and runs happen in the visible editor.
//
// Kaggle's editor is JupyterLab inside a cross-origin iframe; the top page drives
// it through a Postmate RPC client (getSourceWithOutputs, setSource, insertCells,
// runCommand, executeActiveCell, stopExecution, ...). This script finds that
// client among the page's webpack modules by its shape, not by a module id,
// because ids change on every Kaggle deploy.
(() => {
  if (window.__kaggleMcpBridge) return;

  const VERSION = 1;
  const EDIT_PATH = /^\/code\/([^/]+)\/([^/]+)\/edit(?:\/|$)/;
  const BUSY =
    "A cell is running in the Kaggle editor. Rewriting the notebook now would cut off its output; " +
    "wait for it to finish or interrupt it first.";

  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

  // -- the editor's RPC client ---------------------------------------------------

  let client = null;

  function isClient(v) {
    return (
      v &&
      typeof v === "object" &&
      typeof v.getRAWSourceWithOutputs === "function" &&
      typeof v.runCommand === "function" &&
      typeof v.ensureConnected === "function"
    );
  }

  // The runtime exposes module factories (require.m) but not the module cache.
  // Requiring the one factory whose source defines the client returns the
  // instance the editor already created; no other module is executed.
  const CLIENT_MARKERS = ["frameChannel", "getRAWSourceWithOutputs", "runCommand", "ensureConnected"];

  function findClient() {
    for (const key of Object.keys(window)) {
      if (!key.startsWith("webpackChunk") || !Array.isArray(window[key])) continue;
      let req = null;
      try {
        window[key].push([[Symbol("kaggle-mcp")], {}, (r) => (req = r)]);
      } catch (e) {
        continue;
      }
      if (!req || !req.m) continue;
      for (const id of Object.keys(req.m)) {
        let src;
        try {
          src = Function.prototype.toString.call(req.m[id]);
        } catch (e) {
          continue;
        }
        if (src.length > 200000 || !CLIENT_MARKERS.every((m) => src.includes(m))) continue;
        let exp;
        try {
          exp = req(id);
        } catch (e) {
          continue;
        }
        for (const name of Object.keys(exp || {})) {
          let v;
          try {
            v = exp[name];
          } catch (e) {
            continue;
          }
          if (isClient(v)) return v;
        }
      }
    }
    return null;
  }

  async function getClient() {
    if (client && client.frameChannel) return client;
    client = findClient();
    if (!client) throw new Error("The Kaggle editor is not ready yet (no editor client found).");
    await client.ensureConnected();
    return client;
  }

  // Postmate's callAsync matches a reply by method name, so two calls of the
  // same method in flight can swap results. Every frame call goes through this
  // queue; a cell run only holds it until its request has been sent.
  let queue = Promise.resolve();
  function serial(fn) {
    const run = queue.then(fn, fn);
    queue = run.catch(() => {});
    return run;
  }

  async function frame(method, arg) {
    const c = await getClient();
    return c.frameChannel.callAsync(method, arg === undefined ? null : arg);
  }

  const command = (id) => frame("runCommand", id);

  async function readNotebook() {
    return JSON.parse(await frame("getSourceWithOutputs"));
  }

  async function writeNotebook(nb) {
    await frame("setSource", JSON.stringify(nb));
  }

  // -- kernel state (from the editor frame's own events) ---------------------------

  let kernelState = "unknown";
  let kernelChangedAt = Date.now();
  window.addEventListener("message", (event) => {
    const d = event.data;
    if (!d || d.postmate !== "emit" || !d.value || typeof d.value.name !== "string") return;
    // The editor frame is served from *.jupyter-proxy.kaggle.net.
    if (!/^https:\/\/[a-z0-9.-]+\.kaggle\.net$/.test(event.origin)) return;
    const m = /^kernel_(busy|idle|restarting|disconnected|ready)\.Kernel$/.exec(d.value.name);
    if (!m) return;
    kernelState = m[1] === "ready" ? "idle" : m[1];
    kernelChangedAt = Date.now();
  });

  // -- cell helpers ----------------------------------------------------------------

  const uuid = () =>
    (crypto.randomUUID && crypto.randomUUID()) ||
    "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      return (c === "x" ? r : (r & 3) | 8).toString(16);
    });

  const sourceOf = (cell) => (Array.isArray(cell.source) ? cell.source.join("") : cell.source || "");

  function indexOf(nb, id) {
    const i = nb.cells.findIndex((c) => c.id === id);
    if (i < 0) throw new Error(`Unknown cell id ${id}`);
    return i;
  }

  // {token, id, resolved, done, error, retried}. The run command resolving does not
  // mean the cell ran: with the session off, Kaggle starts one and queues the cell,
  // and JupyterLab resolves at once. A run is done when the cell has an execution
  // count (JupyterLab clears it when a run starts or is skipped).
  let currentRun = null;

  function busy() {
    return kernelState === "busy" || (currentRun && !currentRun.done);
  }

  function updateRun(run, nb) {
    const cell = nb.cells.find((c) => c.id === run.id);
    if (!cell) run.done = run.deleted = true;
    else if (run.resolved && cell.execution_count != null) run.done = true;
    return cell || null;
  }

  // Caller holds the queue and has selected the cell.
  async function startRun(run) {
    const c = await getClient();
    run.resolved = false;
    c.frameChannel.callAsync("executeActiveCell", null).then(
      () => (run.resolved = true),
      (e) => {
        run.resolved = run.done = true;
        run.error = String(e && e.message ? e.message : e);
      }
    );
  }

  // The kernel's idle event can land just after a run's promise resolves, and
  // completions or widgets make it busy for a moment too, so wait briefly.
  async function ensureIdle(ms = 2000) {
    const deadline = Date.now() + ms;
    for (;;) {
      if (currentRun && !currentRun.done) updateRun(currentRun, await readNotebook());
      if (!busy()) return;
      if (Date.now() > deadline) throw new Error(BUSY);
      await sleep(100);
    }
  }

  // nbformat 4.4 notebooks carry no cell ids; one rewrite as 4.5 gives every
  // cell a stable id, and JupyterLab keeps ids for cells added later.
  async function readWithIds() {
    const nb = await readNotebook();
    const missing = nb.nbformat_minor < 5 || nb.cells.some((c) => !c.id);
    if (!missing) return nb;
    if (busy()) return nb;
    const seen = new Set();
    nb.nbformat_minor = 5;
    for (const cell of nb.cells) {
      if (!cell.id || seen.has(cell.id)) cell.id = uuid();
      seen.add(cell.id);
    }
    await writeNotebook(nb);
    return readNotebook();
  }

  async function selectIndex(index, count) {
    await command("notebook:enter-command-mode");
    for (let i = 0; i <= count; i++) await command("notebook:move-cursor-up");
    for (let i = 0; i < index; i++) await command("notebook:move-cursor-down");
  }

  function newCell(cellType, source) {
    const cell = { cell_type: cellType, id: uuid(), metadata: {}, source };
    if (cellType === "code") {
      cell.execution_count = null;
      cell.outputs = [];
    }
    return cell;
  }

  // -- methods the server can call ---------------------------------------------------

  function pageRef() {
    const m = EDIT_PATH.exec(location.pathname);
    return m ? `${m[1]}/${m[2]}` : null;
  }

  const methods = {
    // Whether the editor frame is up; does not wait for it. Off the editor it says
    // where the tab is, so a signed-out browser can be told apart from a slow load.
    async ready() {
      const base = {
        ref: pageRef(),
        path: location.pathname,
        loaded: document.readyState === "complete",
        notFound: /can.t find that page/i.test(document.body ? document.body.innerText : ""),
      };
      if (!base.ref) return { ...base, ready: false, reason: "not an editor page" };
      try {
        if (!client || !client.frameChannel) client = findClient();
      } catch (e) {
        client = null;
      }
      if (!client || !client.frameChannel) return { ...base, ready: false, reason: "editor loading" };
      return { ...base, ready: true, title: document.title };
    },

    async status() {
      const nb = await readNotebook();
      return {
        kernel: kernelState,
        cell_count: nb.cells.length,
        running: currentRun && !currentRun.done ? currentRun.id : null,
        ref: pageRef(),
      };
    },

    async getNotebook() {
      return serial(readWithIds);
    },

    async insertCell({ index, cell_type = "code", source = "" }) {
      return serial(async () => {
        const nb = await readWithIds();
        const n = nb.cells.length;
        const at = index == null ? n : Math.max(0, Math.min(n, index));
        if (!busy()) {
          const cell = newCell(cell_type, source);
          nb.cells.splice(at, 0, cell);
          await writeNotebook(nb);
          return { id: cell.id, index: at };
        }
        // A cell is running: insert next to it without rewriting the notebook.
        if (cell_type !== "code" || at === 0) throw new Error(BUSY);
        await selectIndex(at - 1, n);
        await frame("insertCells", [{ content: source }]);
        await sleep(100);
        const after = await readNotebook();
        const cell = after.cells[at];
        if (!cell || sourceOf(cell) !== source) {
          throw new Error("The editor changed while inserting; re-read the notebook.");
        }
        return { id: cell.id, index: at };
      });
    },

    async updateCell({ id, source }) {
      return serial(async () => {
        await ensureIdle();
        const nb = await readWithIds();
        const cell = nb.cells[indexOf(nb, id)];
        cell.source = source;
        await writeNotebook(nb);
        return { id };
      });
    },

    async moveCell({ id, index }) {
      return serial(async () => {
        await ensureIdle();
        const nb = await readWithIds();
        const [cell] = nb.cells.splice(indexOf(nb, id), 1);
        const at = Math.max(0, Math.min(nb.cells.length, index));
        nb.cells.splice(at, 0, cell);
        await writeNotebook(nb);
        return { id, index: at };
      });
    },

    async deleteCell({ id }) {
      return serial(async () => {
        await ensureIdle();
        const nb = await readWithIds();
        nb.cells.splice(indexOf(nb, id), 1);
        await writeNotebook(nb);
        return { id };
      });
    },

    // Selects the cell and starts it; poll runState for the result.
    async runCell({ id }) {
      return serial(async () => {
        if (currentRun && !currentRun.done) {
          throw new Error(`Cell ${currentRun.id} is still running in the editor.`);
        }
        const nb = await readWithIds();
        const index = indexOf(nb, id);
        if (nb.cells[index].cell_type !== "code") throw new Error(`Cell ${id} is not a code cell`);
        await selectIndex(index, nb.cells.length);
        const run = { token: uuid(), id, resolved: false, done: false, error: null, retried: false };
        currentRun = run;
        await startRun(run);
        return { token: run.token, index };
      });
    },

    async runState({ token, id }) {
      return serial(async () => {
        const nb = await readNotebook();
        const run = currentRun && currentRun.token === token ? currentRun : null;
        const cell = run ? updateRun(run, nb) : nb.cells.find((c) => c.id === id) || null;
        // The session came up but the queued run was dropped: run it again, once.
        if (run && !run.done && run.resolved && !run.retried && kernelState === "idle" &&
            Date.now() - kernelChangedAt > 3000) {
          run.retried = true;
          await selectIndex(nb.cells.indexOf(cell), nb.cells.length);
          await startRun(run);
        }
        return {
          // An unknown token means the page reloaded: fall back to the cell itself.
          done: run ? run.done : !!(cell && cell.execution_count != null && kernelState !== "busy"),
          // Waiting for the session to start (the run command returned, the cell has not run).
          queued: !!(run && run.resolved && !run.done),
          error: run ? run.error : null,
          deleted: !cell,
          execution_count: cell ? cell.execution_count : null,
          outputs: cell ? cell.outputs || [] : [],
          kernel: kernelState,
        };
      });
    },

    async interrupt() {
      return serial(() => frame("stopExecution"));
    },

    async restart() {
      return serial(() => frame("restartKernel"));
    },
  };

  async function call(method, params) {
    if (!Object.prototype.hasOwnProperty.call(methods, method)) {
      throw new Error(`Unknown method ${method}`);
    }
    return (await methods[method](params || {})) ?? null;
  }

  window.__kaggleMcpBridge = { version: VERSION, call };
})();
