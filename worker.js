/* spoodpaint worker: Python (Pyodide) runs the pipeline; the two cutout networks run in a separate
   net worker (networker.js) that exists only while a cutout runs, so its memory is returned
   before the painting starts.
   In:  {type:'init', ep} | {type:'photo', rgba, w, h} | {type:'paint', eyes, size, vibrance, seed}
   Out: {type:'status', text, pct} | {type:'ready', info} | {type:'cutout-done', secs}
        {type:'result', jpeg, secs} | {type:'error', stage, message, detail} | {type:'log', line} */
const VERSION = '0.5.1';
importScripts('pyodide/pyodide.js');

const PY_FILES = ['spoodpaint.py', 'eyes.py', 'limbs.py', 'cutout.py', 'web_pipeline.py', 'eye_model.json'];
let py = null, queue = Promise.resolve(), ready = null, opts = {}, info = {};
const say = (text, pct = -1) => postMessage({ type: 'status', text, pct });
self.postStatus = say;            // called from Python

// ---------------------------------------------------------------- net worker lifecycle
let net = null, netReady = null, nextId = 1;
const pending = new Map();
function threads() { const c = navigator.hardwareConcurrency || 4; return self.crossOriginIsolated ? Math.max(1, Math.min(4, c)) : 1; }
function startNet() {
  if (netReady) return netReady;
  net = new Worker('networker.js');
  netReady = new Promise((resolve, reject) => {
    net.onmessage = (ev) => {
      const m = ev.data;
      if (m.type === 'status') say(m.text, m.pct);
      else if (m.type === 'ready') { info.provider = m.provider; resolve(m.provider); }
      else if (m.type === 'out') { pending.get(m.id).resolve(new Float32Array(m.data)); pending.delete(m.id); }
      else if (m.type === 'error') {
        if (m.id && pending.has(m.id)) { pending.get(m.id).reject(new Error(m.message)); pending.delete(m.id); }
        else reject(new Error(m.message));
      }
    };
    net.onerror = (e) => {
      const err = new Error('the cutout engine crashed (' + (e.message || 'probably out of memory') + ')');
      reject(err); for (const p of pending.values()) p.reject(err); pending.clear(); stopNet();
    };
  });
  net.postMessage({ type: 'init', ep: opts.ep || '', threads: threads() });
  return netReady;
}
function stopNet() { if (net) net.terminate(); net = null; netReady = null; }

// called from Python: run a net on a float32 NCHW tensor, return the first output as Float32Array
self.ortRun = async (name, bytesProxy, shapeProxy) => {
  const u8 = bytesProxy instanceof Uint8Array ? bytesProxy : bytesProxy.toJs();
  const shape = Array.isArray(shapeProxy) ? shapeProxy : shapeProxy.toJs();
  const ab = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
  await startNet();
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    net.postMessage({ type: 'run', id, name, data: ab, shape }, [ab]);
  });
};

// ---------------------------------------------------------------- start-up
async function init() {
  const t0 = performance.now();
  say('Starting Python…', 3);
  py = await loadPyodide({ indexURL: new URL('pyodide/', self.location.href).href, stdout: onLog, stderr: onLog });
  say('Loading image libraries (first time ~50 MB)…', 10);
  await py.loadPackage(['numpy', 'scipy', 'scikit-image', 'opencv-python'], { messageCallback: () => { } });
  py.FS.mkdirTree('/sp');
  for (const f of PY_FILES) {
    const r = await fetch('py/' + f + '?v=' + VERSION); if (!r.ok) throw new Error('missing py/' + f);
    py.FS.writeFile('/sp/' + f, new Uint8Array(await r.arrayBuffer()));
  }
  await py.runPythonAsync("import sys; sys.path.insert(0, '/sp'); import web_pipeline");
  await startNet();                // downloads + checks the models now, so the first photo doesn't wait
  stopNet();                       // free its memory until a photo arrives
  Object.assign(info, { secs: (performance.now() - t0) / 1000, threads: threads(), isolated: self.crossOriginIsolated, version: VERSION });
  return info;
}

function onLog(line) {
  postMessage({ type: 'log', line });
  try { py && py.globals.get('web_pipeline')._progress_from_log(line); } catch (e) { }
}

function fail(stage, e) {
  const msg = String(e && e.message || e);
  postMessage({ type: 'error', stage, message: msg.split('\n').filter(Boolean).slice(-1)[0] || msg, detail: msg });
}

onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'init') {
    opts = m;
    ready = init().then((i) => { postMessage({ type: 'ready', info: i }); return true; },
      (e) => { fail('starting up', e); return false; });
    return;
  }
  queue = queue.then(async () => {
    if (!(await ready)) return;
    const wp = py.globals.get('web_pipeline');
    if (m.type === 'photo') {
      try {
        const secs = await wp.set_photo(new Uint8Array(m.rgba), m.w, m.h);
        postMessage({ type: 'cutout-done', secs, provider: info.provider });
      } catch (e) { fail('cutting out the spider', e); }
      finally { stopNet(); }
    } else if (m.type === 'paint') {
      try {
        const r = await wp.paint(JSON.stringify(m.eyes || []), m.size, m.vibrance, m.seed, m.tone, m.warmth);
        const [jpeg, secs] = r.toJs ? r.toJs() : r;
        const ab = jpeg.buffer.slice(jpeg.byteOffset, jpeg.byteOffset + jpeg.byteLength);
        postMessage({ type: 'result', jpeg: ab, secs }, [ab]);
        r.destroy && r.destroy();
      } catch (e) { fail('painting', e); }
    }
  });
};
