/* spoodpaint net worker: hosts onnxruntime-web and the two cutout networks.
   It is created for a cutout and terminated straight after, because WebAssembly memory never
   shrinks: ending the worker is the only way to hand its ~3 GB back before the painting starts.
   In:  {type:'init', ep, threads} | {type:'run', id, name, data(ArrayBuffer f32), shape}
   Out: {type:'status', text, pct} | {type:'ready', provider} | {type:'out', id, data} | {type:'error', id?, message} */
importScripts('ort/ort.webgpu.min.js');

const MODELS = {
  isnet: { url: 'models/isnet_w16.onnx', mb: 91, label: 'outline' },
  birefnet: { url: 'models/birefnet_lite_1120_dc_w16.onnx', mb: 92, label: 'edge' },
};
const MODEL_CACHE = 'spoodpaint-models-v2';
const sessions = {};
let provider = 'wasm';
const say = (text, pct = -1) => postMessage({ type: 'status', text, pct });

async function fetchModel(name) {
  const { url, mb, label } = MODELS[name];
  let cache = null;
  try { cache = await caches.open(MODEL_CACHE); } catch (e) { }
  const hit = cache && await cache.match(url);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const r = await fetch(url);
  if (!r.ok) throw new Error(`download ${url}: HTTP ${r.status}`);
  const total = +r.headers.get('content-length') || mb * 1e6;
  const reader = r.body.getReader(); const parts = []; let got = 0, last = 0;
  for (;;) {
    const { done, value } = await reader.read(); if (done) break;
    parts.push(value); got += value.length;
    if (got - last > 2e6) { last = got; say(`Downloading the ${label} model (one time): ${(got / 1e6).toFixed(0)} of ${(total / 1e6).toFixed(0)} MB`, 100 * got / total); }
  }
  const buf = new Uint8Array(got); let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; }
  if (cache) try { await cache.put(url, new Response(buf, { headers: { 'content-type': 'application/octet-stream' } })); } catch (e) { }
  return buf;
}

async function makeSession(bytes) {
  const opts = { graphOptimizationLevel: 'all', enableCpuMemArena: false, enableMemPattern: false };
  if (provider === 'webgpu') {
    try { return await ort.InferenceSession.create(bytes, { ...opts, executionProviders: ['webgpu', 'wasm'] }); }
    catch (e) { provider = 'wasm'; }
  }
  return ort.InferenceSession.create(bytes, { ...opts, executionProviders: ['wasm'] });
}

async function init(m) {
  ort.env.wasm.wasmPaths = new URL('ort/', self.location.href).href;
  ort.env.wasm.numThreads = m.threads;
  if (self.navigator.gpu && m.ep !== 'wasm') {
    try { if (await navigator.gpu.requestAdapter()) provider = 'webgpu'; } catch (e) { }
  }
  for (const name of ['isnet', 'birefnet']) {
    const bytes = await fetchModel(name);
    say(`Preparing the ${MODELS[name].label} model…`, 95);
    sessions[name] = await makeSession(bytes);
  }
  return provider;
}

onmessage = async (ev) => {
  const m = ev.data;
  if (m.type === 'init') {
    try { postMessage({ type: 'ready', provider: await init(m) }); }
    catch (e) { postMessage({ type: 'error', message: 'loading the cutout models: ' + (e.message || e) }); }
  } else if (m.type === 'run') {
    try {
      const s = sessions[m.name];
      const out = await s.run({ [s.inputNames[0]]: new ort.Tensor('float32', new Float32Array(m.data), m.shape) });
      const t = out[s.outputNames[0]];
      const d = t.location === 'gpu-buffer' ? await t.getData() : t.data;
      const ab = d.buffer.slice(d.byteOffset, d.byteOffset + d.byteLength);
      postMessage({ type: 'out', id: m.id, data: ab }, [ab]);
    } catch (e) {
      postMessage({ type: 'error', id: m.id, message: `running the ${m.name} network: ` + (e.message || e) });
    }
  }
};
