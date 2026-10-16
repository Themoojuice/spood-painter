/* spoodpaint web app: screens, photo intake (picker + Android share sheet), eye taps, results. */
'use strict';
const VERSION = '0.5.0';
const MAX_SIDE = 4000;            // photos larger than this are scaled down before processing (12 MP passes untouched)
const $ = (id) => document.getElementById(id);
const store = {
  get(k, d) { try { const v = localStorage.getItem('sp.' + k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem('sp.' + k, JSON.stringify(v)); } catch (e) { } },
};

// ---------------------------------------------------------------- service worker + isolation
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('sw.js').then(() => navigator.serviceWorker.ready).then(() => {
    // multi-threaded nets need cross-origin isolation, which the service worker provides from the 2nd load on
    if (!self.crossOriginIsolated && !sessionStorage.getItem('sp.reloaded') && !location.search.includes('shared=1')) {
      sessionStorage.setItem('sp.reloaded', '1'); location.reload();
    }
  }).catch(() => { });
}

// ---------------------------------------------------------------- state
const S = {
  ready: false, info: null, photo: null, name: 'photo', bitmap: null, w: 0, h: 0,
  eyes: [], cutDone: false, cutErr: null, painting: false, resultUrl: null, resultBlob: null,
  size: store.get('size', 2000), vib: store.get('vib', 1.25), seed: 7, t0: 0, timer: null,
};
let wakeLock = null;
const worker = new Worker('worker.js');
worker.postMessage({ type: 'init', ep: new URLSearchParams(location.search).get('ep') || store.get('engine', 'wasm') });

function show(id) {
  for (const s of document.querySelectorAll('.screen')) s.classList.toggle('on', s.id === id);
  if (id === 'eyes') requestAnimationFrame(() => { fitCanvas(); resetView(); draw(); });
}
function toast(t) { const el = $('toast'); el.textContent = t; el.classList.add('on'); clearTimeout(toast.t); toast.t = setTimeout(() => el.classList.remove('on'), 2400); }
function fmt(s) { s = Math.round(s); return Math.floor(s / 60) + ':' + String(s % 60).padStart(2, '0'); }
async function keepAwake(on) {
  try {
    if (on && 'wakeLock' in navigator && !wakeLock) wakeLock = await navigator.wakeLock.request('screen');
    if (!on && wakeLock) { await wakeLock.release(); wakeLock = null; }
  } catch (e) { }
}

// ---------------------------------------------------------------- settings
function syncSettings() {
  for (const b of $('sizeSeg').children) b.setAttribute('aria-pressed', String(+b.dataset.v === S.size));
  for (const b of $('engSeg').children) b.setAttribute('aria-pressed', String(b.dataset.v === store.get('engine', 'wasm')));
  $('vib').value = S.vib;
  $('vibLabel').textContent = S.vib.toFixed(2) + '×';
}
$('engSeg').onclick = (e) => {
  const v = e.target.dataset.v; if (!v || v === store.get('engine', 'wasm')) return;
  store.set('engine', v); toast(v === 'webgpu' ? 'Trying the graphics chip: restarting…' : 'Back to CPU: restarting…');
  setTimeout(() => location.reload(), 900);
};
$('sizeSeg').onclick = (e) => { const v = +e.target.dataset.v; if (v) { S.size = v; store.set('size', v); syncSettings(); } };
$('vib').oninput = () => { S.vib = +$('vib').value; store.set('vib', S.vib); syncSettings(); };
syncSettings();

// ---------------------------------------------------------------- worker messages
function engine(text, pct, state) {
  $('engText').textContent = text;
  $('engDot').className = 'dot ' + (state || 'busy');
  $('engBarBox').style.visibility = pct >= 0 && pct < 100 && state !== 'ok' ? 'visible' : 'hidden';
  if (pct >= 0) $('engBar').style.width = pct + '%';
}
worker.onmessage = (ev) => {
  const m = ev.data;
  if (m.type === 'status') {
    if (!S.ready) engine(m.text, m.pct);
    else if (S.painting) { $('workText').textContent = m.text; if (m.pct >= 0) $('workBar').style.width = m.pct + '%'; }
    else if (!S.cutDone) { $('cutText').textContent = m.text + (m.pct >= 0 ? ` (${Math.round(m.pct)}%)` : ''); }
  } else if (m.type === 'ready') {
    S.ready = true; S.info = m.info;
    engine(`Ready · runs on this phone (${m.info.provider === 'webgpu' ? 'graphics chip' : m.info.threads + ' cores'})`, 100, 'ok');
    $('foot').textContent = `spoodpaint web v${VERSION} · ${m.info.provider}${m.info.isolated ? '' : ' · single-thread'} · started in ${m.info.secs.toFixed(0)} s`;
    if (!S.cutDone && S.photo) $('cutText').textContent = 'Cutting out…';
  } else if (m.type === 'cutout-done') {
    S.cutDone = true; $('cutDot').className = 'dot ok'; $('cutText').textContent = `Cutout ready (${fmt(m.secs)})`;
    if (S.painting) $('workText').textContent = 'Painting: starting';
  } else if (m.type === 'result') {
    finishPaint(m);
  } else if (m.type === 'error') {
    S.painting = false; keepAwake(false); clearInterval(S.timer);
    if (m.stage === 'starting up') engine('Could not start: ' + m.message, -1, 'bad');
    if (m.stage === 'cutting out the spider') { S.cutErr = m; $('cutDot').className = 'dot bad'; $('cutText').textContent = 'Cutout failed'; }
    showError(m);
  } else if (m.type === 'log') {
    console.log('[py]', m.line);
  }
};
worker.onerror = (e) => showError({ stage: 'running the engine', message: e.message || 'worker crashed (possibly out of memory)', detail: String(e.filename || '') + ':' + (e.lineno || '') });

function showError(m) {
  let title = 'Something went wrong while ' + m.stage, msg = m.message;
  if (/NO_SPIDER/.test(m.detail || '')) { title = 'No spider found'; msg = 'The cutout couldn’t find a spider in this photo. Try a closer, sharper shot.'; }
  else if (/memory|RangeError|allocation/i.test(m.detail || m.message || '')) msg += '\n\nLikely out of memory: close other apps or choose "Quick" size.';
  if (S.info && S.info.provider === 'webgpu' && /cutting|engine|starting/.test(m.stage)) msg += '\n\nYou\u2019re on the GPU (beta) engine: switch Engine back to CPU on the start screen and try again.';
  $('errTitle').textContent = title; $('errMsg').textContent = msg;
  S.lastError = [`spoodpaint web v${VERSION}`, `when: ${new Date().toISOString()}`, `stage: ${m.stage}`, `photo: ${S.name} ${S.w}x${S.h}`,
    `engine: ${JSON.stringify(S.info)}`, `browser: ${navigator.userAgent}`, '', m.detail || m.message].join('\n');
  $('errDetail').textContent = S.lastError;
  show('error');
}
$('errCopy').onclick = async () => { try { await navigator.clipboard.writeText(S.lastError); toast('Error report copied'); } catch (e) { toast('Couldn’t copy - screenshot it instead'); } };
$('errBack').onclick = () => { show('home'); };

// ---------------------------------------------------------------- photo intake
$('pick').onclick = () => $('file').click();
$('file').onchange = () => { const f = $('file').files[0]; $('file').value = ''; if (f) loadPhoto(f, f.name); };

async function takeShared() {
  if (!location.search.includes('shared=1')) return;
  history.replaceState(null, '', location.pathname);
  try {
    const c = await caches.open('spoodpaint-shared');
    const r = await c.match('shared-photo');
    if (!r) { toast('Nothing was shared'); return; }
    const name = decodeURIComponent(r.headers.get('x-name') || 'photo.jpg');
    const blob = await r.blob(); await c.delete('shared-photo');
    loadPhoto(blob, name);
  } catch (e) { showError({ stage: 'receiving the shared photo', message: String(e.message || e) }); }
}

async function loadPhoto(blob, name) {
  S.name = (name || 'photo').replace(/\.[^.]+$/, '');
  let bmp;
  try { bmp = await createImageBitmap(blob, { imageOrientation: 'from-image' }); }
  catch (e) {
    const heic = /heic|heif/i.test(blob.type + name);
    showError({ stage: 'opening the photo', message: heic ? 'HEIC photos aren’t supported yet. Set the camera to JPEG, or share a JPEG copy.' : 'This file couldn’t be opened as a picture.', detail: String(e) });
    return;
  }
  const sc = Math.min(1, MAX_SIDE / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * sc), h = Math.round(bmp.height * sc);
  const c = new OffscreenCanvas(w, h), g = c.getContext('2d', { willReadFrequently: true });
  g.imageSmoothingQuality = 'high'; g.drawImage(bmp, 0, 0, w, h);
  const rgba = g.getImageData(0, 0, w, h).data.buffer;
  // a lighter copy for the screen
  const ds = Math.min(1, 2048 / Math.max(w, h));
  S.bitmap = ds < 1 ? await createImageBitmap(c, { resizeWidth: Math.round(w * ds), resizeHeight: Math.round(h * ds), resizeQuality: 'high' }) : await createImageBitmap(c);
  bmp.close && bmp.close();
  S.w = w; S.h = h; S.eyes = []; S.cutDone = false; S.cutErr = null; S.photo = true;
  $('cutDot').className = 'dot busy';
  $('cutText').textContent = S.ready ? 'Cutting out…' : 'Starting the engine first (one-time downloads)…';
  worker.postMessage({ type: 'photo', rgba, w, h }, [rgba]);
  const t = c.convertToBlob({ type: 'image/jpeg', quality: .7 }).then((b) => { $('workThumb').src = URL.createObjectURL(b); });
  show('eyes'); setTimeout(() => { $('hint').style.opacity = 0; }, 5000); $('hint').style.opacity = 1;
}

// ---------------------------------------------------------------- eye taps: pan / pinch / tap
const cv = $('cv'), ctx = cv.getContext('2d');
let view = { s: 1, x: 0, y: 0 }, dpr = 1;     // screen = (photo * s) + (x, y), in CSS px
function fitCanvas() { const r = $('stage').getBoundingClientRect(); dpr = window.devicePixelRatio || 1; cv.width = r.width * dpr; cv.height = r.height * dpr; }
function resetView() {
  if (!S.photo) return;
  const r = $('stage').getBoundingClientRect(); const s = Math.min(r.width / S.w, r.height / S.h);
  view = { s, x: (r.width - S.w * s) / 2, y: (r.height - S.h * s) / 2, min: s };
}
function draw() {
  if (!S.bitmap) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.fillStyle = '#111'; ctx.fillRect(0, 0, cv.width, cv.height);
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(S.bitmap, view.x, view.y, S.w * view.s, S.h * view.s);
  S.eyes.forEach(([px, py], i) => {
    const x = px * view.s + view.x, y = py * view.s + view.y;
    ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(0,0,0,.8)'; ctx.beginPath(); ctx.arc(x, y, 13, 0, 7); ctx.stroke();
    ctx.lineWidth = 2; ctx.strokeStyle = '#ffd34d'; ctx.beginPath(); ctx.arc(x, y, 13, 0, 7); ctx.stroke();
    ctx.fillStyle = '#ffd34d'; ctx.beginPath(); ctx.arc(x, y, 2.2, 0, 7); ctx.fill();
    ctx.font = '600 13px system-ui'; ctx.fillText(String(i + 1), x + 16, y - 10);
  });
}
const ptrs = new Map(); let gesture = null;
function clampView() {
  const r = $('stage').getBoundingClientRect();
  view.s = Math.max(view.min, Math.min(view.min * 14, view.s));
  const W = S.w * view.s, H = S.h * view.s;
  view.x = W <= r.width ? (r.width - W) / 2 : Math.min(0, Math.max(r.width - W, view.x));
  view.y = H <= r.height ? (r.height - H) / 2 : Math.min(0, Math.max(r.height - H, view.y));
}
function local(e) { const r = cv.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; }
cv.addEventListener('pointerdown', (e) => {
  cv.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, local(e));
  if (ptrs.size === 1) gesture = { tap: true, t: performance.now(), start: local(e), view: { ...view } };
  else if (ptrs.size === 2) {
    const [a, b] = [...ptrs.values()];
    gesture = { tap: false, pinch: true, d0: Math.hypot(a[0] - b[0], a[1] - b[1]), m0: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2], view: { ...view } };
  }
});
cv.addEventListener('pointermove', (e) => {
  if (!ptrs.has(e.pointerId) || !gesture) return;
  ptrs.set(e.pointerId, local(e));
  if (gesture.pinch && ptrs.size >= 2) {
    const [a, b] = [...ptrs.values()];
    const d = Math.hypot(a[0] - b[0], a[1] - b[1]), m = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const v0 = gesture.view, s = v0.s * d / gesture.d0;
    const px = (gesture.m0[0] - v0.x) / v0.s, py = (gesture.m0[1] - v0.y) / v0.s;
    view.s = s; view.x = m[0] - px * s; view.y = m[1] - py * s; view.min = v0.min; clampView(); draw();
  } else if (ptrs.size === 1) {
    const p = local(e), dx = p[0] - gesture.start[0], dy = p[1] - gesture.start[1];
    if (Math.hypot(dx, dy) > 10) gesture.tap = false;
    if (!gesture.tap) { view.x = gesture.view.x + dx; view.y = gesture.view.y + dy; clampView(); draw(); }
  }
});
function endPtr(e) {
  if (!ptrs.has(e.pointerId)) return;
  const p = local(e); ptrs.delete(e.pointerId);
  if (gesture && gesture.tap && ptrs.size === 0 && performance.now() - gesture.t < 600) tapAt(p);
  if (ptrs.size === 0) gesture = null;
  else if (ptrs.size === 1 && gesture && gesture.pinch) { const q = [...ptrs.values()][0]; gesture = { tap: false, start: q, view: { ...view } }; }
}
cv.addEventListener('pointerup', endPtr); cv.addEventListener('pointercancel', endPtr);
cv.addEventListener('wheel', (e) => {   // desktop testing: wheel zoom
  e.preventDefault(); const [mx, my] = local(e); const px = (mx - view.x) / view.s, py = (my - view.y) / view.s;
  view.s *= e.deltaY < 0 ? 1.2 : 1 / 1.2; clampView(); view.x = mx - px * view.s; view.y = my - py * view.s; clampView(); draw();
}, { passive: false });
function tapAt([x, y]) {
  const hit = S.eyes.findIndex(([px, py]) => Math.hypot(px * view.s + view.x - x, py * view.s + view.y - y) < 22);
  if (hit >= 0) S.eyes.splice(hit, 1);
  else {
    const px = (x - view.x) / view.s, py = (y - view.y) / view.s;
    if (px < 0 || py < 0 || px > S.w || py > S.h) return;
    S.eyes.push([px, py]);
    if (navigator.vibrate) navigator.vibrate(8);
  }
  $('hint').style.opacity = 0; draw();
}
window.addEventListener('resize', () => { if ($('eyes').classList.contains('on')) { fitCanvas(); clampView(); draw(); } });
$('undo').onclick = () => { S.eyes.pop(); draw(); };
$('back').onclick = () => show('home');

// ---------------------------------------------------------------- paint
function startPaint(newSeed) {
  if (S.cutErr) { showError(S.cutErr); return; }
  if (newSeed) S.seed = 1 + Math.floor(Math.random() * 9999);
  S.painting = true; S.t0 = performance.now();
  $('workBar').style.width = '0%';
  $('workText').textContent = !S.ready ? 'Starting the engine (one-time downloads)…' : S.cutDone ? 'Painting: starting' : 'Finishing the cutout…';
  clearInterval(S.timer); S.timer = setInterval(() => { $('timer').textContent = fmt((performance.now() - S.t0) / 1000); }, 500);
  $('timer').textContent = '0:00';
  keepAwake(true); show('work');
  if (S.eyes.length === 1) toast('Tip: one painted eye can look odd – tap both front eyes');
  worker.postMessage({ type: 'paint', eyes: S.eyes, size: S.size, vibrance: S.vib, seed: S.seed });
}
$('paint').onclick = () => startPaint(false);
function finishPaint(m) {
  S.painting = false; clearInterval(S.timer); keepAwake(false);
  const total = (performance.now() - S.t0) / 1000;
  if (S.resultUrl) URL.revokeObjectURL(S.resultUrl);
  S.resultBlob = new Blob([m.jpeg], { type: 'image/jpeg' });
  S.resultUrl = URL.createObjectURL(S.resultBlob);
  $('resultImg').src = S.resultUrl;
  $('resultNote').textContent = `Painted on this phone in ${fmt(total)} · ${S.eyes.length ? S.eyes.length + ' eyes tapped' : 'automatic eyes'} · colour ${S.vib.toFixed(2)}×`;
  show('result');
  if (document.hidden && 'Notification' in window && Notification.permission === 'granted') new Notification('spoodpaint', { body: 'Your painting is ready' });
}
const outName = () => `spoodpaint_${S.name}_${S.seed}.jpg`;
$('save').onclick = () => {
  const a = document.createElement('a'); a.href = S.resultUrl; a.download = outName(); document.body.appendChild(a); a.click(); a.remove();
  toast('Saved to Downloads');
};
$('share').onclick = async () => {
  const f = new File([S.resultBlob], outName(), { type: 'image/jpeg' });
  if (navigator.canShare && navigator.canShare({ files: [f] })) { try { await navigator.share({ files: [f], title: 'spoodpaint' }); } catch (e) { } }
  else toast('Sharing isn’t available here - use Save');
};
$('repaint').onclick = () => startPaint(true);
$('again').onclick = () => { show('home'); $('file').click(); };

window.spDebug = { get view() { return view; }, S };   // used by automated tests
takeShared();
