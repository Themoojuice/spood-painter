/* spoodpaint service worker
   1. Android share sheet: receives POST ./share-target (multipart, field "photo"), parks the photo,
      then redirects to ./?shared=1 where the app picks it up.
   2. Offline: app shell, Python runtime and packages are cached on first use (models are cached by the worker).
   3. Cross-origin isolation headers (COOP/COEP) on our own responses, which unlocks multi-threaded
      WebAssembly for the neural nets. GitHub Pages can't set headers, so we add them here. */
const VERSION = '0.5.2';
const SHELL = 'spoodpaint-shell-' + VERSION;
const RUNTIME = 'spoodpaint-runtime-v1';      // pyodide + ort files: big, versioned by path, keep across app updates
const SHARED = 'spoodpaint-shared';
const SHELL_FILES = ['./', 'index.html', 'app.js', 'worker.js', 'networker.js', 'manifest.webmanifest', 'icon-192.png', 'icon-512.png',
  'py/spoodpaint.py', 'py/eyes.py', 'py/limbs.py', 'py/cutout.py', 'py/web_pipeline.py', 'py/eye_model.json'];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(SHELL).then((c) => c.addAll(SHELL_FILES)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (e) => {
  e.waitUntil((async () => {
    for (const k of await caches.keys())
      if (k.startsWith('spoodpaint-shell-') && k !== SHELL) await caches.delete(k);
    await self.clients.claim();
  })());
});

function isolate(resp) {
  if (!resp || resp.status === 0 || resp.type === 'opaqueredirect') return resp;
  const h = new Headers(resp.headers);
  h.set('Cross-Origin-Opener-Policy', 'same-origin');
  h.set('Cross-Origin-Embedder-Policy', 'require-corp');
  h.set('Cross-Origin-Resource-Policy', 'same-origin');
  return new Response(resp.body, { status: resp.status, statusText: resp.statusText, headers: h });
}

self.addEventListener('fetch', (e) => {
  const url = new URL(e.request.url);
  if (url.origin !== self.location.origin) return;

  if (e.request.method === 'POST' && url.pathname.endsWith('/share-target')) {
    e.respondWith((async () => {
      try {
        const form = await e.request.formData();
        const file = form.getAll('photo').find((f) => f && f.size) || null;
        if (file) {
          const c = await caches.open(SHARED);
          await c.put('shared-photo', new Response(file, { headers: { 'content-type': file.type || 'image/jpeg', 'x-name': encodeURIComponent(file.name || 'photo.jpg') } }));
        }
      } catch (err) { /* fall through: app opens normally */ }
      return Response.redirect('./?shared=1', 303);
    })());
    return;
  }
  if (e.request.method !== 'GET') return;
  if (url.pathname.includes('/models/')) return;          // worker caches these itself (with progress)

  const big = url.pathname.includes('/pyodide/') || url.pathname.includes('/ort/');
  e.respondWith((async () => {
    const cacheName = big ? RUNTIME : SHELL;
    const c = await caches.open(cacheName);
    const key = big ? e.request : url.pathname.endsWith('/') || url.search.startsWith('?shared') ? new Request('./', { headers: e.request.headers }) : new Request(url.origin + url.pathname);
    if (big) {
      const hit = await c.match(key);
      if (hit) return isolate(hit);
      const r = await fetch(e.request);
      if (r.ok) c.put(key, r.clone());
      return isolate(r);
    }
    // app shell: network first (so updates arrive), cache as fallback (offline)
    try {
      const r = await fetch(e.request, { cache: 'no-cache' });
      if (r.ok) c.put(key, r.clone());
      return isolate(r);
    } catch (err) {
      const hit = await c.match(key, { ignoreSearch: true });
      if (hit) return isolate(hit);
      throw err;
    }
  })());
});
