"""spoodpaint in the browser: the same cutout + painting code as the laptop version.
Neural nets run in onnxruntime-web (JS, `ortRun`); everything else is the unchanged Python.

Called from worker.js:
    await set_photo(rgba_bytes, w, h)      -> runs the cutout, keeps the mask
    await paint(eyes_json, size, vibrance, seed) -> JPEG bytes
"""
import gc, json, time
import numpy as np
import cv2
from js import ortRun, postStatus
from pyodide.ffi import to_js

import cutout as co          # fuse() and prune_spurs() reused verbatim (torch is only imported lazily there)
import spoodpaint as sp

MEAN, STD = np.float32([0.485, 0.456, 0.406]), np.float32([0.229, 0.224, 0.225])
BIREF = 1120                 # BiRefNet input (square; training also resized to a fixed size)
ISNET = 1024

_state = {}


def status(text, pct=None):
    postStatus(text, -1 if pct is None else float(pct))


async def _net(name, x):
    """x: float32 (1,3,H,W) -> float32 (H,W) first output channel."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    res = await ortRun(name, to_js(x.tobytes()), to_js(list(x.shape)))
    y = np.frombuffer(res.to_py(), dtype=np.float32)
    return y.reshape(x.shape[2], x.shape[3])


async def alpha_isnet(bgr):
    h, w = bgr.shape[:2]
    x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (ISNET, ISNET), interpolation=cv2.INTER_AREA)
    x = x.astype(np.float32) / 255 - 0.5
    y = await _net('isnet', x.transpose(2, 0, 1)[None])
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    return cv2.resize(y, (w, h), interpolation=cv2.INTER_LINEAR).clip(0, 1)


async def _run_birefnet(rgb):
    h, w = rgb.shape[:2]
    x = (cv2.resize(rgb, (BIREF, BIREF), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 - MEAN) / STD
    y = await _net('birefnet', x.transpose(2, 0, 1)[None])
    return cv2.resize(y, (w, h), interpolation=cv2.INTER_LINEAR)


async def alpha_birefnet(bgr):
    """Same two-pass logic as cutout.alpha_birefnet: whole frame, then a re-run on a crop around the spider."""
    rgb = np.ascontiguousarray(bgr[..., ::-1])
    h, w = rgb.shape[:2]
    status('Cutting out: finding the spider', 30)
    a = await _run_birefnet(rgb)
    ys, xs = np.nonzero(a > 0.3)
    if len(ys) < 50:
        return a
    bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
    pad = 0.25 * max(bw, bh)
    x0, x1 = int(max(0, xs.min() - pad)), int(min(w, xs.max() + pad))
    y0, y1 = int(max(0, ys.min() - pad)), int(min(h, ys.max() + pad))
    if (x1 - x0) * (y1 - y0) > 0.7 * w * h:
        return a
    status('Cutting out: legs and hairs (close-up pass)', 60)
    a2 = await _run_birefnet(np.ascontiguousarray(rgb[y0:y1, x0:x1]))
    out = np.zeros_like(a)
    out[y0:y1, x0:x1] = a2
    return out


async def set_photo(rgba, w, h):
    t = time.time()
    _state.clear(); gc.collect()
    a = np.frombuffer(rgba.to_py(), dtype=np.uint8).reshape(h, w, 4)
    bgr = cv2.cvtColor(a, cv2.COLOR_RGBA2BGR)
    del a
    status('Cutting out: rough outline', 5)
    a_i = await alpha_isnet(bgr)
    a_b = await alpha_birefnet(bgr)
    status('Cutting out: tidying the edge', 90)
    alpha = co.fuse(a_b, a_i).astype(np.float32)
    del a_i, a_b; gc.collect()
    if (alpha > 0.5).sum() < 200:
        raise RuntimeError('NO_SPIDER: the cutout found no spider in this photo')
    _state.update(bgr=bgr, alpha=alpha)
    status('Cutout done', 100)
    return round(time.time() - t, 1)


STAGES = [('maps', 15), ('background', 35), ('shadow', 45), ('eyes', 55), ('spider', 92), ('total', 98)]


def _progress_from_log(line):
    for key, pct in STAGES:
        if line.strip().startswith(key):
            names = {'maps': 'measuring the spider', 'background': 'painting the wash', 'shadow': 'cast shadow',
                     'eyes': 'eyes', 'spider': 'painting the spider', 'total': 'finishing'}
            status('Painting: ' + names[key], pct)
            return


async def paint(eyes_json, size, vibrance, seed):
    if 'alpha' not in _state:
        raise RuntimeError('no photo has been cut out yet')
    t = time.time()
    eyes = [tuple(p) for p in json.loads(eyes_json)] or None
    sp.VIBRANCE = float(vibrance)
    status('Painting: starting', 2)
    res = sp.render(_state['bgr'], _state['alpha'], eye_hints=eyes, long_side=int(size),
                    border='wash', seed=int(seed), mode='plate')
    ok, buf = cv2.imencode('.jpg', res, [cv2.IMWRITE_JPEG_QUALITY, 92])
    del res; gc.collect()
    status('Done', 100)
    return to_js(buf.tobytes()), round(time.time() - t, 1)
