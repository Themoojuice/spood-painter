"""Salticid eyes: find them, then paint them the way an illustrator would.

Detection (no learning, just what eyes look like in a macro photo):
  1. scale-normalised Laplacian-of-Gaussian over a range of radii -> dark round blobs at any size
  2. each candidate is grown into a region (threshold halfway between core and surround), fitted with
     an ellipse (so side-on eyes still count), and measured:
       contrast   surround minus core brightness
       roundness  ellipse axis ratio + how well the region fills its ellipse
       gloss      a small bright catchlight inside the dark disc (glossy lens) - the strongest cue on
                  dark-faced spiders where the eye barely differs from the carapace
       placement  eye must sit on the thick body (carapace), not on a leg or the silhouette edge
  3. group rules: salticid eyes come in mirror pairs and all sit together on the front of the carapace.
     Candidates that pair up, or are individually very convincing, seed the eye cluster; anything far
     from the cluster is dropped.
"""
import cv2, numpy as np
from scipy import ndimage as ndi

def _blur(x, s): return cv2.GaussianBlur(x, (0, 0), s) if s > 0 else x
def _ss(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-8), 0, 1); return t * t * (3 - 2 * t)
def _lum(rgb): return rgb @ np.float32([0.299, 0.587, 0.114])


def detect(rgb, M, S, debug=False, limbw=None, probe=None):
    h, w = M.shape
    inside = M > 0.5
    if inside.sum() < 100: return ([], []) if debug else []
    ys, xs = np.nonzero(inside)
    pad = int(0.05 * S) + 4
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad); x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)
    L = _blur(_lum(rgb[y0:y1, x0:x1]).astype(np.float32), 0.6)
    m = inside[y0:y1, x0:x1]
    dti = cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5)
    thick = ndi.maximum_filter(dti, size=int(0.08 * S) | 1)          # body thickness around each pixel
    lw = limbw[y0:y1, x0:x1] if limbw is not None else np.zeros_like(dti)
    # fill outside the mask with local body brightness so the silhouette edge doesn't look like a blob
    if (~m).any():
        idx = ndi.distance_transform_edt(~m, return_distances=False, return_indices=True)
        L = L[idx[0], idx[1]]

    # 1) scale-space LoG (computed on a downsampled copy for big radii)
    radii = np.geomspace(max(1.5, 0.005 * S), 0.1 * S, 15)
    resp = []
    for r in radii:
        sg = r / np.sqrt(2)
        f = 1.0 if sg < 6 else 6.0 / sg
        Ls = cv2.resize(L, None, fx=f, fy=f, interpolation=cv2.INTER_AREA) if f < 1 else L
        R = (sg * f) ** 2 * ndi.gaussian_laplace(Ls, sg * f)
        if f < 1: R = cv2.resize(R, (L.shape[1], L.shape[0]), interpolation=cv2.INTER_LINEAR)
        resp.append(R)
    resp = np.stack(resp)
    peak = (resp == ndi.maximum_filter(resp, size=(3, 5, 5))) & (resp > 0.012)
    ks, py, px = np.nonzero(peak)
    order = np.argsort(-resp[ks, py, px])[:400]

    cands = []
    NA, NR = 40, 28
    th = np.linspace(0, 2 * np.pi, NA, endpoint=False)
    for k, cy, cx in zip(ks[order], py[order], px[order]):
        r = radii[k]
        if not m[cy, cx] or dti[cy, cx] < 0.45 * r: continue
        if thick[cy, cx] < 0.035 * S or lw[cy, cx] > 0.3: continue      # on a leg / palp, not the carapace
        # 2) star-shaped edge search: along each ray find where brightness rises most steeply.
        #    Works when eyes touch each other (those rays just have no edge) and on oblique eyes.
        rs = np.linspace(0.3 * r, 2.1 * r, NR)
        qx = cx + np.outer(np.cos(th), rs); qy = cy + np.outer(np.sin(th), rs)
        prof = ndi.map_coordinates(L, [qy.ravel(), qx.ravel()], order=1, mode='nearest').reshape(NA, NR)
        prof = ndi.uniform_filter1d(prof, 3, axis=1)
        dprof = np.diff(prof, axis=1)
        # first strong rise along each ray (not the strongest): stops a ray that leaves the eye into
        # dark cuticle from running on to a far, brighter edge
        dmax = dprof.max(axis=1, keepdims=True)
        strong_ = dprof > 0.5 * np.maximum(dmax, 1e-6)
        j = np.argmax(strong_, axis=1)
        for i_ in range(NA):   # climb to the local peak of that rise
            while j[i_] + 1 < dprof.shape[1] and dprof[i_, j[i_] + 1] > dprof[i_, j[i_]]: j[i_] += 1
        er_ray = 0.5 * (rs[j] + rs[j + 1])
        lo = np.array([prof[i, :max(1, j[i] - 1)].mean() for i in range(NA)])
        hi = np.array([prof[i, min(NR - 1, j[i] + 2):min(NR, j[i] + 6)].mean() if j[i] + 2 < NR else prof[i, -1] for i in range(NA)])
        cr = hi - lo
        good = (cr > 0.04) & (er_ray > 0.45 * r) & (er_ray < 1.95 * r)
        # edge sharpness: how many radial steps the rise takes (an eye's rim is crisp, a colour patch is soft)
        width = []
        for i in np.nonzero(good)[0]:
            p = prof[i]; l0, h0 = lo[i], hi[i]
            t20, t80 = l0 + 0.2 * (h0 - l0), l0 + 0.8 * (h0 - l0)
            a_ = np.argmax(p[:j[i] + 2] > t20) if (p[:j[i] + 2] > t20).any() else j[i]
            b_ = j[i] + np.argmax(p[j[i]:] > t80) if (p[j[i]:] > t80).any() else NR - 1
            width.append((rs[min(b_, NR - 1)] - rs[a_]))
        coher = float(good.mean())
        if good.sum() < 8: continue
        # weight rays by edge strength; robust ellipse: refit twice dropping the worst-fitting edge points
        keep_ = good & (cr > 0.5 * np.median(cr[good]))
        if keep_.sum() < 8: keep_ = good
        ptsx = cx + np.cos(th[keep_]) * er_ray[keep_]; ptsy = cy + np.sin(th[keep_]) * er_ray[keep_]
        (ex, ey), (A, B), ang = cv2.fitEllipse(np.stack([ptsx, ptsy], 1).astype(np.float32))
        for _ in range(2):
            ca_, sa_ = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
            U_ = (ptsx - ex) * ca_ + (ptsy - ey) * sa_; V_ = -(ptsx - ex) * sa_ + (ptsy - ey) * ca_
            res_ = np.abs(np.sqrt((U_ / max(A / 2, 1)) ** 2 + (V_ / max(B / 2, 1)) ** 2) - 1)
            ok_ = res_ < max(0.12, np.percentile(res_, 75))
            if ok_.sum() < 6: break
            ptsx, ptsy = ptsx[ok_], ptsy[ok_]
            (ex, ey), (A, B), ang = cv2.fitEllipse(np.stack([ptsx, ptsy], 1).astype(np.float32))
        ra, rb = A / 2, B / 2
        if not (0.4 * r < max(ra, rb) < 2.2 * r): continue
        ratio = min(ra, rb) / max(ra, rb)
        er = np.sqrt(ra * rb)
        if er > 0.11 * S or er < 1.5: continue
        soft = float(np.median(width)) / er if width else 1.0
        # fit quality: how far the edge points sit from the ellipse (relative)
        cosA, sinA = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
        U = (ptsx - ex) * cosA + (ptsy - ey) * sinA; V = -(ptsx - ex) * sinA + (ptsy - ey) * cosA
        fit = float(np.median(np.abs(np.sqrt((U / ra) ** 2 + (V / rb) ** 2) - 1)))
        contrast = float(np.median(cr[good]))
        W = int(1.6 * max(ra, rb)) + 2
        a0, a1 = int(max(0, ey - W)), int(min(L.shape[0], ey + W + 1)); b0, b1 = int(max(0, ex - W)), int(min(L.shape[1], ex + W + 1))
        P = L[a0:a1, b0:b1]; pm = m[a0:a1, b0:b1]
        yy, xx = np.mgrid[a0:a1, b0:b1]
        Uu = (xx - ex) * cosA + (yy - ey) * sinA; Vv = -(xx - ex) * sinA + (yy - ey) * cosA
        rr = np.sqrt((Uu / ra) ** 2 + (Vv / rb) ** 2)
        inner = P[(rr < 0.75) & pm]
        if inner.size < 4: continue
        core = float(np.percentile(inner, 40))
        darkpart = inner[inner < np.percentile(inner, 80)]
        unif = float(np.std(darkpart) / (contrast + 1e-6))
        ring = P[(rr > 1.2) & (rr < 1.6) & pm]
        sur = float(np.percentile(ring, 70)) if ring.size else core + contrast
        hmax = float(inner.max())
        spot = inner > core + 0.6 * (hmax - core)
        gloss = float((hmax - core > 0.2) and (hmax > sur - 0.05) and spot.mean() < 0.25)
        gl_pos = None
        if gloss:
            Pm = np.where((rr < 0.75) & pm, P, -1); jj = np.argmax(Pm)
            gl_pos = (xx.flat[jj] - ex, yy.flat[jj] - ey)
        size_w = 0.45 + 0.55 * _ss(0.006 * S, 0.025 * S, er)
        score = (_ss(0.04, 0.45, contrast) * _ss(0.4, 0.8, coher) * _ss(0.4, 0.75, ratio) * _ss(0.25, 0.08, fit)
                 * _ss(0.4, 0.04, unif) * _ss(0.9, 0.4, soft) * (0.55 + 0.45 * gloss) * size_w)
        cands.append(dict(x=ex + x0, y=ey + y0, r=er, ra=ra, rb=rb, ang=ang, contrast=contrast, ratio=ratio,
                          fill=1 - fit, gloss=gloss, gl=gl_pos, score=score, leak=False, k=int(k), coher=coher, unif=unif, soft=soft,
                          fit=fit, core=core, sur=sur, thick=float(thick[cy, cx] / S), dti=float(dti[cy, cx] / S), rS=float(er / S),
                          chroma=float(np.ptp(np.median(rgb[int(ey + y0)-1:int(ey + y0)+2, int(ex + x0)-1:int(ex + x0)+2].reshape(-1, 3), 0)))))

    # non-max suppression: keep the better of overlapping candidates
    cands.sort(key=lambda c: -c['score'])
    uniq = []
    for c in cands:
        if c['score'] < 0.1: continue
        if all(np.hypot(c['x'] - u['x'], c['y'] - u['y']) > 0.9 * max(c['r'], u['r']) for u in uniq):
            uniq.append(c)

    # 3) probability per candidate from a small logistic model fitted to hand-labelled eyes in 22 photos
    pool = [c for c in uniq if c['ratio'] > 0.5 and c['contrast'] > 0.08 and c['coher'] > 0.6]
    for c in pool: c['p'] = eye_prob(c, pool)
    pool.sort(key=lambda c: -c['p'])
    return (pool, uniq) if debug else pool


_MODEL = None
def _model():
    global _MODEL
    if _MODEL is None:
        import json, os
        _MODEL = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eye_model.json')))
    return _MODEL

def _pairf(c, L):
    best = 0.0
    for u in L:
        if u is c: continue
        d = np.hypot(c['x'] - u['x'], c['y'] - u['y']); rr = max(c['r'], u['r']); q = min(c['r'], u['r']) / rr
        if 1.4 * rr < d < 5 * rr: best = max(best, q * min(1, u['coher']) * (1 - u['fit']))
    return best

def eye_prob(c, pool):
    md = _model()
    v = [c[k] for k in md['feats']] + [_pairf(c, pool), np.log(c['rS'] + 1e-3), c['contrast'] / (c['sur'] + 0.05), c['core'] ** 0.5]
    z = (np.array(v) - md['mean']) / md['scale']
    return float(1 / (1 + np.exp(-(z @ np.array(md['coef']) + md['b']))))


def choose(pool, S, hints=None, L=None):
    """Returns (full, soft).
    full: eyes repainted as glossy lenses - the user's taps (snapped to the detected ellipse when one
          is there), or, with no taps, an automatic pick only when two confident, similar, adjacent
          discs form an anterior-median pair.
    soft: everything else that looks eye-like near the eye group, enhanced in place (darker lens,
          crisper rim, existing catchlight brightened) with strength = confidence. A wrong guess just
          makes a dark spot a little crisper; it never puts a fake eye on the abdomen."""
    full = []
    if hints:
        for (hx, hy) in hints:
            pr = _probe(hx, hy, L, S)
            # a tap is a precise centre: keep a detected ellipse only if it agrees with the tap
            near = [c for c in pool if np.hypot(c['x'] - hx, c['y'] - hy) < 0.3 * c['r'] and 0.7 < c['r'] / pr['r'] < 1.4]
            full.append(dict(min(near, key=lambda c: np.hypot(c['x'] - hx, c['y'] - hy))) if near else pr)
    else:
        conf = [c for c in pool if c['p'] > 0.85 and c['dti'] > 0.05]
        best = None
        for i, c in enumerate(conf):
            for u in conf[i + 1:]:
                rm = 0.5 * (c['r'] + u['r']); d = np.hypot(c['x'] - u['x'], c['y'] - u['y'])
                if (max(c['r'], u['r']) / min(c['r'], u['r']) < 1.3 and 1.8 * rm < d < 3.6 * rm and rm > 0.02 * S
                        and abs(c['core'] - u['core']) < 0.07 and abs(c['chroma'] - u['chroma']) < 0.06):   # a real pair matches in colour
                    sc = c['p'] * u['p']
                    if best is None or sc > best[0]: best = (sc, c, u)
        if best is not None: full = [best[1], best[2]]
    soft = []
    if pool:
        anchor = full[0] if full else pool[0]
        for c in pool:
            if any(np.hypot(c['x'] - f['x'], c['y'] - f['y']) < 0.8 * max(c['r'], f['r']) for f in full): continue
            if c['p'] < 0.6 or np.hypot(c['x'] - anchor['x'], c['y'] - anchor['y']) > 0.3 * S: continue
            soft.append(c)
    return full, soft[:8]

def _probe(hx, hy, L, S):
    """Eye at a tapped point: cast rays from the tap and take the first strong brightness rise on each
    as the rim; robust circle (or ellipse when the rim is clean) around the tap."""
    NA = 48; th = np.linspace(0, 2 * np.pi, NA, endpoint=False)
    rs = np.linspace(0.004 * S, 0.12 * S, 60)
    qx = hx + np.outer(np.cos(th), rs); qy = hy + np.outer(np.sin(th), rs)
    prof = ndi.map_coordinates(L, [qy.ravel(), qx.ravel()], order=1, mode='nearest').reshape(NA, -1)
    prof = ndi.uniform_filter1d(prof, 3, axis=1)
    d = np.diff(prof, axis=1)
    core = np.median(prof[:, :3])
    er = []
    for i in range(NA):
        thr = 0.5 * max(d[i].max(), 1e-6)
        js = np.nonzero((d[i] > thr) & (d[i] > 0.01))[0]
        if not len(js): continue
        j = js[0]
        while j + 1 < len(d[i]) and d[i, j + 1] > d[i, j]: j += 1
        if prof[i, min(j + 4, prof.shape[1] - 1)] - core > 0.05: er.append((th[i], 0.5 * (rs[j] + rs[j + 1])))
    if len(er) < 8:
        r = 0.02 * S; return dict(x=hx, y=hy, r=r, ra=r, rb=r, ang=0.0, gl=None, p=1.0, gloss=0.0)
    t_, r_ = np.array(er).T
    med = np.median(r_); ok = np.abs(r_ - med) < 0.35 * med
    t_, r_ = t_[ok], r_[ok]
    r = float(np.median(r_))
    out = dict(x=hx, y=hy, r=r, ra=r, rb=r, ang=0.0, gl=None, p=1.0, gloss=0.0)
    if len(r_) >= 12:
        px, py = hx + np.cos(t_) * r_, hy + np.sin(t_) * r_
        (ex, ey), (A, B), ang = cv2.fitEllipse(np.stack([px, py], 1).astype(np.float32))
        if np.hypot(ex - hx, ey - hy) < 0.3 * r and 0.6 < min(A, B) / max(A, B):
            out.update(x=0.5 * (ex + hx), y=0.5 * (ey + hy), ra=A / 2, rb=B / 2, ang=ang, r=float(np.sqrt(A * B) / 2))
    return out

def enhance(body, ink, soft, S, ld, src_rgb):
    """in-place eye enhancement, strength from confidence."""
    h, w = body.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for e in soft:
        s = float(_ss(0.6, 0.9, e['p']))
        ex, ey, ra, rb = e['x'], e['y'], max(e['ra'], 1.0), max(e['rb'], 1.0); ang = np.deg2rad(e['ang']); er = np.sqrt(ra * rb)
        B = int(1.5 * max(ra, rb)) + 3
        y0, y1 = int(max(0, ey - B)), int(min(h, ey + B + 1)); x0, x1 = int(max(0, ex - B)), int(min(w, ex + B + 1))
        X, Y = xx[y0:y1, x0:x1] - ex, yy[y0:y1, x0:x1] - ey
        u = X * np.cos(ang) + Y * np.sin(ang); v = -X * np.sin(ang) + Y * np.cos(ang)
        rr = np.sqrt((u / ra) ** 2 + (v / rb) ** 2)
        disc = _ss(1.05, 0.85, rr)
        roi = body[y0:y1, x0:x1]
        Lr = _lum(roi)
        dark = disc * (Lr < np.percentile(Lr[rr < 0.8], 85) + 0.02 if (rr < 0.8).any() else 1)
        roi = roi * (1 - 0.55 * s * dark[..., None])                          # deepen the lens
        # brighten the photo's own catchlight if it has one
        src = _lum(src_rgb[y0:y1, x0:x1])
        if (rr < 0.8).any() and e['gloss'] and e['p'] > 0.75:
            hi = src > max(np.percentile(src[rr < 0.8], 96), 0.5)
            hl = _blur((hi & (rr < 0.85)).astype(np.float32), 0.6)
            roi = roi * (1 - s * hl[..., None]) + 0.97 * s * hl[..., None]
        body[y0:y1, x0:x1] = roi
        if er > 2.5:
            lw = 0.0016 * S + 0.5
            line = _ss(lw / er + 1 / er, 0.0, np.abs(rr - 1.0))
            ink[y0:y1, x0:x1] = np.maximum(ink[y0:y1, x0:x1], line * 0.7 * s)
    return body, ink


def render(body, ink, eyes, S, ld, src_rgb):
    """Paint each eye as a glossy domed lens in its own colour (red-brown AMEs stay red-brown, black eyes
    stay black): darker toward the lit side and the centre, light refracted through to the far rim,
    a soft occlusion shadow around it, one crisp catchlight (placed where the photo had one), a small
    secondary reflection on big eyes, and a clean ink outline. Fur strokes and hatching are cleared
    off the lens so it reads as a smooth glass dome."""
    h, w = body.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for e in eyes:
        ex, ey = e['x'], e['y']
        ra, rb = max(e['ra'], 1.0) * 1.03, max(e['rb'], 1.0) * 1.03
        ang = np.deg2rad(e['ang']); er = np.sqrt(ra * rb)
        B = int(1.8 * max(ra, rb)) + 3
        y0, y1 = int(max(0, ey - B)), int(min(h, ey + B + 1)); x0, x1 = int(max(0, ex - B)), int(min(w, ex + B + 1))
        if y1 <= y0 or x1 <= x0: continue
        X, Y = xx[y0:y1, x0:x1] - ex, yy[y0:y1, x0:x1] - ey
        u = X * np.cos(ang) + Y * np.sin(ang); v = -X * np.sin(ang) + Y * np.cos(ang)
        rr = np.sqrt((u / ra) ** 2 + (v / rb) ** 2)
        aa = 1.2 / er
        disc = _ss(1.0 + aa, 1.0 - aa, rr)
        tow = (X * ld[0] + Y * ld[1]) / (er + 1e-6)
        # the eye's own colour from the photo (core, excluding the catchlight), in Lab
        src = src_rgb[y0:y1, x0:x1][(rr < 0.75)]
        if src.size:
            l_ = _lum(src); src = src[l_ < np.percentile(l_, 80)]
        col = np.median(src, 0).astype(np.float32) if src.size else np.float32([0.15, 0.08, 0.06])
        lab = cv2.cvtColor(col[None, None], cv2.COLOR_RGB2Lab)[0, 0]
        Lc = float(np.clip(lab[0], 8, 40)); ch = np.float32(lab[1:]) * 1.5            # colour: +20%
        def labrgb(Lv, chs):
            return cv2.cvtColor(np.float32([[[Lv, chs[0], chs[1]]]]), cv2.COLOR_Lab2RGB)[0, 0]
        deep = labrgb(Lc * 0.45, ch * 0.6); mid = labrgb(Lc, ch); glow = labrgb(min(58, Lc * 1.75 + 6), ch * 1.45)       # rim glow: brighter
        # dome: darkest at the centre/lit side, glowing far rim where light exits
        t_glow = _ss(0.3, 0.95, rr) * _ss(0.3, -0.8, tow)
        t_deep = (1 - _ss(0.0, 0.75, rr)) * 0.6 + _ss(0.0, 0.9, tow) * 0.4
        lens = mid[None, None] * (1 - t_deep[..., None]) + deep[None, None] * t_deep[..., None]
        lens = lens * (1 - t_glow[..., None]) + glow[None, None] * t_glow[..., None]
        lens *= (1 - 0.35 * _ss(0.85, 1.0, rr))[..., None]                 # dark rim line inside the edge
        occ = _ss(1.5, 1.0, rr) * (1 - disc) * (0.3 + 0.3 * np.clip(-tow, 0, 1))
        roi = body[y0:y1, x0:x1] * (1 - 0.45 * occ[..., None])
        roi = roi * (1 - disc[..., None]) + lens * disc[..., None]
        # catchlight
        if e.get('gl') is not None and np.hypot(*e['gl']) < 0.75 * er and (e['gl'][0] * ld[0] + e['gl'][1] * ld[1]) > 0:
            cx_, cy_ = ex + e['gl'][0], ey + e['gl'][1]
        else:
            cx_, cy_ = ex + ld[0] * er * 0.4, ey + ld[1] * er * 0.4
        if er >= 3:
            cr_ = max(1.4, 0.28 * er)                                      # highlight: ~15% bigger
            d1 = np.hypot((xx[y0:y1, x0:x1] - cx_) * 0.85, yy[y0:y1, x0:x1] - cy_) / cr_
            hl = _ss(1.0 + 1.2 / cr_, 1.0 - 1.2 / cr_, d1) * disc
            roi = roi * (1 - hl[..., None]) + np.float32([0.99, 0.99, 0.98]) * hl[..., None]
            if er >= 7:
                sx, sy = ex - (cx_ - ex) * 0.85, ey - (cy_ - ey) * 0.85
                d2 = np.hypot(xx[y0:y1, x0:x1] - sx, yy[y0:y1, x0:x1] - sy) / max(1.0, 0.095 * er)
                h2 = _ss(1.2, 0.8, d2) * disc * 0.7
                roi = roi * (1 - h2[..., None]) + glow * 1.6 * h2[..., None]
        else:
            d1 = np.hypot(xx[y0:y1, x0:x1] - cx_, yy[y0:y1, x0:x1] - cy_)
            hl = _ss(1.0, 0.3, d1) * disc * 0.85
            roi = roi * (1 - hl[..., None]) + 0.97 * hl[..., None]
        body[y0:y1, x0:x1] = np.clip(roi, 0, 1)
        # clear fur/hatch ink off the lens, then a clean outline
        ink[y0:y1, x0:x1] *= 1 - _ss(1.02, 0.9, rr)
        lw = (0.0017 * S + 0.5) * (0.85 + 0.5 * np.clip(-tow, 0, 1))
        line = _ss(lw / er + aa, 0.0, np.abs(rr - 1.0)) if er > 2 else np.zeros_like(rr)
        ink[y0:y1, x0:x1] = np.maximum(ink[y0:y1, x0:x1], line * 0.95)
    return body, ink
