"""
spoodpaint — macro salticid photo -> gouache-and-ink illustration.

One neural step (the cutout, cutout.py). Everything else is procedural and
driven by maps measured from the photo, so there is no per-species knob:

  scale      sqrt(mask area)          -> every length (strokes, outline, shadow)
  flow       structure tensor + mask  -> direction of hairs, hatching, brush
  hair       local texture energy     -> how many setae get drawn
  gloss      specular + smoothness    -> highlights instead of hair
  light      bright-side centroid     -> outline weight, cast shadow direction
"""
import cv2, numpy as np, os, sys, time
from scipy import ndimage as ndi

RNG = np.random.default_rng(7)

# ----------------------------------------------------------------- utilities
def f32(im): return im.astype(np.float32) / 255.0
def u8(im):  return (np.clip(im, 0, 1) * 255 + 0.5).astype(np.uint8)
def lum(rgb): return rgb @ np.float32([0.299, 0.587, 0.114])
def blur(x, s): return cv2.GaussianBlur(x, (0, 0), s) if s > 0 else x
def smoothstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-8), 0, 1); return t * t * (3 - 2 * t)

def to_lab(rgb):  return cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2Lab)
def from_lab(l):  return np.clip(cv2.cvtColor(l.astype(np.float32), cv2.COLOR_Lab2RGB), 0, 1)

def noise(shape, s, seed=0):
    """band-limited noise in [-1,1]"""
    r = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    n = blur(r, s); return n / (np.abs(n).max() + 1e-8)

def fbm(shape, base, octaves=4, seed=0):
    out = np.zeros(shape, np.float32); a = 1.0
    for o in range(octaves):
        out += a * noise(shape, base / 2 ** o, seed + o); a *= 0.5
    return out / 1.9

# ----------------------------------------------------------------- mask
LIMB_STRENGTH = 0.9      # 0 = legs painted like the body, 1 = fully redrawn as banded tubes
TRIM_BY_COLOUR = False
VIBRANCE = 1.0           # >1 = richer spider colour (app slider); 1.0 = v0.4 look
EYE_LOG = []         # eye positions of the last render, in output pixels (for review sheets)   # colour-based halo trimming: helps on clean leaves, eats dark abdomens on dark bark

def refine_mask(alpha, rgb=None):
    """ISNet alpha -> clean hard silhouette.
    Matting nets haze around defocused legs and bridge the gaps between them; we trim any
    uncertain pixel whose colour is just the background showing through."""
    core = alpha > 0.5
    lab, n = ndi.label(core)
    if n == 0: return alpha.astype(np.float32)
    sizes = ndi.sum(core, lab, range(1, n + 1))
    keep = np.zeros(n + 1, bool); keep[1:] = sizes > 0.03 * sizes.max()
    # detached pieces must sit near the main body (a severed leg segment does; a stalk fragment doesn't)
    main = lab == (np.argmax(sizes) + 1)
    Smain = np.sqrt(main.sum())
    dmain = ndi.distance_transform_edt(~main)
    for i in range(1, n + 1):
        if keep[i] and i != np.argmax(sizes) + 1 and dmain[lab == i].min() > 0.2 * Smain:
            keep[i] = False
    m = keep[lab]
    S = np.sqrt(m.sum())
    if rgb is not None and TRIM_BY_COLOUR:
        # local background colour, filled in under the spider
        bgm = (alpha < 0.05).astype(np.float32)
        bgc = pushpull_fill(rgb, bgm)
        lab_i, lab_b = to_lab(rgb), to_lab(bgc)
        d = np.linalg.norm(lab_i - lab_b, axis=2)
        d = blur(d, 0.003 * S + 0.5)
        like_bg = d < 8 - 4 * smoothstep(0.5, 0.85, alpha)      # the more confident the net, the closer the match must be
        m &= ~(like_bg & (alpha < 0.85))
        m = ndi.binary_opening(m, iterations=1)
        # re-keep only components touching the confident core
        lab2, n2 = ndi.label(m)
        if n2:
            amax = ndi.maximum(alpha, lab2, range(1, n2 + 1))
            sz = ndi.sum(m, lab2, range(1, n2 + 1))
            okc = np.zeros(n2 + 1, bool); okc[1:] = (amax > 0.7) & (sz > 0.002 * S * S)
            m = okc[lab2]
    # fill only small holes (specks), never the background enclosed between legs
    holes = ndi.binary_fill_holes(m) & ~m
    hl, nh = ndi.label(holes)
    if nh:
        hs = ndi.sum(holes, hl, range(1, nh + 1))
        ha = ndi.mean(alpha, hl, range(1, nh + 1))
        m |= ((hs < 0.0015 * S * S) | ((ha > 0.1) & (hs < 0.05 * S * S)))[hl - 1] & holes
    mf = blur(m.astype(np.float32), max(1.0, 0.0025 * S))
    return (mf > 0.5).astype(np.float32)

# ----------------------------------------------------------------- analysis maps
def structure(L, s_grad, s_tens):
    gx = cv2.Sobel(L, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(L, cv2.CV_32F, 0, 1, ksize=3)
    gx, gy = blur(gx, s_grad), blur(gy, s_grad)
    E, F, G = blur(gx * gx, s_tens), blur(gx * gy, s_tens), blur(gy * gy, s_tens)
    tr = E + G; d = np.sqrt((E - G) ** 2 + 4 * F * F)
    l1, l2 = (tr + d) / 2, (tr - d) / 2
    # tangent = eigenvector of the smaller eigenvalue (along edges)
    tx, ty = F, l2 - E
    small = (tx * tx + ty * ty) < 1e-12
    tx = np.where(small, 1, tx); ty = np.where(small, 0, ty)
    nrm = np.sqrt(tx * tx + ty * ty); tx /= nrm; ty /= nrm
    coh = ((l1 - l2) / (l1 + l2 + 1e-6)).astype(np.float32)
    return tx.astype(np.float32), ty.astype(np.float32), coh

def body_flow(mask, scale):
    """direction along limbs: perpendicular to the distance-transform gradient."""
    dt = cv2.distanceTransform((mask > 0.5).astype(np.uint8), cv2.DIST_L2, 5)
    dt = blur(dt, 0.02 * scale)
    gx = cv2.Sobel(dt, cv2.CV_32F, 1, 0); gy = cv2.Sobel(dt, cv2.CV_32F, 0, 1)
    n = np.sqrt(gx * gx + gy * gy) + 1e-6
    return -gy / n, gx / n, dt

def lic(field_x, field_y, tex, length, step=1.0):
    """line integral convolution along an axial (sign-free) direction field.
    The field is sampled in doubled-angle form so interpolation never cancels across a sign flip."""
    h, w = tex.shape[:2]
    c2 = (field_x * field_x - field_y * field_y).astype(np.float32)
    s2 = (2 * field_x * field_y).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    acc = tex.astype(np.float32).copy(); cnt = 1.0
    n = max(1, int(length / step))
    for sgn in (1, -1):
        px, py = xx.copy(), yy.copy()
        dx, dy = field_x * sgn, field_y * sgn
        for i in range(n):
            a = cv2.remap(c2, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            b = cv2.remap(s2, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            th = 0.5 * np.arctan2(b, a)
            vx, vy = np.cos(th), np.sin(th)
            flip = np.where(vx * dx + vy * dy < 0, -1.0, 1.0).astype(np.float32)
            dx, dy = vx * flip, vy * flip
            px += dx * step; py += dy * step
            wgt = 1.0 - i / n
            acc += wgt * cv2.remap(tex, px, py, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            cnt += wgt
    return acc / cnt

# ----------------------------------------------------------------- painting primitives
def kuwahara(rgb, radius, sectors=8, q=8):
    """generalized Kuwahara (Papari 2007): flat painterly patches with crisp edges."""
    r = int(max(2, radius)); k = 2 * r + 1
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
    ang = np.arctan2(yy, xx); rad = np.sqrt(xx * xx + yy * yy)
    g = np.exp(-(rad ** 2) / (2 * (r / 2.2) ** 2)) * (rad <= r)
    num = 0; den = 0
    sq = rgb * rgb
    for i in range(sectors):
        a0 = -np.pi + 2 * np.pi * i / sectors
        d = np.angle(np.exp(1j * (ang - a0 - np.pi / sectors)))
        wsec = np.exp(-(d / (np.pi / sectors)) ** 2 * 2) * g
        wsec[r, r] = g[r, r] / sectors; wsec /= wsec.sum()
        m = cv2.filter2D(rgb, -1, wsec, borderType=cv2.BORDER_REFLECT)
        v = cv2.filter2D(sq, -1, wsec, borderType=cv2.BORDER_REFLECT) - m * m
        s = np.sqrt(np.clip(v.sum(2), 0, None)) + 1e-4
        wt = 1.0 / (1.0 + (s * 255) ** q * 1e-12) if False else s ** (-q)
        wt = wt / (wt.max() + 1e-12) if False else wt
        num = num + m * wt[..., None]; den = den + wt
    return num / den[..., None]

def xdog(L, s, k=1.6, p=20.0, eps=0.0, phi=10.0):
    g1, g2 = blur(L, s), blur(L, s * k)
    d = (1 + p) * g1 - p * g2
    e = np.where(d >= eps, 1.0, 1.0 + np.tanh(phi * (d - eps)))
    return np.clip(e, 0, 1)

def taper_strokes(canvas, alpha_layer, pts, dirs, lengths, widths, colors, opac=1.0):
    """draw tapered spikes (root wide -> sharp tip) into canvas (float RGB) via an AA mask per colour batch."""
    h, w = canvas.shape[:2]
    if len(pts) == 0: return
    lay = np.zeros((h, w), np.uint8)
    col = np.zeros((h, w, 3), np.float32); cw = np.zeros((h, w), np.float32)
    SS = 4  # subpixel shift bits
    for (x, y), (dx, dy), L, W, c in zip(pts, dirs, lengths, widths, colors):
        nx, ny = -dy, dx
        tip = (x + dx * L, y + dy * L)
        a = (x + nx * W * 0.5, y + ny * W * 0.5); b = (x - nx * W * 0.5, y - ny * W * 0.5)
        poly = (np.array([a, tip, b]) * (1 << SS)).astype(np.int32)
        one = np.zeros(1)
        cv2.fillConvexPoly(lay, poly, 255, cv2.LINE_AA, SS)
        cv2.fillConvexPoly(cw, poly, 1.0, cv2.LINE_AA, SS) if False else None
    return lay

def draw_spikes(canvas, pts, dirs, lengths, widths, color, opacity=1.0, ss=2):
    """tapered strokes of one colour, supersampled for clean AA; composited 'over'."""
    h, w = canvas.shape[:2]
    if len(pts) == 0: return canvas
    lay = np.zeros((h * ss, w * ss), np.uint8)
    SH = 3
    for (x, y), (dx, dy), L, W in zip(pts, dirs, lengths, widths):
        nx, ny = -dy, dx
        P = np.array([[x + nx * W / 2, y + ny * W / 2], [x + dx * L, y + dy * L],
                      [x - nx * W / 2, y - ny * W / 2], [x - dx * W * 0.4, y - dy * W * 0.4]])
        cv2.fillPoly(lay, [(P * ss * (1 << SH)).astype(np.int32)], 255, cv2.LINE_AA, SH)
    a = cv2.resize(lay, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 * opacity
    if np.ndim(color) == 1 or (np.ndim(color) == 3 and color.shape[:2] == (h, w)):
        return canvas * (1 - a[..., None]) + np.asarray(color, np.float32) * a[..., None]
    return canvas

def draw_colored_spikes(canvas, pts, dirs, lengths, widths, cols, opacity=1.0, ss=2):
    """per-stroke colour: rasterise ids, then lookup colours."""
    h, w = canvas.shape[:2]
    if len(pts) == 0: return canvas
    out = canvas.copy()
    # bucket colours by quantising, draw each bucket as a single layer
    q = np.clip((cols * 7).round().astype(int), 0, 7)
    keys = q[:, 0] * 64 + q[:, 1] * 8 + q[:, 2]
    for k in np.unique(keys):
        sel = keys == k
        out = draw_spikes(out, pts[sel], dirs[sel], lengths[sel], widths[sel], cols[sel].mean(0), opacity, ss)
    return out

def paper_texture(shape, scale, seed=3):
    """cold-press paper: bumpy height field, lit from top-left."""
    h, w = shape
    hgt = 0.6 * noise(shape, 1.2 * scale, seed) + 0.4 * noise(shape, 3.5 * scale, seed + 1) + 0.25 * noise(shape, 9 * scale, seed + 2)
    gx = cv2.Sobel(hgt, cv2.CV_32F, 1, 0); gy = cv2.Sobel(hgt, cv2.CV_32F, 0, 1)
    shade = -(gx + gy); shade /= (np.abs(shade).max() + 1e-6)
    return hgt, shade

# ----------------------------------------------------------------- background
def ink_band(dist, width):
    """anti-aliased ink line of given width from a distance field (px)."""
    return 1 - smoothstep(width * 0.5 - 0.6, width * 0.5 + 0.6, dist)

def make_stamps(n=8, L=160, W=48, seed=0):
    """brush stamps: alpha with bristle streaks, ragged ends and a darker pooled rim."""
    rng = np.random.default_rng(seed)
    u = np.linspace(0, 1, L)[None, :]; v = np.linspace(-1, 1, W)[:, None]
    out = []
    for k in range(n):
        n1 = blur(rng.standard_normal((W, 1)).astype(np.float32), 1.5); n1 /= np.abs(n1).max() + 1e-6
        n2 = blur(rng.standard_normal((1, L)).astype(np.float32), 6); n2 /= np.abs(n2).max() + 1e-6
        width = 0.85 + 0.12 * n2                                   # wobbling stroke width
        body = smoothstep(0.0, 0.18, width - np.abs(v))
        start = smoothstep(0.0, 0.10 + 0.06 * np.abs(n1), u)
        end = 1 - smoothstep(0.65 + 0.15 * n1, 1.0, u)              # ragged, dry tail
        bristle = 0.6 + 0.4 * (0.5 + 0.5 * n1)
        dry = (0.5 + 0.5 * n1) > (u - 0.55) * 1.6                   # bristles run out toward the end
        a = body * start * end * bristle * dry
        rim = smoothstep(0.55, 0.9, np.abs(v) / width) * body
        out.append((a.astype(np.float32), rim.astype(np.float32)))
    return out

def wash_blotches(base, Sd, rng, mask=None, seed=0, th_dom=0.0):
    """watercolour washes: irregular blobs of locally-sampled colour, laid big-to-small, each with a
    darker dried edge and slightly lighter (backrun) centre. Mimics hand-laid washes on a defocused ground."""
    h, w = base.shape[:2]
    canvas = base.copy()
    ref = blur(base, 0.06 * Sd + 2)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    for size, count, op in ((0.45, 14, 0.55), (0.28, 26, 0.5), (0.16, 50, 0.42)):
        r0 = size * Sd + 6
        warp = fbm((h, w), r0 * 0.22, 3, int(rng.integers(1 << 20)))
        rag = noise((h, w), 1.2, int(rng.integers(1 << 20)))
        for _ in range(int(count * (w * h) / (Sd * Sd * 4.0))):
            cx, cy = rng.uniform(-0.1 * w, 1.1 * w), rng.uniform(-0.1 * h, 1.1 * h)
            if mask is not None and mask[int(np.clip(cy, 0, h - 1)), int(np.clip(cx, 0, w - 1))] < 0.5: continue
            R = r0 * rng.uniform(0.6, 1.4); ex = rng.uniform(0.35, 0.8); th = th_dom + rng.normal(0, 0.35)
            x0, x1 = int(max(0, cx - 1.6 * R)), int(min(w, cx + 1.6 * R)); y0, y1 = int(max(0, cy - 1.6 * R)), int(min(h, cy + 1.6 * R))
            if x1 <= x0 or y1 <= y0: continue
            X, Y = xx[y0:y1, x0:x1] - cx, yy[y0:y1, x0:x1] - cy
            u = (X * np.cos(th) + Y * np.sin(th)); v = (-X * np.sin(th) + Y * np.cos(th)) / ex
            d = np.sqrt(u * u + v * v) / R + 0.75 * warp[y0:y1, x0:x1] + 0.03 * rag[y0:y1, x0:x1]
            inside = smoothstep(1.0, 0.94, d)                          # hard-ish dried edge
            if inside.max() < 0.5: continue
            rim = inside * smoothstep(0.7, 0.98, d)                   # pigment pooled at the edge
            centre = inside * (1 - smoothstep(0.0, 0.6, d))
            m = inside > 0.5
            col = ref[y0:y1, x0:x1][m].mean(0) * (1 + rng.normal(0, 0.08)) * (1 + rng.normal(0, 0.035, 3))
            a = op * inside
            roi = canvas[y0:y1, x0:x1]
            tint = col[None, None, :] * (1 - 0.22 * rim[..., None]) * (1 + 0.05 * centre[..., None])
            canvas[y0:y1, x0:x1] = roi * (1 - a[..., None]) + tint * a[..., None]
    return np.clip(canvas, 0, 1)

def stroke_layer(base, tx, ty, spacing, length, width, opacity, rng, jitter=0.04, mask=None):
    """Hertzmann-style stroke pass: oriented textured strokes, colour picked from `base`."""
    h, w = base.shape[:2]
    canvas = base.copy()
    stamps = make_stamps(seed=int(rng.integers(1 << 30)))
    gy, gx = np.mgrid[spacing / 2:h:spacing, spacing / 2:w:spacing]
    pts = np.stack([gx.ravel(), gy.ravel()], 1) + rng.uniform(-spacing / 2, spacing / 2, (gx.size, 2))
    rng.shuffle(pts)
    ref = blur(base, spacing * 0.3)
    for x, y in pts:
        xi, yi = int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))
        if mask is not None and mask[yi, xi] < 0.5: continue
        Lk = length * rng.uniform(0.7, 1.3); Wk = width * rng.uniform(0.75, 1.25)
        ang = np.arctan2(ty[yi, xi], tx[yi, xi]) + rng.normal(0, 0.12)
        a_st, rim = stamps[rng.integers(len(stamps))]
        sL, sW = a_st.shape[1], a_st.shape[0]
        ca, sa = np.cos(ang), np.sin(ang)
        # affine: stamp (u along length, v across) -> image, centred at (x, y)
        A = np.float32([[ca * Lk / sL, -sa * Wk / sW, 0], [sa * Lk / sL, ca * Wk / sW, 0]])
        A[:, 2] = np.float32([x, y]) - A[:, :2] @ np.float32([sL / 2, sW / 2])
        r = int(max(Lk, Wk) / 2 + 2)
        x0, y0, x1, y1 = max(0, int(x) - r), max(0, int(y) - r), min(w, int(x) + r), min(h, int(y) + r)
        if x1 <= x0 or y1 <= y0: continue
        A2 = A.copy(); A2[:, 2] -= np.float32([x0, y0])
        al = cv2.warpAffine(a_st, A2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR) * opacity
        rm = cv2.warpAffine(rim, A2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        col = (ref[yi, xi] * (1 + rng.normal(0, jitter)) * (1 + rng.normal(0, jitter * 0.4, 3))).astype(np.float32)
        tint = col[None, None, :] * (1 - 0.18 * rm[..., None])
        roi = canvas[y0:y1, x0:x1]
        canvas[y0:y1, x0:x1] = roi * (1 - al[..., None]) + tint * al[..., None]
    return canvas

def pushpull_fill(rgb, keep):
    """fill holes by normalised convolution over a pyramid (fast at any hole size)."""
    h, w = rgb.shape[:2]
    acc = np.zeros_like(rgb); accw = np.zeros((h, w), np.float32)
    im, kp = rgb * keep[..., None], keep.copy()
    lvl = 0
    while True:
        scale = 2 ** lvl
        s_im = cv2.resize(im, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)
        s_kp = cv2.resize(kp, (max(1, w // scale), max(1, h // scale)), interpolation=cv2.INTER_AREA)
        s_im, s_kp = blur(s_im, 2), blur(s_kp, 2)
        acc += cv2.resize(s_im, (w, h), interpolation=cv2.INTER_LINEAR)
        accw += cv2.resize(s_kp, (w, h), interpolation=cv2.INTER_LINEAR)
        lvl += 1
        if (accw > 1e-3).all() and lvl > 3 or max(w, h) // scale < 8: break
    return acc / np.maximum(accw, 1e-6)[..., None]

def paint_background(bg, keep, S, rng, seed):
    h, w = bg.shape[:2]
    ds = min(1.0, 640 / max(h, w)); hs, ws = int(h * ds), int(w * ds)
    b = cv2.resize(bg, (ws, hs), interpolation=cv2.INTER_AREA)
    Sd = S * ds
    Lb = lum(b)
    # large-scale orientation (leaf venation, grain) — smooth enough not to swirl
    tx, ty, coh = structure(Lb, 0.01 * Sd + 1, 0.25 * Sd)
    gcoh = float(np.clip((coh.mean() - 0.08) / 0.35, 0, 1))
    # where the scene has no clear grain, fall back to one dominant direction (a painter's stroke habit)
    c2, s2 = (tx * tx - ty * ty) * coh, 2 * tx * ty * coh
    th0 = 0.5 * np.arctan2(s2.mean(), c2.mean())
    if abs(c2.mean()) + abs(s2.mean()) < 0.05: th0 = np.deg2rad(-35)
    wgt = smoothstep(0.35, 0.75, coh)
    c2b = wgt * (tx * tx - ty * ty) + (1 - wgt) * np.cos(2 * th0)
    s2b = wgt * (2 * tx * ty) + (1 - wgt) * np.sin(2 * th0)
    th = 0.5 * np.arctan2(blur(s2b, 0.1 * Sd), blur(c2b, 0.1 * Sd))
    tx, ty = np.cos(th).astype(np.float32), np.sin(th).astype(np.float32)
    # how much real, in-focus structure the background has (absolute, so defocus mush scores ~0)
    # how much visible structure the background has at "brushstroke" scale (weave, bark, venation);
    # mid-band so mild defocus still counts as structure but a smooth bokeh wash does not
    band = blur(Lb, 0.004 * Sd + 0.7) - blur(Lb, 0.05 * Sd + 2)
    midA = blur(np.abs(band), 0.04 * Sd + 2)
    kdet = smoothstep(0.02, 0.05, midA)
    # 1) painterly patches: Kuwahara on a slightly domain-warped copy (irregular, hand-laid shapes)
    wx = fbm((hs, ws), 0.12 * Sd, 3, seed + 21) * 0.025 * Sd
    wy = fbm((hs, ws), 0.12 * Sd, 3, seed + 22) * 0.025 * Sd
    yy, xx = np.mgrid[0:hs, 0:ws].astype(np.float32)
    bw = cv2.remap(b, xx + wx, yy + wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    big = f32(cv2.edgePreservingFilter(u8(blur(bw, 0.006 * Sd + 0.5)), flags=cv2.RECURS_FILTER, sigma_s=0.3 * Sd + 8, sigma_r=0.3))
    small = f32(cv2.pyrMeanShiftFiltering(u8(np.clip(bw * 1.5 - 0.5 * blur(bw, 0.006 * Sd + 1), 0, 1)), sp=int(0.012 * Sd) + 3, sr=18, maxLevel=1))
    small = blur(small, 0.7)
    base = big * (1 - kdet[..., None]) + small * kdet[..., None]   # textured ground keeps its texture, defocus becomes wash
    # 2) brush streaks along the orientation (LIC on a smooth field)
    nz = blur(rng.standard_normal((hs, ws)).astype(np.float32), 0.8)
    fine_st = lic(tx, ty, nz, length=0.08 * Sd, step=1.0); fine_st /= fine_st.std() + 1e-6
    nz2 = blur(rng.standard_normal((hs, ws)).astype(np.float32), 0.02 * Sd + 1)
    broad = lic(tx, ty, nz2, length=0.15 * Sd, step=1.5); broad /= broad.std() + 1e-6
    # 3) wet-in-wet: blooms (lighter backruns) and darker charges of pigment, hue drift
    lab = to_lab(base)
    f1 = fbm((hs, ws), 0.16 * Sd, 3, seed + 24); f2 = fbm((hs, ws), 0.22 * Sd, 3, seed + 28)
    soft_k = 1 - 0.7 * kdet                               # leave real structure alone
    Lw = lab[..., 0] + soft_k * (8.0 * smoothstep(0.12, 0.3, f1) - 8.0 * smoothstep(0.18, 0.35, f2))
    lab[..., 1] += 3.0 * fbm((hs, ws), 0.25 * Sd, 2, seed + 25)
    lab[..., 2] += 5.0 * fbm((hs, ws), 0.25 * Sd, 2, seed + 26)
    # 4) pigment pooling: every edge gets a darker rim on its darker side
    r = 0.012 * Sd + 1
    pool = np.clip(blur(Lw, r) - Lw, 0, None)
    Lq = Lw - 1.6 * pool
    Lq += (0.15 + 0.6 * gcoh) * fine_st + (0.2 + 1.2 * gcoh) * broad
    # 5) gouache/wash value range: lift and compress, keep colour
    lab[..., 0] = np.clip(Lq * 0.74 + 25, 0, 100)
    lab[..., 1:] *= 0.9
    lab[..., 1:] *= 1.05
    painted = from_lab(lab)
    # brush strokes: two passes, broad then finer, following the scene's grain; skipped on real texture
    soft_mask = (kdet < 0.5).astype(np.float32)
    painted = wash_blotches(painted, Sd, rng, mask=(kdet < 0.25).astype(np.float32), seed=seed, th_dom=th0)
    painted = stroke_layer(painted, tx, ty, spacing=0.05 * Sd + 3, length=0.22 * Sd + 8, width=0.07 * Sd + 3,
                           opacity=0.12 + 0.3 * gcoh, rng=rng, jitter=0.07, mask=soft_mask)
    painted = stroke_layer(painted, tx, ty, spacing=0.035 * Sd + 2, length=0.12 * Sd + 5, width=0.035 * Sd + 2,
                           opacity=0.08 + 0.2 * gcoh, rng=rng, jitter=0.08, mask=soft_mask)
    out = cv2.resize(painted, (w, h), interpolation=cv2.INTER_CUBIC)
    # 6) ink: fine lines that follow real structure, only where structure exists
    txf = cv2.resize(tx, (w, h)); tyf = cv2.resize(ty, (w, h))
    prob = cv2.resize(kdet * smoothstep(0.2, 0.6, coh), (w, h))
    n = int(300 * prob.mean() + 0.5)
    inkc = np.zeros((h, w), np.uint8)
    if n:
        px = rng.uniform(0, w, n * 20); py = rng.uniform(0, h, n * 20)
        ii, jj = py.astype(int).clip(0, h - 1), px.astype(int).clip(0, w - 1)
        ok = (rng.uniform(0, 1, len(px)) < prob[ii, jj]) & (keep[ii, jj] > 0.5)
        px, py = px[ok][:n], py[ok][:n]
        lw = 0.002 * S + 0.5
        stepl = max(2.0, 0.01 * S)
        for x, y in zip(px, py):
            L = rng.uniform(0.1, 0.45) * S; pts = [(x, y)]; dx, dy = txf[int(y), int(x)], tyf[int(y), int(x)]
            for k in range(int(L / stepl)):
                xi, yi = int(np.clip(pts[-1][0], 0, w - 1)), int(np.clip(pts[-1][1], 0, h - 1))
                vx, vy = txf[yi, xi], tyf[yi, xi]
                if vx * dx + vy * dy < 0: vx, vy = -vx, -vy
                dx, dy = vx, vy
                nx_, ny_ = pts[-1][0] + dx * stepl, pts[-1][1] + dy * stepl
                if not (0 <= nx_ < w and 0 <= ny_ < h) or keep[int(ny_), int(nx_)] < 0.5: break
                pts.append((nx_, ny_))
            if len(pts) > 3:
                cv2.polylines(inkc, [(np.array(pts) * 8).astype(np.int32)], False, int(255 * rng.uniform(0.4, 0.8)), max(1, int(round(lw))), cv2.LINE_AA, 3)
    # 7) ink marks on textured ground (bark cracks, weave edges): broken strokes on strong edges
    if kdet.mean() > 0.15:
        Ls = lum(small)
        g = np.hypot(cv2.Sobel(Ls, cv2.CV_32F, 1, 0), cv2.Sobel(Ls, cv2.CV_32F, 0, 1))
        hiT = np.percentile(g, 97) * 255
        ed = (cv2.Canny(u8(blur(Ls, 0.7)), hiT * 0.5, hiT, L2gradient=True) > 0) & (kdet > 0.3)
        lab_e, ne = ndi.label(ed, structure=np.ones((3, 3)))
        if ne:
            sz = ndi.sum(ed, lab_e, range(1, ne + 1)); ed &= (sz > 0.12 * Sd)[lab_e - 1]
        # only the darker side of each edge gets the line, dashed by noise
        edf = cv2.resize(ed.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        de = cv2.distanceTransform((edf < 0.3).astype(np.uint8), cv2.DIST_L2, 3)
        marks = ink_band(de, 0.0016 * S + 0.6) * cv2.resize(kdet, (w, h)) * 0.75
        inkc = np.maximum(inkc, (marks * 255).astype(np.uint8))
    return out, inkc.astype(np.float32) / 255

# ----------------------------------------------------------------- spider
def paint_spider(rgb, M, Mb, S, ld, c0, hair, spec, bg, rng, seed, eye_hints=None):
    h, w = rgb.shape[:2]
    inside = M > 0.5
    pad = int(0.08 * S)
    ys, xs = np.nonzero(inside)
    y0, y1 = max(0, ys.min() - pad), min(h, ys.max() + pad); x0, x1 = max(0, xs.min() - pad), min(w, xs.max() + pad)
    cr = (slice(y0, y1), slice(x0, x1))
    sp = rgb[cr]; mc = inside[cr]

    # -- colour: remove the background's colour cast from low-chroma parts, then enrich real colour
    lab = to_lab(sp)
    bgab = to_lab(bg)[..., 1:][~inside].mean(0)
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    # bounce light / translucency: a pixel is "cast" if its tint points the same way as the background
    # right behind it AND is no stronger than that background (explainable by mixing). Only that
    # along-component is removed; real colour (yellow hair, a lime leg on a dark leaf) survives.
    bl = to_lab(bg[cr])
    bab = blur(bl[..., 1:], 0.05 * S)
    bc = np.linalg.norm(bab, axis=2) + 1e-6
    u = bab / bc[..., None]
    along = (lab[..., 1:] * u).sum(2)
    perp = np.abs(lab[..., 1] * u[..., 1] - lab[..., 2] * u[..., 0])
    dh = np.arctan2(perp, np.clip(along, 1e-6, None))
    cast = smoothstep(4, 12, bc) * (1 - smoothstep(0.25, 0.5, dh)) * (1 - smoothstep(0.9, 1.3, along / bc))
    lab[..., 1:] -= (0.6 * cast * np.clip(along, 0, None))[..., None] * u
    lab[..., 2] += 5.0 * cast                                  # what remains reads as warm (yellowish), translucent cuticle
    chroma = np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)
    boost = 1.0 + 0.45 * smoothstep(8, 25, chroma)
    boost = boost * (1.0 + (VIBRANCE - 1.0) * smoothstep(3, 12, chroma))   # user vibrancy; leaves greys grey
    lab[..., 1:] *= boost[..., None]
    # -- value: stretch to full range (ink-black to paper-white) inside the body
    Lc = lab[..., 0]
    lo, hi = np.percentile(Lc[mc], [1.5, 99.5])
    t = np.clip((Lc - lo) / (hi - lo + 1e-6), 0, 1)
    t = t ** 0.85
    lab[..., 0] = 10 + 88 * (0.55 * t * t * (3 - 2 * t) + 0.45 * t)   # darkest = slate, so form still reads; ink supplies true black
    sp2 = from_lab(lab)

    # -- gouache flats: edge-preserving smoothing (domain transform), keep a whisper of photo texture
    # sharpen (recovers some defocus), then mean-shift: flat colour patches with crisp boundaries,
    # which is what a gouache layer is. Background pixels are replaced by their nearest spider colour
    # first so the filter never drags leaf-green into the legs.
    sp_in = sp2.copy()
    if (~mc).any():
        idx = ndi.distance_transform_edt(~mc, return_distances=False, return_indices=True)
        sp_in = sp2[idx[0], idx[1]]
    shp = np.clip(sp_in * 1.7 - 0.7 * blur(sp_in, 0.005 * S + 1), 0, 1)
    flat = f32(cv2.pyrMeanShiftFiltering(u8(shp), sp=int(0.014 * S) + 3, sr=22, maxLevel=1))
    flat = blur(flat, 0.6)
    detail = sp2 - blur(sp2, 0.004 * S + 0.8)
    paint = flat + 0.12 * detail

    # -- form shading: rounder toward the silhouette (lets flat colour read as volume)
    dti = cv2.distanceTransform(mc.astype(np.uint8), cv2.DIST_L2, 5)
    loc_max = ndi.maximum_filter(dti, size=int(0.12 * S) | 1)
    rim = 1 - smoothstep(0.0, 0.85, dti / (loc_max + 1e-6))
    shadeL = np.clip(-(0), 0, 1)
    gx = cv2.Sobel(blur(mc.astype(np.float32), 0.03 * S), cv2.CV_32F, 1, 0); gy = cv2.Sobel(blur(mc.astype(np.float32), 0.03 * S), cv2.CV_32F, 0, 1)
    gn = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
    away = np.clip((gx * ld[0] + gy * ld[1]) / gn, -1, 1)          # +1 = surface facing away from light
    paint *= (1 - (0.10 + 0.15 * away[..., None].clip(0, 1)) * rim[..., None] ** 2)

    # -- limbs: redraw thin parts as clean banded tubes (see limbs.py); body parts keep the gouache flats
    import limbs
    tube, limbw, joint, linfo = limbs.limb_tube(sp2, mc.astype(np.float32), S, ld, seed=seed)
    limbw = limbw * LIMB_STRENGTH
    # put the leg's own fine pattern (mottling, pale rings) back on top of the clean tube: a band-pass
    # of the photo adds real markings where they exist and nothing where the leg was just blur
    Lsp = lum(sp2)
    bp = blur(Lsp, 0.0015 * S + 0.5) - blur(Lsp, 0.008 * S + 1)
    tube = np.clip(tube + 1.1 * bp[..., None], 0, 1)
    paint = paint * (1 - limbw[..., None]) + (0.85 * tube + 0.15 * paint) * limbw[..., None]
    body = rgb.copy(); body[cr] = np.clip(paint, 0, 1)
    full = lambda a: (lambda z: (z.__setitem__(cr, a), z)[1])(np.zeros((h, w), np.float32))
    dti_f = full(dti)

    # -- flow on the body: parallel to the silhouette (along legs, around the carapace)
    bx, by, _ = body_flow(M, S)

    ink = np.zeros((h, w), np.float32)
    # -- interior lines: strong edges of the flattened image (pattern boundaries, joints)
    Ls = lum(flat)
    Lsu = u8(blur(Ls, 0.0025 * S + 0.5))
    g = np.hypot(cv2.Sobel(Ls, cv2.CV_32F, 1, 0), cv2.Sobel(Ls, cv2.CV_32F, 0, 1))
    hiT = np.percentile(g[mc], 92) * 255; loT = hiT * 0.5
    ed = cv2.Canny(Lsu, loT, hiT, L2gradient=True) > 0
    ed &= dti > 0.006 * S
    lab_e, ne = ndi.label(ed, structure=np.ones((3, 3)))
    if ne:
        sz = ndi.sum(ed, lab_e, range(1, ne + 1))
        ed = (sz > 0.035 * S)[lab_e - 1] & ed
    de = cv2.distanceTransform((~ed).astype(np.uint8), cv2.DIST_L2, 3)
    wl = (0.0022 * S + 0.5)
    inner = ink_band(de, wl) * smoothstep(0.004 * S, 0.012 * S, dti)
    inner *= 1 - 0.7 * limbw                     # on limbs the band joints replace edge-detected lines
    ink[cr] = np.maximum(ink[cr], inner * 0.85)
    # pattern edges: boundaries between flat colour patches with a large colour difference (stripes,
    # chevrons, eye-region bands). Absolute threshold in Lab, so bold patterns get bold lines and a
    # uniformly mottled spider doesn't get scribbled over.
    flab = to_lab(blur(flat, 0.0015 * S + 0.6))
    dE = np.zeros(flab.shape[:2], np.float32)
    for c, wgt in ((0, 1.0), (1, 0.8), (2, 0.8)):
        gx_ = cv2.Sobel(flab[..., c], cv2.CV_32F, 1, 0, ksize=3); gy_ = cv2.Sobel(flab[..., c], cv2.CV_32F, 0, 1, ksize=3)
        dE += wgt * (gx_ * gx_ + gy_ * gy_)
    dE = np.sqrt(dE) / 8.0                                       # ~ dE per pixel
    strong = cv2.Canny(np.clip(dE * 6, 0, 255).astype(np.uint8), 40, 90, L2gradient=True) > 0
    strong &= mc & (dti > 0.008 * S)
    lab_p, npat = ndi.label(strong, structure=np.ones((3, 3)))
    if npat:
        szp = ndi.sum(strong, lab_p, range(1, npat + 1)); strong &= (szp > 0.04 * S)[lab_p - 1]
    dp = cv2.distanceTransform((~strong).astype(np.uint8), cv2.DIST_L2, 3)
    pat = ink_band(dp, 0.0026 * S + 0.6) * (1 - 0.8 * limbw)
    ink[cr] = np.maximum(ink[cr], pat * 0.8)
    ink[cr] = np.maximum(ink[cr], joint * 0.9 * (limbw > 0.3))

    # -- outline: width heavier on the side away from the light
    dto = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 5)
    din = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 5)
    Gx = cv2.Sobel(blur(M, 0.01 * S), cv2.CV_32F, 1, 0); Gy = cv2.Sobel(blur(M, 0.01 * S), cv2.CV_32F, 0, 1)
    Gn = np.hypot(Gx, Gy) + 1e-6; nxo, nyo = -Gx / Gn, -Gy / Gn
    shade_side = np.clip(-(nxo * ld[0] + nyo * ld[1]), 0, 1)
    wid = (0.0032 * S + 0.7) * (0.75 + 0.9 * shade_side) * (1 + 0.2 * noise((h, w), 0.03 * S, seed + 2))
    sd = np.where(inside, -din, dto)
    ink = np.maximum(ink, ink_band(np.abs(sd + 0.25 * wid), wid))

    # -- hatching: fine parallel lines in the shaded part of each colour patch, following form
    Lb_ = lum(body)
    locm = blur(Lb_, 0.03 * S)
    shade = smoothstep(0.02, 0.15, locm - Lb_) * 0.45 + rim_full(rim, cr, h, w) ** 2 * 0.45
    shade *= smoothstep(0.08, 0.25, Lb_)                           # not on ink-black areas
    n_h = int(M.sum() / (0.006 * S) ** 2)
    px = rng.uniform(x0, x1, n_h * 4); py = rng.uniform(y0, y1, n_h * 4)
    ii, jj = py.astype(int).clip(0, h - 1), px.astype(int).clip(0, w - 1)
    ok = inside[ii, jj] & (dti_f[ii, jj] > 0.005 * S) & (rng.uniform(0, 1, len(px)) < shade[ii, jj] * 0.9)
    px, py, ii, jj = px[ok][:n_h], py[ok][:n_h], ii[ok][:n_h], jj[ok][:n_h]
    d = np.stack([bx[ii, jj], by[ii, jj]], 1) + rng.normal(0, 0.06, (len(px), 2))
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
    L_ = rng.uniform(0.012, 0.03, len(px)) * S
    hatch = draw_spikes(np.zeros((h, w, 3), np.float32), np.stack([px, py], 1) - d * L_[:, None] / 2, d, L_, np.full(len(px), 0.0018 * S + 0.5), np.float32([1, 1, 1]))[..., 0]
    ink = np.maximum(ink, hatch * 0.7 * Mb)

    # -- fur: dense short ink strokes along the body flow, denser where darker / hairier
    n_f = int(M.sum() / (0.0045 * S) ** 2)
    px = rng.uniform(x0, x1, n_f * 3); py = rng.uniform(y0, y1, n_f * 3)
    ii, jj = py.astype(int).clip(0, h - 1), px.astype(int).clip(0, w - 1)
    dens = (0.15 + 0.85 * hair[ii, jj]) * (0.3 + 0.7 * (1 - Lb_[ii, jj])) * smoothstep(0.003 * S, 0.012 * S, dti_f[ii, jj])
    ok = inside[ii, jj] & ~spec[ii, jj] & (rng.uniform(0, 1, len(px)) < dens)
    px, py, ii, jj = px[ok][:n_f], py[ok][:n_f], ii[ok][:n_f], jj[ok][:n_f]
    d = np.stack([bx[ii, jj], by[ii, jj]], 1)
    d *= np.sign((d * (np.stack([px, py], 1) - c0)).sum(1, keepdims=True) + 1e-6)
    d += rng.normal(0, 0.22, d.shape); d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
    L_ = rng.uniform(0.008, 0.02, len(px)) * S
    fur = draw_spikes(np.zeros((h, w, 3), np.float32), np.stack([px, py], 1), d, L_, np.full(len(px), 0.0016 * S + 0.45), np.float32([1, 1, 1]))[..., 0]
    ink = np.maximum(ink, fur * 0.75 * Mb * (1 - 0.8 * full(limbw)))

    # -- setae, pale: gouache hairs where the body is light, sampled colour lifted
    n_s = int(M.sum() / (0.016 * S) ** 2)
    px = rng.uniform(x0, x1, n_s * 4); py = rng.uniform(y0, y1, n_s * 4)
    ii, jj = py.astype(int).clip(0, h - 1), px.astype(int).clip(0, w - 1)
    lightish = Lb_[ii, jj] > locm[ii, jj] - 0.01
    ok = inside[ii, jj] & (dti_f[ii, jj] > 0.004 * S) & lightish & ~spec[ii, jj] & (rng.uniform(0, 1, len(px)) < 0.1 + 0.9 * hair[ii, jj])
    px, py, ii, jj = px[ok][:n_s], py[ok][:n_s], ii[ok][:n_s], jj[ok][:n_s]
    d = np.stack([bx[ii, jj], by[ii, jj]], 1)
    d *= np.sign((d * (np.stack([px, py], 1) - c0)).sum(1, keepdims=True) + 1e-6)   # point distally
    d += rng.normal(0, 0.2, d.shape); d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
    L_ = rng.uniform(0.01, 0.025, len(px)) * S
    cols = np.clip(body[ii, jj] * 1.2 + 0.15, 0, 1)
    body = draw_colored_spikes(body, np.stack([px, py], 1), d, L_, np.full(len(px), 0.0022 * S + 0.5), cols, 0.75)

    # -- setae, fringe: fine ink spikes off the silhouette, outward and distal
    fringe = None
    edge_px = np.argwhere(inside & (din <= 1.5))
    if len(edge_px):
        k = min(len(edge_px), int(len(edge_px) * 0.35))
        sel = edge_px[rng.choice(len(edge_px), k, replace=False)]
        fy, fx = sel[:, 0], sel[:, 1]
        thin = 1 - smoothstep(0.02 * S, 0.08 * S, ndi.maximum_filter(dti_f, size=int(0.05 * S) | 1)[fy, fx])  # legs & palps
        p_keep = (0.1 + 0.9 * hair[fy, fx]) * (0.35 + 0.65 * thin)
        kk = rng.uniform(0, 1, len(fx)) < p_keep
        fy, fx, thin = fy[kk], fx[kk], thin[kk]
        on = np.stack([nxo[fy, fx], nyo[fy, fx]], 1)
        al = np.stack([bx[fy, fx], by[fy, fx]], 1)
        al *= np.sign((al * (np.stack([fx, fy], 1) - c0)).sum(1, keepdims=True) + 1e-6)
        d = on * 0.7 + al * 0.75 + rng.normal(0, 0.15, on.shape)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-6
        fl = (0.01 + 0.035 * hair[fy, fx] * rng.uniform(0.2, 1.0, len(fx)) ** 2) * S * (0.6 + 0.6 * thin)
        # hairs take the colour of the body they grow from: dark parts get ink setae, pale parts pale ones
        root = np.stack([fx, fy], 1).astype(np.float32) - d * 1.5
        Lroot = lum(body)[fy, fx]
        pale = (Lroot > 0.45) | (rng.uniform(0, 1, len(fx)) < 0.2)
        wdt = np.full(len(fx), 0.0026 * S + 0.6)
        spikes = draw_spikes(np.zeros((h, w, 3), np.float32), root[~pale], d[~pale], fl[~pale], wdt[~pale], np.float32([1, 1, 1]))[..., 0]
        ink = np.maximum(ink, spikes)
        pcol = np.clip(body[fy, fx][pale] * 1.15 + 0.12, 0, 1)
        fringe_pale = draw_colored_spikes(np.zeros((h, w, 3), np.float32), root[pale], d[pale], fl[pale], wdt[pale], pcol, 1.0)
        fringe_a = draw_spikes(np.zeros((h, w, 3), np.float32), root[pale], d[pale], fl[pale], wdt[pale], np.float32([1, 1, 1]))[..., 0]
        fringe = (fringe_pale, fringe_a)

    # -- gloss: crisp opaque highlights
    # specular spots become clean painted shapes: merged, smoothed, small specks dropped,
    # with a soft lighter halo (gouache highlight laid wet-on-dry)
    sf = cv2.morphologyEx(spec.astype(np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))).astype(np.float32)
    shape = blur(sf, 0.004 * S + 0.8)
    hl = smoothstep(0.38, 0.55, shape)
    lbl_h, nh = ndi.label(hl > 0.5)
    if nh:
        sz = ndi.sum(hl > 0.5, lbl_h, range(1, nh + 1))
        hl *= blur(((sz > 3e-5 * S * S)[lbl_h - 1] & (lbl_h > 0)).astype(np.float32), 0.8)
    halo = smoothstep(0.1, 0.4, blur(hl, 0.01 * S)) * (1 - hl)
    body = body * (1 - 0.92 * hl[..., None]) + np.float32([0.97, 0.97, 0.95]) * 0.92 * hl[..., None]
    body = body + 0.10 * halo[..., None] * (1 - body)

    # -- eyes (eyes.py): candidates with a confidence each; taps or a confident AME pair get the full
    #    glossy-lens repaint, other likely eyes near them are enhanced in place
    import eyes as eyemod
    pool = eyemod.detect(rgb, M, S, limbw=full(limbw))
    full_e, soft_e = eyemod.choose(pool, S, hints=eye_hints, L=blur(lum(rgb), 0.6))
    body, ink = eyemod.enhance(body, ink, soft_e, S, ld, rgb)
    body, ink = eyemod.render(body, ink, full_e, S, ld, rgb)
    if eye_hints: print("   hints", [(round(x), round(y)) for x, y in eye_hints], "->", [(round(e["x"]), round(e["y"]), round(e["r"])) for e in full_e])
    EYE_LOG[:] = [(e['x'], e['y'], e['r']) for e in full_e + soft_e]
    print(f"  eyes: {len(full_e)} repainted, {len(soft_e)} enhanced" + (" (from taps)" if eye_hints else ""))
    return body, ink, fringe

def rim_full(rim, cr, h, w):
    z = np.zeros((h, w), np.float32); z[cr] = rim; return z

# ----------------------------------------------------------------- main render
def render(bgr, alpha, long_side=2000, border='wash', seed=7, mode='plate', eye_hints=None):
    """eye_hints: optional list of (x, y) eye centres in the ORIGINAL photo's pixel coordinates."""
    t0 = time.time()
    box = [0, 0]
    bgr, alpha = auto_crop(bgr, alpha, fill=0.62 if mode == 'scene' else 0.70, box=box)
    h0, w0 = bgr.shape[:2]
    sc = long_side / max(h0, w0)
    img = cv2.resize(bgr, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    alpha = cv2.resize(alpha, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    rgb = f32(img[..., ::-1].copy())
    frame = None
    if mode == 'plate':
        # room around the photo so the wash can be a free puddle and cut-off legs can taper into paper
        hp_, wp_ = rgb.shape[:2]; p = int(0.12 * max(hp_, wp_))
        rgb = cv2.copyMakeBorder(rgb, p, p, p, p, cv2.BORDER_REPLICATE)
        alpha = cv2.copyMakeBorder(alpha, p, p, p, p, cv2.BORDER_CONSTANT, value=0)
        frame = (p, p, wp_, hp_)
    h, w = rgb.shape[:2]
    padp = frame[0] if frame is not None else 0
    hints_r = [((x - box[0]) * sc + padp, (y - box[1]) * sc + padp) for (x, y) in eye_hints] if eye_hints else None
    M = refine_mask(alpha, rgb)
    S = float(np.sqrt(M.sum()))                     # spider scale in px
    Mb = blur(M, 0.004 * S + 0.5)
    L = lum(rgb)
    rng = np.random.default_rng(seed)

    # ---- maps
    tx, ty, coh = structure(L, 1.0, 0.012 * S)
    bx, by, dt = body_flow(M, S)
    hp = L - blur(L, 0.006 * S + 1)
    tex_energy = np.sqrt(blur(hp * hp, 0.01 * S))
    inside = M > 0.5
    te = tex_energy / (np.percentile(tex_energy[inside], 90) + 1e-6) if inside.any() else tex_energy
    hair = smoothstep(0.25, 0.9, te)                # 0 smooth/glossy .. 1 hairy
    sat_rgb = rgb.max(2) - rgb.min(2)
    spec = (L > np.percentile(L[inside], 97)) & (sat_rgb < 0.25) & inside if inside.any() else np.zeros_like(inside)
    # light: from mask centroid toward bright-pixel centroid
    yy, xx = np.nonzero(inside)
    if len(yy):
        wL = L[inside] ** 3
        c0 = np.array([xx.mean(), yy.mean()]); c1 = np.array([(xx * wL).sum(), (yy * wL).sum()]) / wL.sum()
        ld = c1 - c0; ld = ld / (np.linalg.norm(ld) + 1e-6) if np.linalg.norm(ld) > 0.01 * S else np.array([-0.6, -0.8])
        ld = 0.5 * ld + 0.5 * np.array([-0.6, -0.8]); ld /= np.linalg.norm(ld)
    else:
        c0 = np.array([w / 2, h / 2]); ld = np.array([-0.6, -0.8])
    print(f"  maps {time.time()-t0:.1f}s  S={S:.0f}px light={ld.round(2)}")

    # ================================================================ BACKGROUND
    # 1) remove the spider by masked push-pull fill (normalised convolution)
    hole = cv2.dilate(((M > 0.5) | (alpha > 0.4)).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(0.03 * S) | 1,) * 2))
    vig = lens_vignette(L, M)                         # black porthole corners from the macro lens
    keep = 1 - np.maximum(hole.astype(np.float32), vig)
    filled = pushpull_fill(rgb, keep)
    bg = rgb * keep[..., None] + filled * (1 - keep[..., None])
    if mode == 'scene':
        bgp, bg_ink = paint_background(bg, keep, S, rng, seed)
    else:
        bgp, bg_ink = plate_background(bg, M, S, ld, rng, seed), np.zeros((h, w), np.float32)
    print(f"  background {time.time()-t0:.1f}s")

    # ================================================================ CAST SHADOW
    sv = -ld * 0.10 * S
    sh = np.zeros((h, w), np.float32)
    for i, f in enumerate(np.linspace(0.45, 1.0, 5)):
        T = np.float32([[1, 0, sv[0] * f], [0, 1, sv[1] * f]])
        sh = np.maximum(sh, blur(cv2.warpAffine(M, T, (w, h)), 0.003 * S + 0.025 * S * f) * (1.15 - 0.5 * f))
    sh = np.clip(sh, 0, 1) * (1 - Mb)
    ang = np.arctan2(ld[1], ld[0]) + np.pi / 4
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    period = max(3.0, 0.008 * S)
    ph = (xx * np.cos(ang) + yy * np.sin(ang)) / period + 0.5 * noise((h, w), 0.05 * S, seed + 5)
    hatch = smoothstep(0.7, 0.92, 0.5 + 0.5 * np.cos(2 * np.pi * ph)) * smoothstep(0.3, 0.75, sh)
    print(f"  shadow {time.time()-t0:.1f}s")

    # ================================================================ SPIDER
    body, ink, fringe = paint_spider(rgb, M, Mb, S, ld, c0, hair, spec, bg, rng, seed, eye_hints=hints_r)
    print(f"  spider {time.time()-t0:.1f}s")

    if mode == 'plate':
        fade = frame_fade(h, w, S, seed, frame)
        Mb = Mb * fade; ink = ink * fade; sh = sh * fade; hatch = hatch * fade
        if fringe is not None: fringe = (fringe[0], fringe[1] * fade)

    # ================================================================ COMPOSITE
    out = bgp * (1 - (0.32 if mode == 'scene' else 0.22) * sh[..., None])
    INK = np.float32([0.07, 0.06, 0.07])
    out = out * (1 - 0.55 * hatch[..., None]) + INK * 0.55 * hatch[..., None]
    out = out * (1 - 0.55 * bg_ink[..., None]) + INK * 0.55 * bg_ink[..., None]
    out = out * (1 - Mb[..., None]) + body * Mb[..., None]
    if fringe is not None:
        fp, fa = fringe
        out = out * (1 - 0.9 * fa[..., None]) + fp * 0.9 * (fa[..., None] > 0)
    out = out * (1 - ink[..., None]) + INK * ink[..., None]

    # ---- paper: pigment granulates into the valleys of the tooth (more pigment, more granulation)
    hgt, shade = paper_texture((h, w), max(0.8, 0.0018 * long_side), seed)
    pigment = 1 - lum(out)
    out = out * (1 - (0.03 + 0.10 * pigment) * np.clip(-hgt, 0, None))[..., None]
    out = out * (1 + 0.035 * shade[..., None])
    # paper white is cream, ink-black stays black
    cream = np.float32([0.975, 0.955, 0.915])
    out = out * cream
    if mode == 'plate':
        # trim empty paper: frame the painted area (wash + spider) with an even margin
        painted = (np.abs(out - cream).max(2) > 0.04).astype(np.uint8)
        painted = cv2.morphologyEx(painted, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        ys_, xs_ = np.nonzero(painted)
        if len(ys_):
            mg = int(0.05 * max(h, w))
            Y0, Y1 = max(0, ys_.min() - mg), min(h, ys_.max() + mg); X0, X1 = max(0, xs_.min() - mg), min(w, xs_.max() + mg)
            out = out[Y0:Y1, X0:X1]
            EYE_LOG[:] = [(x - X0, y - Y0, r) for x, y, r in EYE_LOG]
        pp = int(0.02 * min(out.shape[:2]))
        EYE_LOG[:] = [(x + pp, y + pp, r) for x, y, r in EYE_LOG]
        out = pad_paper(out, cream, seed, frac=0.02)
    elif border == 'deckle':
        out = deckle(out, seed)
    elif border == 'wash':
        protect = np.clip(Mb + blur(ink, 2), 0, 1)
        out = wash_border(out, protect, S, seed, cream)
    print(f"  total {time.time()-t0:.1f}s")
    return u8(out)[..., ::-1]

def plate_background(bg, M, S, ld, rng, seed):
    """No scene: paper with a loose watercolour wash behind the spider, in the scene's colours
    (lightened), shaped like a hand-laid puddle with a dried pigment edge and a few wet blooms."""
    h, w = M.shape
    paper = np.ones((h, w, 3), np.float32)
    ds = min(1.0, 700 / max(h, w)); hs, ws = int(h * ds), int(w * ds); Sd = S * ds
    m = cv2.resize(M, (ws, hs), interpolation=cv2.INTER_AREA)
    b = cv2.resize(bg, (ws, hs), interpolation=cv2.INTER_AREA)
    # 1) wash shape: distance from the spider, warped -> organic puddle around it, longer along the light axis
    dist = cv2.distanceTransform((m < 0.5).astype(np.uint8), cv2.DIST_L2, 5) / (Sd + 1e-6)
    ys, xs = np.nonzero(m > 0.5); cy, cx = ys.mean(), xs.mean()
    yy, xx = np.mgrid[0:hs, 0:ws].astype(np.float32)
    # elliptical puddle sized from the spider's extent (legs included), plus a little hug of its outline
    y0_, y1_ = np.percentile(ys, [1, 99]); x0_, x1_ = np.percentile(xs, [1, 99])
    ax_, ay_ = max(0.5 * (x1_ - x0_), 0.3 * Sd), max(0.5 * (y1_ - y0_), 0.3 * Sd)
    cx, cy = 0.5 * (x0_ + x1_), 0.5 * (y0_ + y1_)
    ell = np.hypot((xx - cx) / ax_, (yy - cy) / ay_)
    field = 0.2 * dist + 0.42 * np.maximum(ell - 0.25, 0)
    dedge = np.minimum.reduce([xx, yy, ws - 1 - xx, hs - 1 - yy]) / (Sd + 1e-6)
    field += 0.9 * smoothstep(0.15, 0.0, dedge)                # the puddle closes before the frame edge
    field += 0.22 * fbm((hs, ws), 0.35 * Sd, 3, seed + 51) + 0.04 * fbm((hs, ws), 0.04 * Sd, 2, seed + 52)
    edge = 0.62
    wash = smoothstep(edge + 0.012, edge - 0.012, field)
    # second, smaller glaze laid over the first once it dried: its own hard edge
    f2 = field + 0.12 * fbm((hs, ws), 0.25 * Sd, 3, seed + 55)
    wash2 = smoothstep(0.3 + 0.01, 0.3 - 0.01, f2)
    # 2) colour: scene colours, blurred, lightened toward paper; stronger near the spider
    col = blur(b, 0.25 * Sd)
    lab = to_lab(col)
    lab[..., 0] = 72 + 0.25 * (lab[..., 0] - 50)
    lab[..., 1:] *= 0.75 * VIBRANCE ** 0.5
    col = from_lab(lab)
    near = smoothstep(edge, 0.02, field)                       # 1 next to the spider, 0 at the rim
    depth = (0.35 + 0.45 * near + 0.12 * fbm((hs, ws), 0.2 * Sd, 3, seed + 53)) * wash
    # 3) pigment pooled at the dried edge, blooms (lighter backruns) inside
    rim = wash * smoothstep(edge - 0.07, edge - 0.004, field)
    bloom = wash * smoothstep(0.25, 0.45, fbm((hs, ws), 0.12 * Sd, 2, seed + 54))
    rim2 = wash2 * smoothstep(0.24, 0.298, f2)
    depth = depth + 0.35 * rim - 0.18 * bloom + 0.18 * wash2 + 0.2 * rim2
    # granulation: pigment settles in specks
    gran = noise((hs, ws), 0.7, seed + 56)
    depth *= 1 + 0.12 * gran
    out = paper[:hs, :ws] * (1 - depth[..., None]) + col * depth[..., None]
    # 4) a few streaky dry-brush strokes leaving the puddle (direction = away from light)
    out = cv2.resize(np.clip(out, 0, 1), (w, h), interpolation=cv2.INTER_CUBIC)
    return out

def frame_fade(h, w, S, seed, frame=None):
    """parts of the spider cut off by the photo's edge: taper them out instead of a hard crop."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    x0, y0, fw, fh = frame if frame is not None else (0, 0, w, h)
    d = np.minimum.reduce([xx - x0, yy - y0, x0 + fw - 1 - xx, y0 + fh - 1 - yy])
    L = 0.06 * S * (1 + 0.5 * noise((h, w), 0.02 * S, seed + 61))
    return smoothstep(0.0, 1.0, d / (L + 1e-6)).astype(np.float32)

def pad_paper(out, cream, seed, frac=0.07):
    """extra paper margin so the plate breathes."""
    h, w = out.shape[:2]; p = int(frac * min(h, w))
    canvas = np.ones((h + 2 * p, w + 2 * p, 3), np.float32) * cream
    canvas[p:p + h, p:p + w] = out
    # fade the composite's own rectangle edge into paper so no box is visible
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy])
    a = smoothstep(0, 0.02 * min(h, w), d)[..., None]
    canvas[p:p + h, p:p + w] = out * a + cream * (1 - a)
    return canvas

def auto_crop(bgr, alpha, fill=0.62, aspect=4 / 3, box=None):
    """frame the spider like an illustration: its bounding box spans ~`fill` of the frame."""
    h, w = alpha.shape
    ys, xs = np.nonzero(alpha > 0.5)
    if len(ys) < 50: return bgr, alpha
    y0, y1 = np.percentile(ys, [0.2, 99.8]); x0, x1 = np.percentile(xs, [0.2, 99.8])
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    asp = aspect if w >= h else 1 / aspect
    cw = max((x1 - x0) / fill, (y1 - y0) / fill * asp)
    cw = min(cw, w, h * asp); ch = cw / asp
    if cw > 0.92 * w and ch > 0.92 * h: return bgr, alpha
    X0 = int(np.clip(cx - cw / 2, 0, w - cw)); Y0 = int(np.clip(cy - ch / 2, 0, h - ch))
    if box is not None: box[:] = [X0, Y0]
    return bgr[Y0:Y0 + int(ch), X0:X0 + int(cw)], alpha[Y0:Y0 + int(ch), X0:X0 + int(cw)]

def find_eyes(L, M, dti, S):
    """Salticid eyes: dark, round, similar-sized blobs that come in neighbouring pairs (AMEs side by
    side, ALEs beside them). A lone dark spot is ignored, so abdominal spots don't become eyes."""
    inside = M > 0.5
    if not inside.any(): return []
    Lb = blur(L, 0.002 * S + 0.5)
    cands = []
    for pct in (6, 10, 15):
        thr = np.percentile(Lb[inside], pct)
        dark = ((Lb < thr) & inside & (dti > 0.006 * S)).astype(np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        n, lab, st, cen = cv2.connectedComponentsWithStats(dark)
        for i in range(1, n):
            a = st[i, cv2.CC_STAT_AREA]
            if not (1.5e-4 * S * S < a < 1.2e-2 * S * S): continue
            cs, _ = cv2.findContours((lab == i).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            per = cv2.arcLength(cs[0], True)
            circ = 4 * np.pi * a / (per * per + 1e-6)
            (cx, cy), r = cv2.minEnclosingCircle(cs[0])
            fill = a / (np.pi * r * r + 1e-6)
            if circ < 0.7 or fill < 0.62: continue
            # an eye is darker than its surround by a clear margin
            ring = (np.hypot(*np.mgrid[-int(2 * r) - 1:int(2 * r) + 2, -int(2 * r) - 1:int(2 * r) + 2]) )
            y0, y1 = int(max(0, cy - 2 * r)), int(min(L.shape[0], cy + 2 * r + 1)); x0, x1 = int(max(0, cx - 2 * r)), int(min(L.shape[1], cx + 2 * r + 1))
            patch = Lb[y0:y1, x0:x1]; pm = inside[y0:y1, x0:x1]
            yy, xx = np.mgrid[y0:y1, x0:x1]; dd = np.hypot(xx - cx, yy - cy)
            surround = patch[(dd > 1.3 * r) & (dd < 2 * r) & pm]
            core = patch[dd < 0.7 * r]
            if surround.size < 5 or core.size < 3 or np.median(surround) - np.median(core) < 0.12: continue
            cands.append((cx, cy, r, np.median(surround) - np.median(core), circ, fill))
    # de-duplicate across thresholds (keep the larger)
    cands.sort(key=lambda c: -c[2]); uniq = []
    for c in cands:
        if all(np.hypot(c[0] - u[0], c[1] - u[1]) > 0.8 * max(c[2], u[2]) for u in uniq): uniq.append(c)
    # keep blobs that have a partner of similar size within ~5 radii
    eyes = []
    for i, c in enumerate(uniq):
        paired = any(i != j and 0.6 < c[2] / u[2] < 1.7 and 1.6 * max(c[2], u[2]) < np.hypot(c[0] - u[0], c[1] - u[1]) < 5.5 * max(c[2], u[2])
                     for j, u in enumerate(uniq))
        # a lone eye (side / three-quarter view) must be a near-perfect, high-contrast disc
        lone_ok = c[3] > 0.22 and c[4] > 0.82 and c[5] > 0.75 and c[2] > 0.012 * S
        if paired or lone_ok:
            eyes.append((c[0], c[1], c[2] * 1.05))
    return eyes[:8]

def lens_vignette(L, M):
    """dark border-connected region outside a central disc = macro-lens porthole; returns soft mask."""
    h, w = L.shape
    dark = (blur(L, 3) < 0.10) & (M < 0.5)
    lab, n = ndi.label(dark)
    if n == 0: return np.zeros((h, w), np.float32)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])); border = border[border > 0]
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w / 2) / (w / 2), (yy - h / 2) / (h / 2))
    out = np.zeros((h, w), bool)
    for b in border:
        reg = lab == b
        a = reg.mean()
        if a < 0.3 and np.median(r[reg]) > 0.95: out |= reg
    out = cv2.dilate(out.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(np.float32)
    return out

def wash_border(out, protect, S, seed, paper):
    """the wash stops short of the edge with a ragged, lumpy brush margin; paper shows through."""
    h, w = out.shape[:2]
    mn = min(h, w); m = 0.055 * mn
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy])
    edge = m * (1 + 0.6 * fbm((h, w), 0.10 * mn, 2, seed + 31)) + 0.02 * mn * fbm((h, w), 0.02 * mn, 3, seed + 32) \
           + 0.006 * mn * noise((h, w), 0.003 * mn + 0.8, seed + 33)
    a = smoothstep(edge - 1.0, edge + 1.0, d)
    a = np.maximum(a, protect)
    rim = np.exp(-np.clip(d - edge, 0, None) / (0.006 * mn)) * (d > edge)
    tinted = out * (1 - 0.14 * rim)[..., None]
    return tinted * a[..., None] + paper * (1 - a[..., None])

def deckle(out, seed):
    h, w = out.shape[:2]
    m = int(0.045 * min(h, w))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy])
    wob = m * (1 + 0.35 * fbm((h, w), m * 0.8, 3, seed + 9))
    a = smoothstep(wob - 2, wob + 2, d)
    paper = np.float32([0.96, 0.95, 0.92])
    return out * a[..., None] + paper * (1 - a[..., None])

if __name__ == '__main__':
    import argparse
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import cutout
    ap = argparse.ArgumentParser(description='macro salticid photo -> gouache & ink illustration')
    ap.add_argument('inputs', nargs='+')
    ap.add_argument('-o', '--outdir', default='out')
    ap.add_argument('--border', default='wash', choices=['wash', 'deckle', 'none'])
    ap.add_argument('--size', type=int, default=2000, help='long side of output, px')
    ap.add_argument('--seed', type=int, default=7, help='different seed = different brushwork')
    ap.add_argument('--no-crop', action='store_true')
    ap.add_argument('--eyes', default=None, help='eye centres in photo pixels, "x,y;x,y". Otherwise read from photo.eyes.json if present (see pick_eyes.py)')
    ap.add_argument('--mode', default='plate', choices=['plate', 'scene'], help='plate = spider on paper with a wash; scene = painted background')
    ap.add_argument('--masks', default='work/masks2', help='folder of precomputed alpha PNGs (computed if missing)')
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    if args.no_crop: auto_crop = lambda b, a, box=None, **k: (b, a)
    for f in args.inputs:
        print(os.path.basename(f))
        im = cv2.imread(f)
        mp = os.path.join(args.masks, os.path.splitext(os.path.basename(f))[0] + '.png')
        a = cv2.imread(mp, 0) / 255.0 if os.path.exists(mp) else cutout.alpha(im)
        hints = None
        side = os.path.splitext(f)[0] + '.eyes.json'
        if args.eyes and len(args.inputs) == 1:
            hints = [tuple(map(float, p.split(','))) for p in args.eyes.split(';') if p.strip()]
        elif os.path.exists(side):
            import json; hints = [tuple(p) for p in json.load(open(side))]
        res = render(im, a.astype(np.float32), eye_hints=hints, long_side=args.size, border=None if args.border == 'none' else args.border, seed=args.seed, mode=args.mode)
        cv2.imwrite(os.path.join(args.outdir, os.path.splitext(os.path.basename(f))[0] + '.jpg'), res, [cv2.IMWRITE_JPEG_QUALITY, 92])
        import json; json.dump(EYE_LOG, open(os.path.join(args.outdir, os.path.splitext(os.path.basename(f))[0] + '.eyelog.json'), 'w'))
