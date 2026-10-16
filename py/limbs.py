"""Limb redraw: turn blurry leg pixels into clean illustrated tubes.

Medial axis of the silhouette -> every limb pixel maps to its nearest axis point.
Colour is averaged across the leg's width (kills cross-wise blur, keeps lengthwise banding),
snapped to a small palette along the axis (crisp segment bands), then re-rendered as a
lit cylinder. Band changes along the axis become ink joint lines across the leg.
"""
import cv2, numpy as np
from scipy import ndimage as ndi
from skimage.morphology import medial_axis
from skimage.measure import label as sklabel

def _blur(x, s): return cv2.GaussianBlur(x, (0, 0), s) if s > 0 else x
def _ss(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-8), 0, 1); return t * t * (3 - 2 * t)

def limb_tube(rgb, M, S, ld, k_max=5, seed=0):
    """rgb float [0,1] HxWx3, M hard mask. Returns (tube_rgb, limb_weight, joint_ink, info)."""
    h, w = M.shape
    m = M > 0.5
    sk, dist = medial_axis(m, return_distance=True)
    # prune tiny spurs: keep skeleton where radius is meaningful
    sk &= dist > 0.004 * S
    if sk.sum() < 10:
        z = np.zeros((h, w), np.float32); return rgb.copy(), z, z, {}
    # nearest axis point for every pixel
    d_ax, (iy, ix) = ndi.distance_transform_edt(~sk, return_indices=True)
    r = dist[iy, ix]                                   # local half-width
    # limbs = thin parts. threshold from the radius distribution along the axis (bimodal: legs vs body)
    rs = dist[sk]
    tau = min(0.06 * S, max(0.02 * S, np.percentile(rs, 60) * 1.6))
    limb_ax = sk & (dist < tau)
    limbw = (_ss(tau * 1.15, tau * 0.85, r) * m).astype(np.float32)
    limbw = _blur(limbw, 0.004 * S + 0.5) * m
    # axis id per pixel
    ax_id = np.zeros((h, w), np.int32); ax_id[sk] = np.arange(1, sk.sum() + 1)
    pid = ax_id[iy, ix]
    n = sk.sum()
    lab = cv2.cvtColor(rgb.astype(np.float32), cv2.COLOR_RGB2Lab)
    # cross-section mean colour (core of the leg weighted more: edges are blurred into background)
    v = np.clip(d_ax / (r + 1e-6), 0, 1)
    wgt = (m & (v < 0.55)).astype(np.float32) * (1 - v)
    idx = np.arange(1, n + 1)
    cs = np.stack([ndi.sum(lab[..., c] * wgt, pid, idx) for c in range(3)], 1)
    cw = ndi.sum(wgt, pid, idx)[:, None] + 1e-6
    axcol = cs / cw                                    # n x 3 Lab per axis pixel
    # smooth along the axis (masked blur on the skeleton image)
    img = np.zeros((h, w, 3), np.float32); img[sk] = axcol
    s1 = 0.004 * S + 0.6
    num = _blur(img * sk[..., None], s1); den = _blur(sk.astype(np.float32), s1)[..., None] + 1e-6
    axs = (num / den)[sk]
    # palette over limb axis colours: k-means in Lab, k chosen by spread
    L = axs[limb_ax[sk]]
    if len(L) > 20:
        spread = np.sqrt(L.var(0).sum())
        k = int(np.clip(round(spread / 5), 3, k_max + 1))
        crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
        _, lbl, cent = cv2.kmeans(L.astype(np.float32), k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
        dd = ((axs[:, None, :] - cent[None]) ** 2).sum(2)
        snap_id = dd.argmin(1)
        # median-ish cleanup along axis: majority vote in a small window on the skeleton image
        vote = np.zeros((h, w, k), np.float32)
        for j in range(k):
            t = np.zeros((h, w), np.float32); t[sk] = (snap_id == j); vote[..., j] = _blur(t, 0.006 * S + 0.8)
        snap_id = vote[sk].argmax(1)
        snapped = cent[snap_id]
        ax_final = 0.55 * snapped + 0.45 * axs
    else:
        snap_id = np.zeros(n, int); ax_final = axs
    # paint the tube: colour from axis point, cylinder shading by signed cross position
    colL = np.zeros((h, w, 3), np.float32)
    colL[m] = ax_final[pid[m] - 1]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    side = ((xx - ix) * ld[0] + (yy - iy) * ld[1]) / (d_ax + 1e-6)   # +1 lit side, -1 shadow side
    side = np.where(d_ax < 0.5, 0, side)
    shade = 1 - 0.42 * _ss(0.25, 1.0, v) * (0.55 + 0.45 * np.clip(-side, 0, 1))
    hl = np.exp(-((v - 0.42) / 0.16) ** 2) * np.clip(side, 0, 1)
    colL[..., 0] = colL[..., 0] * shade + 22 * hl * (1 - colL[..., 0] / 110)
    colL[..., 1:] *= 1.35                              # averaging across the leg dilutes chroma; restore it
    tube = np.clip(cv2.cvtColor(colL, cv2.COLOR_Lab2RGB), 0, 1)
    # joint ink: where the snapped band changes along a limb axis -> short line across the leg
    sid_img = np.full((h, w), -1, np.int32); sid_img[sk] = snap_id
    chg = np.zeros((h, w), bool)
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        a = sid_img; b = np.roll(np.roll(sid_img, -dy, 0), -dx, 1)
        diff = np.zeros((h, w), np.float32)
        ok = (a >= 0) & (b >= 0) & (a != b)
        if ok.any() and len(L) > 20:
            diff[ok] = np.linalg.norm(cent[a[ok]] - cent[b[ok]], axis=1)
        chg |= ok & (diff > 14)
    chg &= limb_ax
    # thin out: one mark per band boundary
    lab_c, nc = ndi.label(chg, structure=np.ones((3, 3)))
    joint = np.zeros((h, w), np.uint8)
    if nc:
        cy, cx = np.array(ndi.center_of_mass(chg, lab_c, range(1, nc + 1))).T
        # local axis direction from skeleton neighbourhood (structure tensor of skeleton)
        skf = _blur(sk.astype(np.float32), 1.5)
        gx = cv2.Sobel(skf, cv2.CV_32F, 1, 0); gy = cv2.Sobel(skf, cv2.CV_32F, 0, 1)
        E, F, G = _blur(gx * gx, 3), _blur(gx * gy, 3), _blur(gy * gy, 3)
        lw = max(1, int(round(0.0022 * S + 0.4)))
        for yv, xv in zip(cy, cx):
            yi, xi = int(round(yv)), int(round(xv))
            e, f, g = E[yi, xi], F[yi, xi], G[yi, xi]
            th = 0.5 * np.arctan2(2 * f, e - g)        # dominant gradient dir = across the axis
            rr = dist[yi, xi] if sk[yi, xi] else r[yi, xi]
            dx_, dy_ = np.cos(th) * rr * 0.95, np.sin(th) * rr * 0.95
            # slight curve: draw as 3-point polyline bowed along the axis
            bow = 0.25 * rr
            p = np.array([[xv - dx_, yv - dy_], [xv - np.sin(th) * bow, yv + np.cos(th) * bow], [xv + dx_, yv + dy_]])
            cv2.polylines(joint, [(p * 8).astype(np.int32)], False, 255, lw, cv2.LINE_AA, 3)
    joint = (joint.astype(np.float32) / 255) * m * (limbw > 0.3)
    # adaptive: how much of each leg's real detail would the band-snapping throw away?
    # residual = colour difference between the photo's leg core and its snapped band colour.
    # Banded legs (Cytaea-style rings) have low residual -> redraw fully; finely mottled legs have high
    # residual -> keep the painted texture and only add the outline and joints.
    resid = np.linalg.norm(lab - colL_flat_store(ax_final, pid, m, h, w), axis=2)
    core = m & (v < 0.6) & (limbw > 0.3)
    lbl, nl = ndi.label(limbw > 0.3)
    scale = np.ones((h, w), np.float32); scores = []
    if nl:
        idx = np.arange(1, nl + 1)
        rmean = ndi.mean(np.where(core, resid, 0), lbl, idx) / (ndi.mean(core.astype(np.float32), lbl, idx) + 1e-6)
        wleg = np.clip((16.0 - rmean) / (16.0 - 8.0), 0.15, 1.0)      # 8 -> full redraw, 16+ -> mostly painted
        scale = np.concatenate([[1.0], wleg]).astype(np.float32)[lbl]
        scores = list(np.round(rmean, 1))
    limbw = limbw * _blur(scale, 0.01 * S + 1)
    return tube, limbw, joint, {'tau': tau, 'k': int(snap_id.max() + 1), 'leg_residual': scores}

def colL_flat_store(ax_final, pid, m, h, w):
    out = np.zeros((h, w, 3), np.float32); out[m] = ax_final[pid[m] - 1]; return out
