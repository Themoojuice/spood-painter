"""Spider cutout.

Primary: BiRefNet (lite, General-2K weights, MIT) in PyTorch, two passes:
  1) whole frame -> locate the spider
  2) re-run on a crop around it, so thin legs are several times larger in the
     network's input (the single biggest win for fine setae and leg tips).
Fallback: ISNet ONNX (rembg isnet-general-use) if torch / BiRefNet is unavailable.
"""
import os, sys, time
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BIREFNET_REPO = os.environ.get('BIREFNET_REPO', '/home/claude/zhengpeng7/birefnet')
BIREFNET_W = os.path.join(HERE, 'models/birefnet_lite2k.pth')
ISNET_W = os.path.join(HERE, 'models/isnet.onnx')
MEAN, STD = np.float32([0.485, 0.456, 0.406]), np.float32([0.229, 0.224, 0.225])

_M = None
def _birefnet():
    global _M
    if _M is None:
        os.environ.setdefault('BIREFNET_BB', '6')      # swin_v1_t  (lite)
        os.environ.setdefault('BIREFNET_TASK', '4')    # General-2K
        sys.path.insert(0, BIREFNET_REPO)
        import torch
        torch.set_num_threads(os.cpu_count())
        from models.birefnet import BiRefNet
        m = BiRefNet(bb_pretrained=False)
        sd = torch.load(BIREFNET_W, map_location='cpu', weights_only=True)
        sd = {k.replace('module.', '').replace('_orig_mod.', '').replace('squeeze_0.', 'squeeze_module.0.'): v for k, v in sd.items()}
        m.load_state_dict(sd, strict=True)
        _M = m.eval()
    return _M

def _run_birefnet(rgb, long_side):
    import torch
    h, w = rgb.shape[:2]
    s = long_side / max(h, w)
    # Swin-T needs feature maps that tile cleanly into 7x7 windows at every stage: some sizes that are
    # merely /32 (e.g. 1024x640) silently return an empty mask. Multiples of 224 = 32*7 are safe.
    q = 224
    W, H = max(q, int(round(w * s / q)) * q), max(q, int(round(h * s / q)) * q)
    x = (cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 - MEAN) / STD
    with torch.inference_mode():
        y = _birefnet()(torch.from_numpy(x.transpose(2, 0, 1)[None].copy()))[-1].sigmoid()[0, 0].numpy()
    return cv2.resize(y, (w, h), interpolation=cv2.INTER_LINEAR)

def alpha_birefnet(bgr, long_side=1120, refine=True):
    rgb = bgr[..., ::-1]
    h, w = rgb.shape[:2]
    a = _run_birefnet(rgb, long_side)
    if not refine: return a
    ys, xs = np.nonzero(a > 0.3)
    if len(ys) < 50: return a
    bw, bh = xs.max() - xs.min(), ys.max() - ys.min()
    pad = 0.25 * max(bw, bh)
    x0, x1 = int(max(0, xs.min() - pad)), int(min(w, xs.max() + pad))
    y0, y1 = int(max(0, ys.min() - pad)), int(min(h, ys.max() + pad))
    if (x1 - x0) * (y1 - y0) > 0.7 * w * h: return a       # already fills the frame
    a2 = _run_birefnet(rgb[y0:y1, x0:x1], long_side)
    out = a * 0.0
    out[y0:y1, x0:x1] = a2
    return out

_S = None
def alpha_isnet(bgr):
    global _S
    import onnxruntime as ort
    if _S is None:
        o = ort.SessionOptions(); o.intra_op_num_threads = os.cpu_count()
        _S = ort.InferenceSession(ISNET_W, o)
    h, w = bgr.shape[:2]
    x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 - 0.5
    y = _S.run(None, {'input_image': x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
    y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    return cv2.resize(y, (w, h), interpolation=cv2.INTER_LINEAR).clip(0, 1)

def fuse(a_b, a_i):
    """BiRefNet is sharper on legs and setae but occasionally drops the whole body (dark spider on a
    warm ground) or returns almost nothing. ISNet is blurrier but rarely misses the body.
    Rule: trust BiRefNet, add ISNet's confident core, fall back to ISNet if BiRefNet lost the animal."""
    nb, ni = (a_b > 0.5).sum(), (a_i > 0.5).sum()
    if ni > 0 and nb < 0.5 * ni:
        return a_i
    # BiRefNet also likes other salient things (leaf midrib, a stalk). Gate it to the neighbourhood of
    # anything ISNet thought was spider, even faintly; ISNet is reliable about *where* the animal is.
    S = np.sqrt(max(ni, 1))
    k = int(0.045 * S) | 1
    support = cv2.dilate((a_i > 0.2).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    support = cv2.GaussianBlur(support.astype(np.float32), (0, 0), 0.02 * S + 1)
    fused = np.maximum(a_b * np.clip(support * 1.5, 0, 1), np.where(a_i > 0.85, a_i, 0))
    return prune_spurs(fused, a_i)

def prune_spurs(a, a_i):
    """Remove stalks / leaf midribs: parts only BiRefNet believes in (ISNet < 0.5) that run to the
    frame edge and reach far from the spider's core. Real leg tips beyond ISNet's core are short;
    a midrib is long, straight and leaves the frame."""
    from scipy import ndimage as ndi
    h, w = a.shape
    core = a_i > 0.5
    if core.sum() < 50: return a
    S = np.sqrt(core.sum())
    extra = (a > 0.5) & ~core
    lab, n = ndi.label(extra)
    if n == 0: return a
    dcore = ndi.distance_transform_edt(~core)
    border = np.zeros((h, w), bool); border[:2] = border[-2:] = True; border[:, :2] = border[:, -2:] = True
    out = a.copy()
    for i in range(1, n + 1):
        comp = lab == i
        reach = dcore[comp].max()
        touches = (comp & border).any()
        ys, xs = np.nonzero(comp)
        if len(ys) < 20: continue
        c = np.cov(np.stack([xs, ys]).astype(np.float32)); ev = np.sort(np.linalg.eigvalsh(c))
        elong = np.sqrt(ev[1] / (ev[0] + 1e-6))
        if (touches and reach > 0.25 * S) or (reach > 0.6 * S and elong > 4):
            out[comp] = np.minimum(out[comp], a_i[comp])
    return out

def alpha(bgr):
    a_i = alpha_isnet(bgr)
    try:
        a_b = alpha_birefnet(bgr)
    except Exception as e:
        print('  BiRefNet unavailable (%s) -> ISNet only' % e)
        return a_i
    return fuse(a_b, a_i)

if __name__ == '__main__':
    # python cutout.py OUTDIR img...   -> OUTDIR/name.png (fused) plus raw/ copies of both nets for re-fusing
    out = sys.argv[1]; os.makedirs(os.path.join(out, 'raw'), exist_ok=True)
    for f in sys.argv[2:]:
        t = time.time(); im = cv2.imread(f); n = os.path.splitext(os.path.basename(f))[0]
        a_i = alpha_isnet(im); a_b = alpha_birefnet(im)
        cv2.imwrite(os.path.join(out, 'raw', n + '_isnet.png'), (a_i * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(out, 'raw', n + '_birefnet.png'), (a_b * 255).astype(np.uint8))
        cv2.imwrite(os.path.join(out, n + '.png'), (fuse(a_b, a_i) * 255).astype(np.uint8))
        print(n, round(time.time() - t, 1), 's', flush=True)
