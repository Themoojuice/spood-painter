import cv2, numpy as np, onnxruntime as ort, json
U = __import__('os').environ.get('PAIRS_DIR', './pairs/')
PAIRS = {'A': ('6a1bf7f7', 'c16dedb7'), 'B': ('0bc3c588', 'bf01966d'), 'C': ('2509d584', 'bb647fff'),
         'D': ('13e360ee', 'a789d20c'), 'E': ('86b5af57', '97d066e4'), 'F': ('723d7273', 'f9368a28')}
isn = ort.InferenceSession(__import__('os').environ.get('ISNET', 'models/isnet.onnx'))
def mask(bgr):
    h, w = bgr.shape[:2]
    x = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (1024, 1024), interpolation=cv2.INTER_AREA).astype(np.float32) / 255 - .5
    y = isn.run(None, {'input_image': x.transpose(2, 0, 1)[None]})[0][0, 0]; y = (y - y.min()) / (y.max() - y.min() + 1e-8)
    return cv2.resize(y, (w, h)) > 0.5
sift = cv2.SIFT_create(6000); bfm = cv2.BFMatcher()
def L(im): return cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32) / 255
def dog(x, s): return cv2.GaussianBlur(x, (0, 0), s) - cv2.GaussianBlur(x, (0, 0), 2 * s)
bands = [0.75, 1.5, 3, 6, 12]

def align_feat(G, Pd):
    k1, d1 = sift.detectAndCompute(cv2.cvtColor(G, cv2.COLOR_BGR2GRAY), None); k2, d2 = sift.detectAndCompute(cv2.cvtColor(Pd, cv2.COLOR_BGR2GRAY), None)
    m = [a for a, b in bfm.knnMatch(d1, d2, k=2) if a.distance < 0.85 * b.distance]
    A, inl = cv2.estimateAffinePartial2D(np.float32([k1[a.queryIdx].pt for a in m]), np.float32([k2[a.trainIdx].pt for a in m]), method=cv2.RANSAC, ransacReprojThreshold=10)
    return A, int(inl.sum())
def align_mask(G, Pd):
    """similarity from silhouette moments, refined by ECC on blurred masks"""
    mg, mp = mask(G).astype(np.float32), mask(Pd).astype(np.float32)
    def mom(m):
        M_ = cv2.moments(m); cx, cy = M_['m10'] / M_['m00'], M_['m01'] / M_['m00']
        return cx, cy, np.sqrt(M_['m00'])
    gx_, gy_, gs = mom(mg); px_, py_, ps = mom(mp); s = ps / gs
    A = np.float32([[s, 0, px_ - s * gx_], [0, s, py_ - s * gy_]])
    W = np.eye(2, 3, dtype=np.float32); W[:] = A
    src = cv2.GaussianBlur(mg, (0, 0), 6); dst = cv2.GaussianBlur(mp, (0, 0), 6)
    try:
        _, W = cv2.findTransformECC(dst, src, W, cv2.MOTION_AFFINE, (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6), None, 5)
    except cv2.error: pass
    return W, -1
def compare(G, Pd, how='feat', M=None):
    A, n = align_feat(G, Pd) if how == 'feat' else align_mask(G, Pd)
    Gw = cv2.warpAffine(G, A, (Pd.shape[1], Pd.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    a8 = (cv2.GaussianBlur(L(Pd), (0, 0), 3) * 255).astype(np.uint8); b8 = (cv2.GaussianBlur(L(Gw), (0, 0), 3) * 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(a8, b8, None, 0.5, 5, 41, 5, 7, 1.5, 0)
    gy, gx = np.mgrid[0:Pd.shape[0], 0:Pd.shape[1]].astype(np.float32)
    Ga = cv2.remap(Gw, gx + flow[..., 0], gy + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    if M is None: M = cv2.erode(mask(Pd).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    lp, lg = L(Pd), L(Ga); row = []
    for s_ in bands:
        bp, bg = dog(lp, s_), dog(lg, s_)
        ep, eg = bp[M].std(), bg[M].std()
        r = np.corrcoef(bp[M], bg[M])[0, 1]
        ep_map, eg_map = cv2.GaussianBlur(np.abs(bp), (0, 0), 4 * s_), cv2.GaussianBlur(np.abs(bg), (0, 0), 4 * s_)
        re = np.corrcoef(ep_map[M], eg_map[M])[0, 1]
        row.append(dict(band=s_, ratio=round(float(eg / ep), 2), corr=round(float(r), 2), where=round(float(re), 2)))
    return row, Ga, M, n
