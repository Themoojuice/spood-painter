import cv2, numpy as np, itertools
U = __import__('os').environ.get('PAIRS_DIR', './pairs/')
photos = ['b0cfd05d', '13e360ee', '86b5af57', '6a1bf7f7', '0bc3c588', '2509d584', '723d7273']
paints = ['bf01966d', 'c16dedb7', 'bb647fff', 'a789d20c', '97d066e4', 'f9368a28', '6faef768', '22b99250', '37a76e62', 'cc496533']
sift = cv2.SIFT_create(4000)
def feats(k, W=1000):
    im = cv2.imread(U + k + '-image.jpg', 0); s = W / im.shape[1]
    im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return sift.detectAndCompute(im, None), s
F = {k: feats(k) for k in photos + paints}
bf = cv2.BFMatcher()
for p in paints:
    best = []
    for q in photos:
        (kp1, d1), _ = F[p]; (kp2, d2), _ = F[q]
        m = [a for a, b in bf.knnMatch(d1, d2, k=2) if a.distance < 0.8 * b.distance]
        if len(m) < 8: best.append((0, q)); continue
        A, inl = cv2.estimateAffinePartial2D(np.float32([kp1[a.queryIdx].pt for a in m]), np.float32([kp2[a.trainIdx].pt for a in m]), method=cv2.RANSAC, ransacReprojThreshold=8)
        best.append((int(inl.sum()) if inl is not None else 0, q))
    best.sort(reverse=True); print(p, '->', best[:2])
