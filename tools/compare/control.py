import cv2, numpy as np, onnxruntime as ort, sys, json
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..', '..', 'py')); import spoodpaint as sp
from analyse_lib import *
out = {}
for k, (ph, pa) in PAIRS.items():
    P = cv2.imread(U + ph + '-image.jpg'); G = cv2.imread(U + pa + '-image.jpg')
    s = G.shape[1] / P.shape[1]; Pd = cv2.resize(P, (G.shape[1], int(round(P.shape[0] * s))), interpolation=cv2.INTER_AREA)
    how = 'mask' if k == 'B' else 'feat'
    gem, Ga, M, n = compare(G, Pd, how)
    # control: spoodpaint (current app settings) on the same photo, then the same alignment + scoring
    a = mask(P).astype(np.float32)
    sp.VIBRANCE, sp.TONE_LIFT, sp.INK_AMOUNT, sp.WARMTH = 1.25, 0.6, 1 - 0.42 * 0.6, 0.6
    R = sp.render(P, a, long_side=2000, seed=7, mode='plate')
    cv2.imwrite(f'{k}_spoodpaint.jpg', R)
    try: ctl, Ca, _, n2 = compare(R, Pd, 'feat', M)
    except Exception as e: ctl, Ca, _, n2 = compare(R, Pd, 'mask', M)
    cv2.imwrite(f'{k}_paint_aligned.jpg', Ga); cv2.imwrite(f'{k}_ctl_aligned.jpg', Ca); cv2.imwrite(f'{k}_photo.jpg', Pd); cv2.imwrite(f'{k}_mask.png', M.astype(np.uint8) * 255)
    out[k] = dict(gemini=gem, spoodpaint=ctl)
    print(f'\n{k}  ({"silhouette" if how == "mask" else str(n) + " feature"} alignment)      Gemini                    spoodpaint (control)')
    print('   band      paint/photo  same-marks  same-places   paint/photo  same-marks  same-places')
    for g_, c_ in zip(gem, ctl):
        print('   %5.2fpx      %5.2fx      %5.2f      %5.2f         %5.2fx      %5.2f      %5.2f' % (g_['band'], g_['ratio'], g_['corr'], g_['where'], c_['ratio'], c_['corr'], c_['where']), flush=True)
json.dump(out, open('control.json', 'w'), indent=1)
