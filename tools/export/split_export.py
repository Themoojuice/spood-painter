"""Export BiRefNet lite as 3 sequential ONNX parts so each part's peak memory fits the 4 GB
WebAssembly heap (running the whole graph in one go needs ~4.4 GB there at 1120 px).
  enc_full : image(1120)          -> f1..f4       Swin-T on the full image
  enc_half : image(1120)          -> h1..h4       Swin-T on the half-size image (BiRefNet's multi-scale input)
  dec      : image, f1..f4, h1..h4 -> alpha      concat, squeeze block, decoder (low-memory tail), sigmoid
Usage: python split_export.py SIZE
"""
import sys, os
import torch
import torch.nn.functional as F
import load_birefnet as L, lowmem, chunked_conv
from models.birefnet import image2patches


class EncFull(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x): return tuple(s.m.bb(x))


class EncHalf(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x):
        H, W = x.shape[2:]
        return tuple(s.m.bb(F.interpolate(x, size=(H // 2, W // 2), mode='bilinear', align_corners=True)))


class Dec(torch.nn.Module):
    def __init__(s, m): super().__init__(); s.m = m
    def forward(s, x, f1, f2, f3, f4, h1, h2, h3, h4):
        xs = [torch.cat([f, F.interpolate(h, size=f.shape[2:], mode='bilinear', align_corners=True)], dim=1)
              for f, h in ((f1, h1), (f2, h2), (f3, h3), (f4, h4))]
        x4 = s.m.squeeze_module(xs[3])
        return s.m.decoder([x, xs[0], xs[1], xs[2], x4])[-1].sigmoid()


def build():
    m = L.load(True)
    assert m.config.mul_scl_ipt == 'cat' and m.config.squeeze_block
    lowmem.patch_decoder(m.decoder, image2patches)
    for p in m.parameters(): p.requires_grad_(False)
    return m


if __name__ == '__main__':
    S = int(sys.argv[1]) if len(sys.argv) > 1 else 1120
    m = build(); chunked_conv.enable()
    x = torch.randn(1, 3, S, S)
    with torch.no_grad():
        f = EncFull(m)(x); h = EncHalf(m)(x)
        torch.onnx.export(EncFull(m), (x,), f'bf{S}_enc_full.onnx', input_names=['image'], output_names=['f1', 'f2', 'f3', 'f4'], opset_version=17, dynamo=False)
        torch.onnx.export(EncHalf(m), (x,), f'bf{S}_enc_half.onnx', input_names=['image'], output_names=['h1', 'h2', 'h3', 'h4'], opset_version=17, dynamo=False)
        torch.onnx.export(Dec(m), (x, *f, *h), f'bf{S}_dec.onnx', input_names=['image', 'f1', 'f2', 'f3', 'f4', 'h1', 'h2', 'h3', 'h4'], output_names=['alpha'], opset_version=17, dynamo=False)
    for p in ('enc_full', 'enc_half', 'dec'):
        print(p, os.path.getsize(f'bf{S}_{p}.onnx') / 1e6, 'MB')
