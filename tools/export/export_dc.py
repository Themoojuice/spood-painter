"""Export BiRefNet lite using the standard ONNX DeformConv op (opset 19) for torchvision's
deform_conv2d, plus the low-memory decoder tail. Optional split into encoder/decoder parts.
usage: python export_dc.py SIZE [split]"""
import sys, os
import torch
import torch.nn.functional as F
from torch.onnx import symbolic_helper as sh
import load_birefnet as L, lowmem
from models.birefnet import image2patches


@sh.parse_args('v', 'v', 'v', 'v', 'v', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'i', 'b')
def deform_sym(g, inp, weight, offset, mask, bias, sh_, sw_, ph, pw, dh, dw, gw, go, use_mask):
    args = [inp, weight, offset, bias] + ([mask] if use_mask else [])
    return g.op('DeformConv', *args, strides_i=[sh_, sw_], pads_i=[ph, pw, ph, pw], dilations_i=[dh, dw],
                group_i=gw, offset_group_i=go)


torch.onnx.register_custom_op_symbolic('torchvision::deform_conv2d', deform_sym, 19)


def build():
    m = L.load(portable=False)
    lowmem.patch_decoder(m.decoder, image2patches)
    for p in m.parameters(): p.requires_grad_(False)
    return m


if __name__ == '__main__':
    S = int(sys.argv[1]); split = len(sys.argv) > 2
    m = build(); x = torch.randn(1, 3, S, S)
    with torch.no_grad():
        if not split:
            torch.onnx.export(L.Wrapped(m), (x,), f'bdc_{S}.onnx', input_names=['image'], output_names=['alpha'],
                              opset_version=19, dynamo=False)
            print('exported', os.path.getsize(f'bdc_{S}.onnx') / 1e6, 'MB')
        else:
            import split_export as SE
            f = SE.EncFull(m)(x); h = SE.EncHalf(m)(x)
            torch.onnx.export(SE.Dec(m), (x, *f, *h), f'bdc{S}_dec.onnx',
                              input_names=['image', 'f1', 'f2', 'f3', 'f4', 'h1', 'h2', 'h3', 'h4'], output_names=['alpha'],
                              opset_version=19, dynamo=False)
            print('exported dec', os.path.getsize(f'bdc{S}_dec.onnx') / 1e6, 'MB')
