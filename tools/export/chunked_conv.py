"""Make big convolutions memory-friendly for onnxruntime-web.

The WebAssembly CPU runtime computes a k x k convolution by unrolling its whole input into one
C*k*k*H*W buffer (im2col). In BiRefNet a few decoder convs need 0.3-1 GB for that at 1120 px.
conv(x, W) = sum over channel chunks of conv(x[chunk], W[:, chunk]) is exact, and each chunk's
buffer is small. Applied at export time by patching nn.Conv2d (shapes are static, so the chunk
count is decided during tracing).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

LIMIT_BYTES = 64e6
_orig = nn.Conv2d._conv_forward


def _chunked(self, input, weight, bias):
    kh, kw = weight.shape[2], weight.shape[3]
    if self.groups != 1 or kh * kw == 1 or self.padding_mode != 'zeros':
        return _orig(self, input, weight, bias)
    C = input.shape[1]
    H, W = input.shape[2], input.shape[3]
    sh, sw = self.stride
    need = C * kh * kw * math.ceil(H / sh) * math.ceil(W / sw) * 4
    if need <= LIMIT_BYTES:
        return _orig(self, input, weight, bias)
    n = math.ceil(need / LIMIT_BYTES)
    step = math.ceil(C / n)
    out = None
    for i in range(0, C, step):
        o = F.conv2d(input[:, i:i + step], weight[:, i:i + step], None, self.stride, self.padding, self.dilation)
        out = o if out is None else out + o
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


def enable():
    nn.Conv2d._conv_forward = _chunked


def disable():
    nn.Conv2d._conv_forward = _orig
