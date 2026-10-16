"""Drop-in replacement for torchvision.ops.deform_conv2d (stride/pad/dilation, one offset group,
modulated) built from grid_sample + 1x1 convs, so BiRefNet exports to ONNX ops that
onnxruntime-web supports. Loops over kernel points and accumulates, which keeps memory at
one sampled feature map at a time instead of C*K channels."""
import torch
import torch.nn.functional as F


def deform_conv2d_portable(input, offset, weight, bias=None, stride=(1, 1), padding=(0, 0),
                           dilation=(1, 1), mask=None):
    if isinstance(stride, int): stride = (stride, stride)
    if isinstance(padding, int): padding = (padding, padding)
    if isinstance(dilation, int): dilation = (dilation, dilation)
    N, C, H, W = input.shape
    O, Cw, kh, kw = weight.shape
    Ho, Wo = offset.shape[2], offset.shape[3]
    dt = input.dtype
    ys = (torch.arange(Ho, dtype=dt, device=input.device) * stride[0] - padding[0]).view(1, Ho, 1)
    xs = (torch.arange(Wo, dtype=dt, device=input.device) * stride[1] - padding[1]).view(1, 1, Wo)
    sy, sx = 2.0 / max(H - 1, 1), 2.0 / max(W - 1, 1)
    out = None
    for i in range(kh):
        for j in range(kw):
            k = i * kw + j
            py = ys + i * dilation[0] + offset[:, 2 * k]          # (N, Ho, Wo)
            px = xs + j * dilation[1] + offset[:, 2 * k + 1]
            grid = torch.stack([px * sx - 1.0, py * sy - 1.0], dim=-1)
            v = F.grid_sample(input, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            if mask is not None:
                v = v * mask[:, k:k + 1]
            t = F.conv2d(v, weight[:, :, i:i + 1, j:j + 1])
            out = t if out is None else out + t
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out
