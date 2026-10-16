"""Low-memory BiRefNet decoder tail, algebraically equivalent to the original.

Original last stage (all at full input resolution, ~1.5 GB of activations at 1120 px):
    up  = bilinear_upsample(decoder_block1(_p1))            # 96 ch, full res
    ipt = conv3x3(64->24)(conv3x3(3->64)(image))            # 64 + 24 ch, full res, no nonlinearity
    out = conv1x1(120->1)(cat(up, ipt))

Both branches are linear, so:
    conv1x1 on `up` commutes with the per-channel bilinear upsample -> do the 96->1 mix at 1/4 res;
    conv3x3 o conv3x3 o (1x1 slice) collapses to one 5x5 conv 3->1.
Result: only 1-channel maps at full resolution. Identical except a 1-px frame at the image border,
where the original zero-pads the 64-channel intermediate.
"""
import torch
import torch.nn.functional as F


def collapse_ipt(conv1, conv2, wb):
    """conv2(conv1(x)) followed by 1x1 weights wb (24,) -> single 5x5 conv (1,3,5,5) + bias."""
    W1, b1 = conv1.weight, conv1.bias            # (64,3,3,3), (64,)
    W2, b2 = conv2.weight, conv2.bias            # (24,64,3,3), (24,)
    W2b = torch.einsum('o,omij->mij', wb, W2)    # (64,3,3)
    b2b = (wb * b2).sum()
    K = torch.zeros(1, W1.shape[1], 5, 5, dtype=W1.dtype)
    for qy in range(3):
        for qx in range(3):
            for sy in range(3):
                for sx in range(3):
                    K[0, :, qy + sy, qx + sx] += torch.einsum('m,mc->c', W2b[:, qy, qx], W1[:, :, sy, sx])
    bias = (b1[:, None, None] * W2b).sum() + b2b
    return K, bias


def patch_decoder(decoder, image2patches):
    """Replace decoder.forward (inference path) with the low-memory tail."""
    C_ipt = decoder.ipt_blk1.conv_out.out_channels
    W = decoder.conv_out1[0].weight.detach()[0, :, 0, 0]          # (120,)
    b_out = decoder.conv_out1[0].bias.detach()[0]
    Wa, Wb = W[:-C_ipt], W[-C_ipt:]
    K, b_ipt = collapse_ipt(decoder.ipt_blk1.conv1, decoder.ipt_blk1.conv_out, Wb.detach())
    decoder.register_buffer('lm_Wa', Wa.view(1, -1, 1, 1).clone())
    decoder.register_buffer('lm_K', K.detach().clone())
    decoder.register_buffer('lm_b', (b_ipt + b_out).detach().view(1).clone())

    def forward(features):
        self = decoder
        x, x1, x2, x3, x4 = features
        patches_batch = image2patches(x, patch_ref=x4, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
        x4 = torch.cat((x4, self.ipt_blk5(F.interpolate(patches_batch, size=x4.shape[2:], mode='bilinear', align_corners=True))), 1)
        p4 = self.decoder_block4(x4)
        p4 = p4 * self.gdt_convs_attn_4(self.gdt_convs_4(p4)).sigmoid()
        _p4 = F.interpolate(p4, size=x3.shape[2:], mode='bilinear', align_corners=True)
        _p3 = _p4 + self.lateral_block4(x3)
        patches_batch = image2patches(x, patch_ref=_p3, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
        _p3 = torch.cat((_p3, self.ipt_blk4(F.interpolate(patches_batch, size=x3.shape[2:], mode='bilinear', align_corners=True))), 1)
        p3 = self.decoder_block3(_p3)
        p3 = p3 * self.gdt_convs_attn_3(self.gdt_convs_3(p3)).sigmoid()
        _p3 = F.interpolate(p3, size=x2.shape[2:], mode='bilinear', align_corners=True)
        _p2 = _p3 + self.lateral_block3(x2)
        patches_batch = image2patches(x, patch_ref=_p2, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
        _p2 = torch.cat((_p2, self.ipt_blk3(F.interpolate(patches_batch, size=x2.shape[2:], mode='bilinear', align_corners=True))), 1)
        p2 = self.decoder_block2(_p2)
        p2 = p2 * self.gdt_convs_attn_2(self.gdt_convs_2(p2)).sigmoid()
        _p2 = F.interpolate(p2, size=x1.shape[2:], mode='bilinear', align_corners=True)
        _p1 = _p2 + self.lateral_block2(x1)
        patches_batch = image2patches(x, patch_ref=_p1, transformation='b c (hg h) (wg w) -> b (c hg wg) h w') if self.split else x
        _p1 = torch.cat((_p1, self.ipt_blk2(F.interpolate(patches_batch, size=x1.shape[2:], mode='bilinear', align_corners=True))), 1)
        p1_low = self.decoder_block1(_p1)
        # ---- low-memory tail
        y_low = F.conv2d(p1_low, self.lm_Wa)                                       # 96 -> 1 at 1/4 res
        y_up = F.interpolate(y_low, size=x.shape[2:], mode='bilinear', align_corners=True)
        y_ipt = F.conv2d(x, self.lm_K, padding=2)                                  # collapsed image branch
        return [y_up + y_ipt + self.lm_b.view(1, 1, 1, 1)]

    decoder.forward = forward
    return decoder
