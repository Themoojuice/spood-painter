import os, sys, torch
HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault('BIREFNET_BB', '6')     # swin_v1_t (lite)
os.environ.setdefault('BIREFNET_TASK', '4')   # General-2K
sys.path.insert(0, os.path.join(HERE, '..', 'BiRefNet'))
sys.path.insert(0, HERE)


def load(portable=True):
    import models.modules.deform_conv as dc
    if portable:
        from deform_portable import deform_conv2d_portable
        dc.deform_conv2d = deform_conv2d_portable
    from models.birefnet import BiRefNet
    m = BiRefNet(bb_pretrained=False)
    sd = torch.load(os.path.join(HERE, '..', 'birefnet_lite2k.pth'), map_location='cpu', weights_only=True)
    sd = {k.replace('module.', '').replace('_orig_mod.', '').replace('squeeze_0.', 'squeeze_module.0.'): v
          for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    return m.eval()


class Wrapped(torch.nn.Module):
    """image (1,3,H,W) normalised -> probability (1,1,H,W)"""
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        return self.m(x)[-1].sigmoid()
