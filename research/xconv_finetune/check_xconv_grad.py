"""Gradient sanity check for the vendored in-house pyxconv (luqigroup/xconv_pv).

Validates, on realistically-sized feature maps (so the circular-roll boundary
artifact does not dominate):
  * forward output is exact (XConv forward IS a real conv);
  * the probed weight-gradient is unbiased in direction, i.e. the sample-mean of
    dw over many probe draws aligns with the exact gradient (cos -> 1);
  * the new 2D grouped/depthwise path produces the correct weight shape and a
    correctly-aligned gradient (this is the in-house addition over stock);
  * 3D grouped conv is *not* supported (documented limitation).

Grouped/depthwise needs xmode='independent' (the gaussian/orthogonal probes
collapse the channel axis that the grouped code slices on).
"""
from __future__ import annotations

import os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pyxconv

torch.manual_seed(0)
torch.set_num_threads(4)


def check(dims, ci, co, k, sp, groups, ps, mode, N, stride=1):
    pad = k // 2
    convF = F.conv2d if dims == 2 else F.conv3d
    Layer = pyxconv.Xconv2D if dims == 2 else pyxconv.Xconv3D
    X = torch.randn(1, ci, *sp)
    try:
        layer = Layer(ci, co, k, stride=stride, padding=pad, bias=False,
                      groups=groups, ps=ps, mode=mode)
    except Exception as e:  # construction issue
        print(f"{dims}D g={groups:<2} s={stride} ps={ps} {mode:<11} | CONSTRUCT ERROR: {type(e).__name__}: {e}")
        return
    W = layer.weight.detach().clone()
    try:
        Y = convF(X, W, bias=None, stride=stride, padding=pad, groups=groups)
        G = torch.ones_like(Y)
        Wl = W.clone().requires_grad_(True)
        (convF(X, Wl, bias=None, stride=stride, padding=pad, groups=groups) * G).sum().backward()
        dWe = Wl.grad.detach(); nrm = dWe.norm()
        run = torch.zeros_like(dWe); fwd_ok = True
        for _ in range(N):
            layer.zero_grad(set_to_none=True)
            Yx = layer(X)
            fwd_ok = fwd_ok and torch.allclose(Yx.detach(), Y, atol=1e-4)
            (Yx * G).sum().backward()
            run += layer.weight.grad.detach()
        mg = run / N
        shape_ok = tuple(mg.shape) == tuple(W.shape)
        cos = F.cosine_similarity(mg.flatten(), dWe.flatten(), dim=0).item()
        kind = "depthwise" if groups == ci == co else ("grouped" if groups > 1 else "dense")
        print(f"{dims}D g={groups:<2} s={stride} ({kind:<9}) sp={str(tuple(sp)):<14} ps={ps:<4} "
              f"{mode:<11} N={N:<4}| fwd_exact={fwd_ok} | wshape={tuple(W.shape)} ok={shape_ok} | "
              f"cos(mean,exact)={cos:7.4f}")
    except Exception as e:
        print(f"{dims}D g={groups:<2} s={stride} ps={ps} {mode:<11} | BACKWARD ERROR: {type(e).__name__}: {e}")


print("In-house pyxconv gradient validation (cos -> 1 means unbiased direction)\n")
print("--- 2D dense (groups=1) ---")
check(2, 8, 8, 3, (48, 48), 1, 64, "independent", N=300)

print("\n--- 2D grouped / depthwise (the in-house addition) ---")
check(2, 8, 8, 3, (48, 48), 2, 64, "independent", N=300)   # grouped
check(2, 8, 8, 3, (48, 48), 8, 64, "independent", N=300)   # depthwise (groups=ci=co)

print("\n--- 2D / 3D strided (stride-2) -- the down-path convs, previously untested ---")
check(2, 8, 8, 3, (48, 48), 1, 64, "independent", N=300, stride=2)
check(3, 4, 4, 3, (24, 24, 24), 1, 256, "independent", N=150, stride=2)

print("\n--- 3D dense across SegResNet's down-path map sizes (boundary bias grows as maps shrink) ---")
for sp in [(48, 48, 48), (24, 24, 24), (12, 12, 12), (8, 8, 8)]:
    check(3, 4, 4, 3, sp, 1, 256, "independent", N=120)

print("\n--- 3D grouped (expected: unsupported in the weight-grad probe) ---")
check(3, 4, 4, 3, (24, 24, 24), 2, 256, "independent", N=50)
