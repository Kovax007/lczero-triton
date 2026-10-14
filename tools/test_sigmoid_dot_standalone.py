#!/usr/bin/env python3
"""Standalone check of `bt4/kernels/sigmoid_dot.py` (backend 10-13): the Triton kernel against torch on random FP16 operands.

    output[row] = sigmoid(sum_k x[row, k] * w[k] + b)   -- FP32 accumulate, FP32 output
for the row counts of every batch rung the lab programs use. Prints max |kernel - torch(float64)| per row count; exits 1
above 2e-6 (FP32 rounding of a 128-term dot product and one exp). CONTROL: the same kernel with the weight column
negated must differ. Needs one idle GPU (CUDA_VISIBLE_DEVICES), no numpy.
"""
import sys

import torch

from lczero_triton.bt4.kernels.sigmoid_dot import _sigmoid_dot_kernel

torch.manual_seed(20261013)
worst_all, bad = 0.0, 0
for rows in (1, 8, 16, 32, 64, 84, 128):
    width = 128
    x = (torch.randn((rows, width), device="cuda") * 1.5).to(torch.float16)
    w = (torch.randn((width, 1), device="cuda") * 0.3).to(torch.float16)
    b = torch.tensor([-2.0], device="cuda").to(torch.float16)
    out = torch.full((rows,), -7.0, dtype=torch.float32, device="cuda")
    _sigmoid_dot_kernel[(rows,)](out, x, w, b, rows, width)
    want = torch.sigmoid(x.double() @ w.double()[:, 0] + b.double())
    worst = float((out.double() - want).abs().max())
    ctl = torch.full((rows,), -7.0, dtype=torch.float32, device="cuda")
    _sigmoid_dot_kernel[(rows,)](ctl, x, (-w.float()).to(torch.float16), b, rows, width)
    control = float((ctl.double() - want).abs().max())
    spread = (float(want.min()), float(want.max()))
    ok = worst <= 2e-6 and control > 1e-3 and bool(torch.isfinite(out).all())
    print(f"rows {rows:4d}: max |kernel - torch float64| {worst:.3e}  (values {spread[0]:.4f} .. {spread[1]:.4f})  "
          f"control (negated weights) max diff {control:.3e}  {'ok' if ok else 'FAIL'}")
    worst_all = max(worst_all, worst)
    bad += not ok
# saturation: a huge logit must give exactly 0 / 1, never NaN
x = torch.full((2, 128), 30.0, device="cuda").to(torch.float16)
w = torch.full((128, 1), 1.0, device="cuda").to(torch.float16)
x[1] = -x[1]
out = torch.empty((2,), dtype=torch.float32, device="cuda")
_sigmoid_dot_kernel[(2,)](out, x, w, torch.zeros(1, device="cuda").to(torch.float16), 2, 128)
sat = out.tolist()
print(f"saturation (logit +3840 / -3840): {sat}")
bad += sat != [1.0, 0.0]
print("SIGMOID_DOT OK" if not bad else f"SIGMOID_DOT FAILED ({bad})", f"worst {worst_all:.3e}")
sys.exit(1 if bad else 0)
