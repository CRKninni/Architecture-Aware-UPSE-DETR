"""
Per-sample Information Bottleneck Attribution (IBA) for DETR spatial tokens.

Schulz et al. 2020 — Restricting the Flow: optimize mask λ on encoder memory so
Z = λ·R + (1−λ)·baseline preserves the target logit while minimizing I(Z;R).

Fast post-hoc variant (no retraining):
  • Sigmoid mask α on HW encoder tokens (init α=5 → λ≈1)
  • Adam steps on α only; model frozen
  • Loss = −logit_target + β · Σ(−log λ)   (variational compression proxy)
  • Attribution = normalized −log λ  (bits proxy per token)
"""

import os

import numpy as np
import torch


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _iba_cfg():
    return {
        "steps": int(os.environ.get("UPSE_DETR_IBA_STEPS", "8")),
        "beta": float(os.environ.get("UPSE_DETR_IBA_BETA", "0.01")),
        "lr": float(os.environ.get("UPSE_DETR_IBA_LR", "1.0")),
        "init_alpha": float(os.environ.get("UPSE_DETR_IBA_INIT", "5.0")),
    }


def spatial_iba_bits(model, img, query_idx, class_idx, n_spatial, steps=None, beta=None, lr=None):
    """
    Per-sample IBA on encoder memory → spatial bit map [n_spatial].
    """
    if class_idx is None:
        return None

    cfg = _iba_cfg()
    steps = cfg["steps"] if steps is None else steps
    beta = cfg["beta"] if beta is None else beta
    lr = cfg["lr"] if lr is None else lr

    device = next(model.parameters()).device
    n = int(n_spatial)
    if n <= 0:
        return None
    alpha = torch.full((n,), cfg["init_alpha"], device=device, requires_grad=True)
    opt = torch.optim.Adam([alpha], lr=lr)

    def _inject(_module, inputs):
        inp = list(inputs)
        memory = inp[1]
        if memory is None or memory.ndim < 2:
            return tuple(inp)
        hw = min(int(memory.shape[0]), n)
        if hw <= 0:
            return tuple(inp)
        lam = torch.sigmoid(alpha[:hw]).view(hw, 1, 1)
        base = memory[:hw].mean(dim=0, keepdim=True).expand(hw, -1, -1)
        masked = memory.clone()
        masked[:hw] = lam * memory[:hw] + (1.0 - lam) * base
        inp[1] = masked
        return tuple(inp)

    handle = model.transformer.decoder.register_forward_pre_hook(_inject)
    model.eval()
    try:
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            model.zero_grad(set_to_none=True)
            outputs = model(img)
            logit = outputs["pred_logits"][0, int(query_idx), int(class_idx)]
            lam = torch.sigmoid(alpha)
            kl = -torch.sum(torch.log(lam + 1e-8))
            loss = -logit + beta * kl
            loss.backward()
            opt.step()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

    with torch.no_grad():
        lam = torch.sigmoid(alpha).detach().cpu().numpy()
    bits = -np.log(lam + 1e-8)
    return _norm(bits)
