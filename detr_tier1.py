"""Tier-1 spatial branches for DETR encoder/cross-attn tokens."""

import numpy as np
import torch


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _to_2d_cam(grad, cam):
    g = grad.detach().clamp(min=0)
    a = cam.detach()
    if g.dim() == 4:
        g, a = g[0], a[0]
    w = g.sum(dim=(1, 2))
    w = w / (w.sum() + 1e-8)
    r = (g * a * w[:, None, None]).sum(dim=0)
    return torch.clamp(r, min=0)


def grad_x_input_tokens(pairs, n_tokens):
    acc = None
    for grad, cam in pairs:
        r = _to_2d_cam(grad, cam).cpu().numpy()
        n = min(r.shape[0], n_tokens)
        col = np.maximum(r[:n, :n].sum(axis=0), 0.0)
        acc = col if acc is None else acc + col
    return _norm(acc) if acc is not None else None


def attention_flow_tokens(pairs, n_tokens):
    flow = None
    for grad, cam in pairs:
        a = cam.detach().float()
        if a.dim() == 4:
            a = a[0].mean(dim=0)
        elif a.dim() == 3:
            a = a.mean(dim=0)
        n = min(a.shape[0], n_tokens)
        a = a[:n, :n]
        eye = torch.eye(n, device=a.device)
        m = 0.5 * eye + 0.5 * a
        m = m / (m.sum(dim=-1, keepdim=True) + 1e-8)
        flow = m if flow is None else m @ flow
    if flow is None:
        return None
    recv = flow.sum(dim=0).cpu().numpy()
    return _norm(np.maximum(recv[:n_tokens], 0.0))


def notice_query_cross(cross_pairs, query_idx, n_tokens, keep_frac=0.55):
    acc = None
    for grad, cam in cross_pairs:
        g = grad.detach().clamp(min=0)
        a = cam.detach()
        if g.dim() != 4:
            continue
        nh = g.shape[1]
        head_mass = g.abs().sum(dim=(0, 2, 3)).cpu().numpy()
        k = max(1, int(np.ceil(nh * keep_frac)))
        top = np.argsort(-head_mass)[:k]
        row = torch.zeros(g.shape[-1], device=g.device)
        for h in top:
            wh = g[0, h].sum().clamp(min=1e-8)
            row = row + (g[0, h, query_idx] * a[0, h, query_idx]) * (wh / (g[0].sum() + 1e-8))
        flow = row.cpu().numpy()
        if flow.shape[0] > n_tokens:
            flow = flow[:n_tokens]
        flow = np.maximum(flow, 0.0)
        acc = flow if acc is None else acc + flow
    return _norm(acc) if acc is not None else None


def glimpse_layer_blend(pairs, n_tokens, decay=0.82):
    acc, w_sum = None, 0.0
    for i, (grad, cam) in enumerate(pairs):
        w = decay ** (len(pairs) - 1 - i)
        r = _to_2d_cam(grad, cam).cpu().numpy()
        n = min(r.shape[0], n_tokens)
        col = np.maximum(r[:n, :n].sum(axis=0), 0.0)
        acc = w * col if acc is None else acc + w * col
        w_sum += w
    if acc is None:
        return None
    return _norm(acc / max(w_sum, 1e-8))
