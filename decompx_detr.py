"""
DecompX-style margin-drop spatial scores for DETR.

Routes encoder memory to the target query logit via LibraGrad-weighted cross-attn,
then scores each spatial token by predicted margin loss (DecompX `drop` mode).
No full transformer decomposition — fast enough for COCO AP eval.
"""

import numpy as np
import torch

from detr_joint import collect_decoder_cross, cross_row_to_spatial


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _cross_routing_row(model, query_idx, n_spatial, use_lrp=False):
    acc = None
    for grad, cam in collect_decoder_cross(model, use_lrp=use_lrp):
        row = cross_row_to_spatial(grad, cam, query_idx, n_spatial)
        if row is not None:
            acc = row if acc is None else acc + row
    return _norm(acc) if acc is not None else None


def spatial_margin_drop(model, logits, memory, query_idx, class_idx, n_spatial, use_lrp=False):
    """
    Per spatial-token margin-drop score after a LibraGrad backward.

    logits: [num_classes+1] for target query
    memory: encoder output [HW, B, C] or [B, C, H, W]
    """
    if logits is None or memory is None:
        return None

    cross_row = _cross_routing_row(model, query_idx, n_spatial, use_lrp=use_lrp)
    if cross_row is None:
        return None

    if memory.dim() == 4:
        mem = memory[0].flatten(1).t()
    elif memory.dim() == 3:
        mem = memory[:, 0, :]
    else:
        return None

    n = min(mem.shape[0], n_spatial)
    mem = mem[:n].float()
    cross_row = cross_row[:n]

    w = model.class_embed.weight[int(class_idx)].float()
    linear = torch.relu(mem @ w).detach().cpu().numpy()
    contrib = cross_row * linear

    logits_f = logits.detach().float().cpu().numpy()
    t = int(class_idx)
    comp = logits_f.copy()
    comp[t] = -1e9
    margin_full = float(logits_f[t] - comp.max())

    reduced = logits_f[t] - contrib
    comp2 = np.tile(logits_f, (n, 1))
    comp2[:, t] = -1e9
    runner_up = comp2.max(axis=1)
    margin_without = reduced - runner_up
    drop = margin_full - margin_without
    return _norm(np.maximum(drop, 0.0))
