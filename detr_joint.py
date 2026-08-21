"""DETR joint-attention helpers (encoder-decoder, spatial tokens)."""

import numpy as np
import torch

from detr_tier1 import (
    attention_flow_tokens,
    glimpse_layer_blend,
    grad_x_input_tokens,
    notice_query_cross,
)


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _head_weighted(grad, cam):
    g = grad.clamp(min=0)
    if g.dim() == 4:
        g, a = g[0], cam[0]
    else:
        a = cam
    w = g.sum(dim=(1, 2))
    w = w / (w.sum() + 1e-8)
    return (g * a * w[:, None, None]).sum(dim=0).clamp(min=0)


def collect_encoder_self(model, use_lrp=True):
    pairs = []
    for blk in model.transformer.encoder.layers:
        grad = blk.self_attn.get_attn_gradients()
        attn = blk.self_attn.get_attn_cam() if use_lrp else blk.self_attn.get_attn()
        if grad is None or attn is None:
            continue
        pairs.append((grad.detach(), attn.detach()))
    return pairs


def collect_decoder_cross(model, use_lrp=True):
    pairs = []
    for blk in model.transformer.decoder.layers:
        grad = blk.multihead_attn.get_attn_gradients()
        attn = blk.multihead_attn.get_attn_cam() if use_lrp else blk.multihead_attn.get_attn()
        if grad is None or attn is None:
            continue
        pairs.append((grad.detach(), attn.detach()))
    return pairs


def collect_decoder_self(model, use_lrp=True):
    pairs = []
    for blk in model.transformer.decoder.layers:
        grad = blk.self_attn.get_attn_gradients()
        attn = blk.self_attn.get_attn_cam() if use_lrp else blk.self_attn.get_attn()
        if grad is None or attn is None:
            continue
        pairs.append((grad.detach(), attn.detach()))
    return pairs


def cross_row_to_spatial(grad, cam, query_idx, n_spatial):
    """Query→image cross-attn row for one object query."""
    r = _head_weighted(grad, cam).cpu().numpy()
    if r.shape[0] <= query_idx:
        return None
    row = r[query_idx]
    if row.shape[0] > n_spatial:
        row = row[:n_spatial]
    elif row.shape[0] < n_spatial:
        pad = np.zeros(n_spatial, dtype=row.dtype)
        pad[: row.shape[0]] = row
        row = pad
    return _norm(np.maximum(row, 0.0))


def encoder_self_to_spatial(pairs, n_spatial):
    acc = None
    for grad, cam in pairs:
        r = _head_weighted(grad, cam).cpu().numpy()
        n = min(r.shape[0], n_spatial)
        blk = r[:n, :n]
        col = np.maximum(blk.sum(axis=0), 0.0)
        acc = col if acc is None else acc + col
    return _norm(acc) if acc is not None else None


def gwcr_spatial(enc_pairs, cross_pairs, query_idx, n_spatial, cross_weight=0.55):
    cross = None
    for grad, cam in cross_pairs:
        row = cross_row_to_spatial(grad, cam, query_idx, n_spatial)
        if row is not None:
            cross = row if cross is None else cross + row
    self_rel = encoder_self_to_spatial(enc_pairs, n_spatial)
    if cross is not None:
        cross = _norm(cross)
    parts, weights = [], []
    if cross is not None:
        parts.append(cross_weight * cross)
        weights.append(cross_weight)
    if self_rel is not None:
        parts.append((1.0 - cross_weight) * self_rel)
        weights.append(1.0 - cross_weight)
    if not parts:
        return None
    return _norm(np.sum(parts, axis=0))


def cross_query_spatial(cross_pairs, query_idx, n_spatial):
    """Raw query→image cross-attn mass (LXMERT cross_raw analog)."""
    acc = None
    for grad, cam in cross_pairs:
        row = cross_row_to_spatial(grad, cam, query_idx, n_spatial)
        if row is not None:
            acc = row if acc is None else acc + row
    return _norm(acc) if acc is not None else None


def grad_masses(enc_pairs, cross_pairs):
    self_m, cross_m = 0.0, 0.0
    for grad, cam in enc_pairs:
        self_m += float(_head_weighted(grad, cam).sum())
    for grad, cam in cross_pairs:
        cross_m += float(_head_weighted(grad, cam).sum())
    return self_m, cross_m


def upse_lite_spatial(enc_pairs, n_spatial):
    """Encoder grad-cam + flow rank-fuse (LXMERT upse_lite analog)."""
    gxi = grad_x_input_tokens(enc_pairs, n_spatial)
    flow = attention_flow_tokens(enc_pairs, n_spatial)
    parts = [p for p in (gxi, flow) if p is not None]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return rank_fuse(parts)


def tier1_spatial(enc_pairs, cross_pairs, query_idx, n_spatial):
    cross_only = []
    for grad, cam in cross_pairs:
        g, a = grad, cam
        if g.dim() == 4:
            g, a = g[0], cam[0]
        cross_only.append((g[:, query_idx : query_idx + 1, :], a[:, query_idx : query_idx + 1, :]))
    return {
        "gxi": grad_x_input_tokens(enc_pairs, n_spatial),
        "flow": attention_flow_tokens(enc_pairs, n_spatial),
        "heads": notice_query_cross(cross_pairs, query_idx, n_spatial),
        "layers": glimpse_layer_blend(enc_pairs, n_spatial),
    }


def rank_fuse(arrays, weights=None):
    arrays = [_norm(a) for a in arrays if a is not None and len(a) > 0]
    if not arrays:
        raise ValueError("rank_fuse needs arrays")
    n = len(arrays[0])
    score = np.zeros(n, dtype=np.float64)
    for i, a in enumerate(arrays):
        ranks = np.argsort(np.argsort(-a[:n]))
        w = float(weights[i]) if weights else 1.0
        score += w * (n - ranks)
    return _norm(score)
