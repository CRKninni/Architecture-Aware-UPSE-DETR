"""UPSE spectral core for DETR spatial tokens (DSM, PCA-0, LOST seed)."""

import math
import sys
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from sklearn.decomposition import PCA

# Reuse object_discovery helpers from METER repo when available.
METER = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "upse-meter"))
if METER not in sys.path:
    sys.path.insert(0, METER)

try:
    from object_discovery import row_sum as _row_sum
except ImportError:
    from pymatting.util.util import row_sum as _row_sum


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def memory_to_feats(memory, n_spatial):
    """Encoder memory [HW,B,C] or [B,C,H,W] -> [n_spatial, C]."""
    if memory is None:
        return None
    if memory.dim() == 4:
        mem = memory[0].flatten(1).t()
    elif memory.dim() == 3:
        mem = memory[:, 0, :]
    else:
        return None
    n = min(mem.shape[0], n_spatial)
    return mem[:n].float().detach()


def _diagonal(w):
    d = _row_sum(w)
    d[d < 1e-12] = 1.0
    return diags(d)


def _fiedler(feats, how_many=5):
    feats = F.normalize(feats, p=2, dim=-1)
    w = (feats @ feats.T).clamp(min=0)
    w = (w / (w.max() + 1e-8)).cpu().numpy()
    d = np.array(_diagonal(w).todense())
    lap = d - w
    k = min(how_many, lap.shape[0] - 2)
    if k < 1:
        return np.ones(feats.shape[0], dtype=np.float64)
    try:
        _, vecs = eigs(lap, k=k, which="LM", sigma=-0.5, M=d)
    except Exception:
        try:
            _, vecs = eigs(lap, k=k, which="LM", sigma=-0.5)
        except Exception:
            _, vecs = eigs(lap, k=k, which="LM")
    vecs = np.abs(vecs.T.real)
    fev = vecs[1] if vecs.shape[0] > 1 else vecs[0]
    return _norm(fev)


def _pca0(feats):
    x = feats.cpu().numpy()
    k = min(5, x.shape[0], x.shape[1])
    if k < 1:
        return np.ones(x.shape[0], dtype=np.float64)
    pc = PCA(n_components=k).fit_transform(x)
    col = np.abs(pc[:, 0])
    return _norm(col)


def _lost_seed(feats, iters=15, threshold=0.5):
    """LOST-style seed expansion on Fiedler map."""
    base = _fiedler(feats)
    n = len(base)
    side = int(math.sqrt(n))
    if side * side != n:
        return base
    heat = base.reshape(side, side)
    idx = int(np.argmax(base))
    sr, sc = idx // side, idx % side
    mask = np.zeros_like(heat)
    mask[sr, sc] = 1.0
    for _ in range(iters):
        dil = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1) - mask
        for r, c in zip(*np.where(dil > 0)):
            if heat[r, c] >= threshold:
                mask[r, c] = 1.0
    return _norm((heat * mask).flatten())


def rank_fuse(arrays, weights=None):
    arrays = [_norm(a) for a in arrays if a is not None and len(a) > 0]
    if not arrays:
        return None
    n = len(arrays[0])
    score = np.zeros(n, dtype=np.float64)
    for i, a in enumerate(arrays):
        ranks = np.argsort(np.argsort(-a[:n]))
        w = float(weights[i]) if weights is not None else 1.0
        score += w * (n - ranks)
    return _norm(score)


def spatial_upse_core(memory, n_spatial, use_dsm=True, use_pca=True, use_seed=True):
    """DSM + PCA-0 + LOST seed rank-fuse on encoder memory."""
    feats = memory_to_feats(memory, n_spatial)
    if feats is None:
        return None, {}
    parts, labels = [], []
    comps = {}
    if use_dsm:
        dsm = _fiedler(feats)
        dsm = dsm[:n_spatial] if len(dsm) >= n_spatial else np.pad(dsm, (0, n_spatial - len(dsm)))
        comps["dsm"] = dsm
        parts.append(dsm)
        labels.append("dsm")
    if use_pca:
        pca0 = _pca0(feats)
        pca0 = pca0[:n_spatial] if len(pca0) >= n_spatial else np.pad(pca0, (0, n_spatial - len(pca0)))
        comps["pca"] = pca0
        parts.append(pca0)
        labels.append("pca")
    if use_seed:
        seed = _lost_seed(feats)
        seed = seed[:n_spatial] if len(seed) >= n_spatial else np.pad(seed, (0, n_spatial - len(seed)))
        comps["seed"] = seed
        parts.append(seed)
        labels.append("seed")
    if not parts:
        return None, comps
    w = {"dsm": 1.25, "pca": 1.10, "seed": 1.20}
    weights = [w.get(lb, 1.0) for lb in labels]
    return rank_fuse(parts, weights=weights), comps
