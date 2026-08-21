"""
DETR UPSE v3 — beat Chefer on segm AP (CAM → Otsu → mask IoU).

Root cause of v1/v2 loss vs Chefer:
  • LibraGrad modified the Chefer backward anchor
  • Global cross/GWCR fusion flattened Otsu histograms
  • Gamma sharpening broke bimodality

v3 fix:
  Pass 1 — pure Chefer (LibraGrad off) → immutable anchor
  Pass 2 — LibraGrad (mild) backward → collect GWCR, UPSE-lite, tier1, DecompX
  Fuse — peak-gated boosts only on top Chefer quantiles + background compression
  Guard — revert to Chefer if Otsu foreground fraction goes out of range

Modes (UPSE_DETR_FUSION env):
  ap   — default; beat Chefer segm AP
  pert — LXMERT sharp Img+ for perturbation AUC
"""

import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Transformer-MM-Explainability"))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import cv2
import numpy as np
import torch

from DETR.modules.ExplanationGenerator import Generator

from decompx_detr import spatial_margin_drop
from libra_grad_detr import balance_attention_gradients, disable_libra_grad_detr, enable_libra_grad_detr
from detr_joint import (
    collect_decoder_cross,
    collect_encoder_self,
    cross_query_spatial,
    grad_masses,
    gwcr_spatial,
    rank_fuse,
    tier1_spatial,
    upse_lite_spatial,
)

# AP peak-gated fusion (Otsu)
_AP_PEAK_Q = 0.72
_AP_TOP_Q = 0.88
_AP_CONSENSUS_Q = 0.35
_AP_BRANCH_W = 0.62
_AP_DX_W = 0.28
_AP_BG_SCALE = 0.80
_AP_FG_LO = 0.004
_AP_FG_HI = 0.52

# LibraGrad for signal pass (not anchor)
_AP_LIBRA_RETENTION = 0.22
_AP_LIBRA_CROSS_BOOST = 1.0

# Pert fusion (LXMERT v2)
_POS_IMG_W = 0.58
_IMG_PEAK_Q = 0.52
_IMG_TOP_Q = 0.80
_IMG_GWCR_PEAK_W = 0.54
_TIER1_NEG_BLEND = 0.88


def _query_idx(target_index):
    if torch.is_tensor(target_index):
        return int(target_index.reshape(-1)[0].item())
    return int(target_index)


def _norm(x):
    x = np.asarray(x, dtype=np.float64)
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-12:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def _rank_fuse(arrays, weights=None):
    arrays = [_norm(a) for a in arrays if a is not None and len(a) > 0]
    if not arrays:
        raise ValueError("rank_fuse needs at least one array")
    n = len(arrays[0])
    score = np.zeros(n, dtype=np.float64)
    for i, a in enumerate(arrays):
        ranks = np.argsort(np.argsort(-a[:n]))
        w = float(weights[i]) if weights is not None else 1.0
        score += w * (n - ranks)
    return _norm(score)


def _neg_image_fuse(branches):
    branches = [b for b in branches if b is not None]
    if not branches:
        raise ValueError("neg_image_fuse needs branches")
    w = [1.0, 1.35, 1.4, 1.15, 1.2][: len(branches)]
    if len(w) < len(branches):
        w = w + [1.0] * (len(branches) - len(w))
    broad = _norm(np.maximum.reduce(branches))
    consensus = _rank_fuse(branches, weights=w)
    return _norm(np.maximum(broad, consensus))


def _tier1_neg_complement(branches):
    branches = [b for b in branches if b is not None]
    if not branches:
        return None
    w = [1.0, 1.15, 1.25, 1.05][: len(branches)]
    if len(w) < len(branches):
        w = w + [1.0] * (len(branches) - len(w))
    return _rank_fuse(branches, weights=w)


def _sharp_image_imgplus(chefer, neg_stack, gwcr=None):
    c = _norm(chefer)
    n = _norm(neg_stack)
    legacy = np.maximum(n, _POS_IMG_W * c)
    peak_thr = float(np.quantile(c, _IMG_PEAK_Q))
    top_thr = float(np.quantile(c, _IMG_TOP_Q))
    out = c.copy()
    peak = c >= peak_thr
    out[peak] = legacy[peak]
    if gwcr is not None:
        g = _norm(gwcr)
        out[peak] = np.maximum(out[peak], _IMG_GWCR_PEAK_W * g[peak])
    top = c >= top_thr
    out[top] = np.maximum(out[top], legacy[top])
    return _norm(out)


def _otsu_foreground_frac(vec):
    x = np.asarray(vec, dtype=np.float64)
    if x.max() - x.min() < 1e-12:
        return 0.0
    cam = (x - x.min()) / (x.max() - x.min()) * 255.0
    _, th = cv2.threshold(cam.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float((th > 0).mean())


def _ap_beat_chefer_fuse(chefer_vec, gwcr, upse_lite, gxi, heads, dx=None):
    """
    Peak-gated UPSE + DecompX on pure Chefer anchor.
    LibraGrad branches only sharpen top Chefer peaks where consensus agrees.
    """
    c = np.asarray(chefer_vec, dtype=np.float64)
    c_n = _norm(c)
    scale = float(c_n.max()) if c_n.max() > 1e-12 else 1.0
    fused = c_n.copy()

    peak_thr = float(np.quantile(c_n, _AP_PEAK_Q))
    top_thr = float(np.quantile(c_n, _AP_TOP_Q))
    peak = c_n >= peak_thr
    top = c_n >= top_thr

    branches = [gwcr, upse_lite, gxi, heads]
    branches = [b for b in branches if b is not None]
    if branches and peak.any():
        weights = [1.25, 1.15, 1.20, 1.10][: len(branches)]
        consensus = _rank_fuse([_norm(b) for b in branches], weights=weights)
        agree_thr = float(np.quantile(consensus[peak], _AP_CONSENSUS_Q))
        agree = peak & (consensus >= agree_thr)
        fused[agree] = np.maximum(
            fused[agree],
            _AP_BRANCH_W * consensus[agree] + (1.0 - _AP_BRANCH_W) * fused[agree],
        )

    if dx is not None and top.any():
        dx_n = _norm(dx)
        fused[top] = np.maximum(fused[top], fused[top] + _AP_DX_W * dx_n[top])

    fused = np.maximum(fused, c_n)

    med = float(np.median(fused))
    fused[fused < med] *= _AP_BG_SCALE
    fused = _norm(fused) * scale

    chefer_fg = _otsu_foreground_frac(c_n * scale)
    fused_fg = _otsu_foreground_frac(fused)
    if fused_fg > _AP_FG_HI or fused_fg < _AP_FG_LO:
        if _AP_FG_LO <= chefer_fg <= _AP_FG_HI:
            return c_n * scale
    return fused


def _libra_scale(self_m, cross_m):
    return float(np.clip(np.sqrt(self_m / (cross_m + 1e-8)), 0.65, 2.0))


def _fusion_mode():
    return os.environ.get("UPSE_DETR_FUSION", "ap").strip().lower()


class UPSEGenerator(Generator):
    """UPSE for DETR — v3 AP fusion targets Chefer segm AP."""

    def _collect_signals(self, model, use_lrp, q, n_spatial):
        enc_pairs = collect_encoder_self(model, use_lrp=use_lrp)
        cross_pairs = collect_decoder_cross(model, use_lrp=use_lrp)
        if not cross_pairs:
            return None

        self_m, cross_m = grad_masses(enc_pairs, cross_pairs)
        cross_w = float(np.clip(cross_m / (self_m + cross_m + 1e-8), 0.30, 0.70))

        gwcr = gwcr_spatial(enc_pairs, cross_pairs, q, n_spatial, cross_weight=cross_w)
        cross_raw = cross_query_spatial(cross_pairs, q, n_spatial)
        upse_lite = upse_lite_spatial(enc_pairs, n_spatial)
        t1 = tier1_spatial(enc_pairs, cross_pairs, q, n_spatial)
        return gwcr, cross_raw, upse_lite, t1, enc_pairs, cross_pairs

    def _signal_backward(self, img, target_index, index, kwargs, mem_holder, logits_holder):
        """Pass 2: LibraGrad backward for branch gradients (not used as CAM anchor)."""
        enable_libra_grad_detr(
            self.model,
            softmax_retention=_AP_LIBRA_RETENTION,
            cross_boost=_AP_LIBRA_CROSS_BOOST,
        )
        hook = self.model.transformer.encoder.register_forward_hook(
            lambda m, i, o: mem_holder.__setitem__("memory", o.detach())
        )
        try:
            outputs = self.model(img)
            logits = outputs["pred_logits"]
            q = _query_idx(target_index)
            if index is None:
                index = logits[0, q, :-1].max(0)[1]
            logits_holder["logits"] = logits[0, q].detach()
            logits_holder["class_idx"] = int(index.item() if torch.is_tensor(index) else index)

            one_hot = torch.zeros_like(logits).to(logits.device)
            one_hot[0, q, logits_holder["class_idx"]] = 1
            one_hot_v = one_hot.clone()
            one_hot.requires_grad_(True)
            loss = torch.sum(one_hot * logits)
            self.model.zero_grad()
            loss.backward(retain_graph=True)
            if kwargs.get("use_lrp", False):
                self.model.relprop(
                    one_hot_v,
                    alpha=1,
                    target_index=target_index,
                    target_class=logits_holder["class_idx"],
                )
            balance_attention_gradients(self.model)
        finally:
            hook.remove()
            disable_libra_grad_detr(self.model)

    def generate_upse(
        self,
        img,
        target_index,
        index=None,
        use_lrp=False,
        normalize_self_attention=True,
        apply_self_in_rule_10=True,
    ):
        q = _query_idx(target_index)
        kwargs = dict(
            index=index,
            use_lrp=use_lrp,
            normalize_self_attention=normalize_self_attention,
            apply_self_in_rule_10=apply_self_in_rule_10,
        )
        mode = _fusion_mode()

        if mode == "pert":
            disable_libra_grad_detr(self.model)
            chefer = self.generate_ours(img, target_index, **kwargs)
            chefer_vec = chefer.reshape(-1).detach().cpu().numpy()
            n_spatial = int(chefer_vec.shape[0])

            mem_holder, logits_holder = {}, {}
            self._signal_backward(img, target_index, index, kwargs, mem_holder, logits_holder)
            sig = self._collect_signals(self.model, use_lrp, q, n_spatial)
            if sig is None:
                return chefer
            gwcr, cross_raw, upse_lite, t1, enc_pairs, cross_pairs = sig
            libra = _libra_scale(*grad_masses(enc_pairs, cross_pairs))
            cross_img = _norm(cross_raw * libra) if cross_raw is not None else None
            tier1 = _tier1_neg_complement(
                [t1.get("gxi"), t1.get("flow"), t1.get("heads"), t1.get("layers")]
            )
            base_neg = _neg_image_fuse([gwcr, upse_lite, cross_img, _norm(chefer_vec)])
            neg_img = base_neg if tier1 is None else _norm(
                np.maximum(base_neg, _TIER1_NEG_BLEND * tier1)
            )
            fused = _sharp_image_imgplus(chefer_vec, neg_img, gwcr=gwcr)
        else:
            # Pass 1: pure Chefer anchor (identical to ours_no_lrp)
            disable_libra_grad_detr(self.model)
            chefer = self.generate_ours(img, target_index, **kwargs)
            chefer_vec = chefer.reshape(-1).detach().cpu().numpy()
            n_spatial = int(chefer_vec.shape[0])

            # Pass 2: LibraGrad + UPSE + DecompX signals
            mem_holder, logits_holder = {}, {}
            self._signal_backward(img, target_index, index, kwargs, mem_holder, logits_holder)
            sig = self._collect_signals(self.model, use_lrp, q, n_spatial)
            if sig is None:
                return chefer

            gwcr, _, upse_lite, t1, _, _ = sig
            dx = spatial_margin_drop(
                self.model,
                logits_holder.get("logits"),
                mem_holder.get("memory"),
                q,
                logits_holder.get("class_idx"),
                n_spatial,
                use_lrp=use_lrp,
            )
            fused = _ap_beat_chefer_fuse(
                chefer_vec,
                gwcr,
                upse_lite,
                t1.get("gxi"),
                t1.get("heads"),
                dx=dx,
            )

        out = torch.tensor(fused, dtype=chefer.dtype, device=chefer.device)
        return out.reshape_as(chefer)
