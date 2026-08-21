"""LibraGrad hooks for DETR Chefer backward (encoder/decoder/cross-attn)."""

import torch

from DETR.modules.layers import Add, LayerNorm, MultiheadAttention

_HANDLES = []
_STATE = {"enabled": False, "softmax_retention": 0.18, "cross_boost": 1.12}


def _tag_attention_roles(model):
    enc = model.transformer.encoder.layers
    dec = model.transformer.decoder.layers
    for blk in enc:
        blk.self_attn.attn_role = "enc_self"
    for blk in dec:
        blk.self_attn.attn_role = "dec_self"
        blk.multihead_attn.attn_role = "dec_cross"


def _libra_add_hook(module, grad_input, grad_output):
    return tuple(g * 0.5 if g is not None else None for g in grad_input)


def _libra_layernorm_hook(module, grad_input, grad_output):
    if not grad_output or grad_output[0] is None:
        return grad_input
    g = grad_output[0]
    return tuple(0.5 * g if gi is not None else None for gi in grad_input)


def _wrap_save_attn(module, retention):
    orig = module.save_attn_gradients

    def _wrap(g):
        if g is not None and _STATE["enabled"]:
            role = getattr(module, "attn_role", "")
            if role == "dec_cross":
                g = g * retention * 0.75
            elif role == "enc_self":
                g = g * retention
            else:
                g = g * retention * 0.9
        orig(g)
        return g

    module._libra_orig_save = orig
    module.save_attn_gradients = _wrap


def balance_attention_gradients(model):
    if not _STATE["enabled"]:
        return
    enc_mass, cross_mass = 0.0, 0.0
    enc_mods, cross_mods = [], []
    for mod in model.modules():
        if not isinstance(mod, MultiheadAttention):
            continue
        g = mod.get_attn_gradients()
        if g is None:
            continue
        m = float(g.detach().abs().sum())
        role = getattr(mod, "attn_role", "")
        if role == "dec_cross":
            cross_mass += m
            cross_mods.append(mod)
        elif role == "enc_self":
            enc_mass += m
            enc_mods.append(mod)
    if cross_mass < 1e-12 or enc_mass < 1e-12:
        return
    scale_cross = _STATE["cross_boost"] * float((enc_mass / (cross_mass + 1e-8)) ** 0.5)
    scale_cross = float(max(0.85, min(scale_cross, 2.2)))
    scale_enc = float((cross_mass / (enc_mass + 1e-8)) ** 0.25)
    scale_enc = max(0.55, min(scale_enc, 1.0))
    for mod in cross_mods:
        g = mod.get_attn_gradients()
        if g is not None:
            mod.attn_gradients = g * scale_cross
    for mod in enc_mods:
        g = mod.get_attn_gradients()
        if g is not None:
            mod.attn_gradients = g * scale_enc


def enable_libra_grad_detr(model, softmax_retention=0.18, cross_boost=1.12):
    disable_libra_grad_detr(model)
    _STATE["enabled"] = True
    _STATE["softmax_retention"] = softmax_retention
    _STATE["cross_boost"] = cross_boost
    _tag_attention_roles(model)
    for mod in model.modules():
        if isinstance(mod, MultiheadAttention):
            _wrap_save_attn(mod, softmax_retention)
        if isinstance(mod, Add):
            _HANDLES.append(mod.register_full_backward_hook(_libra_add_hook))
        if isinstance(mod, LayerNorm):
            _HANDLES.append(mod.register_full_backward_hook(_libra_layernorm_hook))


def disable_libra_grad_detr(model):
    _STATE["enabled"] = False
    for h in _HANDLES:
        h.remove()
    _HANDLES.clear()
    for mod in model.modules():
        if isinstance(mod, MultiheadAttention) and hasattr(mod, "_libra_orig_save"):
            mod.save_attn_gradients = mod._libra_orig_save
            del mod._libra_orig_save
