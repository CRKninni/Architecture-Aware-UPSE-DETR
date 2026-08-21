"""
DETR comprehensive perturbation — 4 AUCs (LXMERT/METER protocol).

Image +/-  : mask spatial encoder cells (query→image CAM)
Text +/-   : mask object-query slots (query self-attention row; DETR has no text)

Methods: ours_no_lrp (Chefer), upse_detr (UPSE)
"""

import argparse
import gc
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc
from tqdm import tqdm

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Transformer-MM-Explainability"))
UPSE = os.path.abspath(os.path.dirname(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
DETR_DIR = os.path.join(REPO, "DETR")
if DETR_DIR not in sys.path:
    sys.path.insert(0, DETR_DIR)
if UPSE not in sys.path:
    sys.path.insert(0, UPSE)

import DETR.util.misc as utils
from DETR.datasets import build_dataset
from DETR.models import build_model
from DETR.modules.ExplanationGenerator import Generator
from DETR.util.misc import nested_tensor_from_tensor_list

PERT_STEPS = [0, 0.25, 0.5, 0.75, 0.8, 0.85, 0.9, 0.95, 1]


def _norm_cam(x):
    x = x.detach().float().cpu().flatten()
    mn, mx = x.min(), x.max()
    if (mx - mn) > 1e-12:
        x = (x - mn) / (mx - mn)
    return x


def _select_k(scores, k, keep_least=False):
    if k <= 0 or scores.numel() == 0:
        return torch.tensor([], dtype=torch.long)
    k = min(int(k), scores.numel())
    _, idx = scores.topk(k, largest=not keep_least)
    return idx


def _build_args(coco_path, resume):
    from DETR.main import get_args_parser

    parser = argparse.ArgumentParser(parents=[get_args_parser()])
    args = parser.parse_args([])
    args.coco_path = coco_path
    args.dataset_file = "coco"
    # Build on CPU — criterion/matcher don't need GPU for perturbation eval.
    args.device = "cpu"
    args.resume = resume
    args.eval = True
    args.masks = False
    args.batch_size = 1
    args.num_workers = 0
    return args


class DetrPertModel:
    def __init__(self, coco_path, device="cuda", resume=None):
        self.device = device
        self.args = _build_args(coco_path, resume or "")
        self.model, _, _ = build_model(self.args)
        self.model.to(device)
        self.model.eval()
        if resume:
            if resume.startswith("https"):
                ckpt = torch.hub.load_state_dict_from_url(resume, map_location="cpu", check_hash=True)
            else:
                ckpt = torch.load(resume, map_location="cpu")
            self.model.load_state_dict(ckpt["model"], strict=False)
        self.gen = Generator(self.model)
        self._upse = None
        self.pert_steps = PERT_STEPS

    def _upse_gen(self):
        if self._upse is None:
            from detr_adaptive import UPSEGenerator

            self._upse = UPSEGenerator(self.model)
        return self._upse

    def _pick_query(self, outputs):
        probas = outputs["pred_logits"].softmax(-1)[0, :, :-1]
        conf = probas.max(-1).values
        keep = conf > 0.5
        if keep.any():
            q = int(keep.nonzero()[0].item())
        else:
            q = int(conf.argmax().item())
        target_cls = int(probas[q].argmax().item())
        return q, target_cls

    def _correct(self, outputs, q, target_cls):
        pred = outputs["pred_logits"][0, q, :-1].argmax().item()
        return 1.0 if pred == target_cls else 0.0

    def _spatial_hw(self):
        if hasattr(self.model, "spatial_dim") and self.model.spatial_dim:
            return int(self.model.spatial_dim[0]), int(self.model.spatial_dim[1])
        return None, None

    def generate_cams(self, method, samples, q):
        idx = torch.tensor([q], device=samples.tensors.device)
        with torch.enable_grad():
            if method == "ours_no_lrp":
                g = self.gen
                spatial = g.generate_ours(samples, idx, use_lrp=False)
                query_row = g.R_q_q[q].detach()
            elif method == "upse_detr":
                g = self._upse_gen()
                spatial = g.generate_upse(samples, idx, use_lrp=False)
                query_row = g.R_q_q[q].detach()
            elif method == "ours_with_lrp":
                g = self.gen
                spatial = g.generate_ours(samples, idx, use_lrp=True)
                query_row = g.R_q_q[q].detach()
            else:
                raise ValueError(method)
        return _norm_cam(spatial), _norm_cam(query_row)

    def _mask_spatial(self, samples, cam_spatial, h, w, step, keep_least):
        n = cam_spatial.numel()
        k = int((1 - step) * n)
        idx = _select_k(cam_spatial, k, keep_least=keep_least)
        grid = torch.zeros(n)
        if idx.numel() > 0:
            grid[idx] = 1.0
        grid = grid.reshape(h, w).unsqueeze(0).unsqueeze(0)
        tensor = samples.tensors.clone()
        _, _, H, W = tensor.shape
        m = F.interpolate(grid, size=(H, W), mode="nearest").to(tensor.device)
        mean = tensor.mean(dim=(2, 3), keepdim=True)
        tensor = tensor * m + mean * (1 - m)
        return nested_tensor_from_tensor_list([tensor[0]])

    def _forward_query_mask(self, samples, cam_query, step, keep_least):
        n = cam_query.numel()
        k = int((1 - step) * n)
        idx = _select_k(cam_query, k, keep_least=keep_least)
        orig = self.model.query_embed.weight.data.clone()
        mask = torch.zeros(n, device=orig.device)
        if idx.numel() > 0:
            mask[idx] = 1.0
        self.model.query_embed.weight.data = orig * mask.unsqueeze(1)
        try:
            return self.model(samples)
        finally:
            self.model.query_embed.weight.data = orig

    def perturbation_comprehensive(self, samples, cam_spatial, cam_query, q, target_cls):
        h, w = self._spatial_hw()
        if h is None:
            side = int(round(cam_spatial.numel() ** 0.5))
            h = w = side
        out = {
            "image_positive": [],
            "image_negative": [],
            "text_positive": [],
            "text_negative": [],
        }
        for step in self.pert_steps:
            # Image+: keep least-relevant spatial cells
            s_pos = self._mask_spatial(samples, cam_spatial, h, w, step, keep_least=True)
            out["image_positive"].append(self._correct(self.model(s_pos), q, target_cls))
            # Image−: keep most-relevant spatial cells
            s_neg = self._mask_spatial(samples, cam_spatial, h, w, step, keep_least=False)
            out["image_negative"].append(self._correct(self.model(s_neg), q, target_cls))
            # Query "text"+: keep least-relevant query slots
            out["text_positive"].append(
                self._correct(self._forward_query_mask(samples, cam_query, step, True), q, target_cls)
            )
            # Query "text"−: keep most-relevant query slots
            out["text_negative"].append(
                self._correct(self._forward_query_mask(samples, cam_query, step, False), q, target_cls)
            )
        return out


def generate_maps(method, mp, samples, q):
    return mp.generate_cams(method, samples, q)


def run_eval(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    mp = DetrPertModel(args.coco_path, device=device, resume=args.resume or None)
    dataset = build_dataset(image_set="val", args=mp.args)
    indices = list(range(len(dataset)))
    random.seed(args.seed)
    random.shuffle(indices)
    indices = indices[: args.num_samples]

    accum = {k: np.zeros(len(PERT_STEPS)) for k in
             ("text_positive", "text_negative", "image_positive", "image_negative")}
    processed = 0

    for i in tqdm(indices, desc=args.method):
        img, target = dataset[i]
        samples = nested_tensor_from_tensor_list([img]).to(device)
        with torch.no_grad():
            base = mp.model(samples)
        q, target_cls = mp._pick_query(base)
        try:
            cam_i, cam_t = generate_maps(args.method, mp, samples, q)
            curr = mp.perturbation_comprehensive(samples, cam_i, cam_t, q, target_cls)
            processed += 1
            for k in accum:
                accum[k] += np.array(curr[k], dtype=np.float64)
        except Exception as exc:
            print(f"Skip {i}: {exc}")
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    mean = {k: (accum[k] / max(processed, 1) * 100).tolist() for k in accum}
    return {
        "method": args.method,
        "n_processed": processed,
        "seed": args.seed,
        "note": "text_* = object-query slot perturbation (DETR has no language input)",
        "auc_image_positive": float(auc(PERT_STEPS, mean["image_positive"])),
        "auc_image_negative": float(auc(PERT_STEPS, mean["image_negative"])),
        "auc_text_positive": float(auc(PERT_STEPS, mean["text_positive"])),
        "auc_text_negative": float(auc(PERT_STEPS, mean["text_negative"])),
        "values_image_positive": [round(v, 2) for v in mean["image_positive"]],
        "values_image_negative": [round(v, 2) for v in mean["image_negative"]],
        "values_text_positive": [round(v, 2) for v in mean["text_positive"]],
        "values_text_negative": [round(v, 2) for v in mean["text_negative"]],
        "x_values": PERT_STEPS,
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--coco_path", required=True)
    p.add_argument("--resume", default="https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth")
    p.add_argument("--method", default="upse_detr",
                   choices=["ours_no_lrp", "ours_with_lrp", "upse_detr"])
    p.add_argument("--num-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--output", default="")
    args = p.parse_args()
    if not args.output:
        args.output = f"energy_detr_{args.method}_n{args.num_samples}.json"

    print(f"DETR {args.method} perturbation n={args.num_samples} seed={args.seed}")
    res = run_eval(args)
    with open(args.output, "w") as f:
        json.dump(res, f, indent=2)
    print(
        f"Img+ {res['auc_image_positive']:.2f}  Img- {res['auc_image_negative']:.2f}  "
        f"Txt+ {res['auc_text_positive']:.2f}  Txt- {res['auc_text_negative']:.2f}"
    )
    print(f"Saved {args.output}")
