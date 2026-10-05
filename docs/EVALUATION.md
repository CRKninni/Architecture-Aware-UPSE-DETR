# Evaluation metrics (DETR)

This repository does **not** report VQA **AOPC**. DETR is evaluated on **weakly supervised instance segmentation** following Chefer et al. (Table 1).

## Primary metrics (reported)

**File:** [`results/ap_upse_v3_segm.json`](../results/ap_upse_v3_segm.json)

| Metric | Meaning |
|--------|---------|
| **AP**, **AP_M**, **AP_L** | COCO segmentation average precision (IoU **0.20 : 0.95**) |
| **AR**, **AR_M**, **AR_L** | Average recall |

- **Data:** COCO **2017 validation**, **`n = 5000`** images (fixed sampling seed in eval scripts — see `setup_coco_detr.sh`).
- **Maps:** attribution → query-level masks → binary segmentation; scored against COCO instances.
- **Locked method:** UPSE v3 (`run_upse_ap_v3.sh`, `detr_adaptive_v3_frozen.py`).

## AOPC (LXMERT / VisualBERT only)

**AOPC** is defined on VQA **confidence** under combined text/image deletion and insertion. It applies to:

- [Architecture-Aware-UPSE-LXMERT](https://github.com/CRKninni/Architecture-Aware-UPSE-LXMERT) — see `docs/EVALUATION_AOPC.md`
- [Architecture-Aware-UPSE-VisualBERT](https://github.com/CRKninni/Architecture-Aware-UPSE-VisualBERT) — see `docs/EVALUATION_AOPC.md`

DETR explanations are judged by **localization AP/AR**, not AOPC.

## Reproduce segmentation AP

```bash
source ~/.virtualenvs/torch_112/bin/activate
bash setup_coco_detr.sh   # once — COCO 2017 val
bash run_upse_ap_v3.sh
```

Requires [Transformer-MM-Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability) on `PYTHONPATH`.
