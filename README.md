# UPSE for DETR

Peak-gated LibraGrad + UPSE fusion for DETR object discovery on COCO (Chefer et al. Table 1 protocol).

**Author:** Charan Ramtej Kodi ([CRKninni](https://github.com/CRKninni)) — Research Scholar, University of Hyderabad

## Locked method: UPSE v3

Chefer anchor → LibraGrad signals → peak-gated fuse + background crush. Beats Chefer segm AP on COCO val (IoU 0.20:0.95):

| Segm AP | Ours v3 | Chefer repro |
|---------|---------|--------------|
| AP | **13.2** | 13.1 |
| AP_M | 14.4 | 14.4 |
| AP_L | 25.0 | 24.6 |

Full table: [`results/ap_upse_v3_segm.json`](results/ap_upse_v3_segm.json)

**Evaluation details:** COCO 2017 val segmentation AP/AR — [`docs/EVALUATION.md`](docs/EVALUATION.md) (AOPC applies to VQA repos only).

## Reproduce

```bash
source ~/.virtualenvs/torch_112/bin/activate
bash setup_coco_detr.sh   # once
bash run_upse_ap_v3.sh
```

Requires [Transformer-MM-Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability) DETR code on `PYTHONPATH`.

## Citation

If you use this code, please cite our UPSE paper.
