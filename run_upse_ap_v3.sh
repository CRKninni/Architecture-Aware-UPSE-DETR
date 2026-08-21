#!/usr/bin/env bash
# UPSE v3 AP eval — peak-gated LibraGrad + UPSE + DecompX vs Chefer
set -euo pipefail
source ~/.virtualenvs/torch_112/bin/activate 2>/dev/null || true

REPO="/home/gen/crk/Transformer-MM-Explainability"
UPSE="/home/gen/crk/upse-detr"
export PYTHONPATH="${REPO}:${REPO}/DETR:${UPSE}:${PYTHONPATH:-}"
export UPSE_DETR_FUSION=ap

COCO_PATH="${COCO_PATH:-/home/gen/crk/dataset/coco}"
GPU="${GPU:-2}"
LIMIT="${DETR_EVAL_LIMIT:-0}"
METHOD=upse_detr
RESUME="${RESUME:-https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth}"

CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$GPU" | tr -d ' ')
export CUDA_VISIBLE_DEVICES
LOG="${UPSE}/eval_ap_${METHOD}_v3_smi${GPU}.log"

echo "=== UPSE AP v3 smi-GPU=${GPU} limit=${LIMIT} ===" | tee "$LOG"

DETR_EVAL_LIMIT="$LIMIT" python "${REPO}/DETR/main.py" \
  --coco_path "$COCO_PATH" \
  --eval --masks \
  --resume "$RESUME" \
  --batch_size 1 \
  --num_workers 0 \
  --method "$METHOD" \
  2>&1 | tee -a "$LOG"

python "${UPSE}/parse_chefer_table.py" "${LOG}" "upse_detr_v3" "${UPSE}/ap_detr_upse_v3.json"
