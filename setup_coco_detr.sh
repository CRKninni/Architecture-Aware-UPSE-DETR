#!/usr/bin/env bash
# Prepare COCO 2017 layout for Chefer DETR eval (val2017 + instances_val2017.json)
set -euo pipefail

ROOT="${1:-/home/gen/crk/dataset/coco}"
mkdir -p "$ROOT/annotations"

if [[ -f "$ROOT/annotations/instances_val2017.json" && -d "$ROOT/val2017" ]]; then
  echo "COCO 2017 val already present at $ROOT"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [[ ! -f "$ROOT/annotations/instances_val2017.json" ]]; then
  echo "Downloading COCO 2017 annotations zip ..."
  wget -q -O "$TMP/ann.zip" \
    http://images.cocodataset.org/annotations/annotations_trainval2017.zip
  unzip -q -j "$TMP/ann.zip" "annotations/instances_val2017.json" -d "$ROOT/annotations"
fi

if [[ ! -d "$ROOT/val2017" ]]; then
  echo "Downloading val2017 images (~778MB) ..."
  wget -q -O "$TMP/val2017.zip" http://images.cocodataset.org/zips/val2017.zip
  unzip -q "$TMP/val2017.zip" -d "$ROOT"
fi

echo "Ready: $ROOT/{val2017,annotations/instances_val2017.json}"
