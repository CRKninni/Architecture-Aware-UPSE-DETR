"""Parse Chefer Table 1 metrics from a DETR eval log (IoU 0.20:0.95)."""
import json
import re
import sys

PAPER = {
    "detr_bbox": {"AP": 51.8, "AP_M": 56.3, "AP_L": 67.6, "AR": 67.4, "AR_M": 72.8, "AR_L": 85.1},
    "rollout": {"AP": 0.1, "AP_M": 0.1, "AP_L": 0.2, "AR": 0.4, "AR_M": 0.1, "AR_L": 0.9},
    "raw_attention": {"AP": 5.6, "AP_M": 9.6, "AP_L": 6.9, "AR": 11.7, "AR_M": 21.8, "AR_L": 10.8},
    "gradcam": {"AP": 2.3, "AP_M": 2.3, "AP_L": 4.7, "AR": 5.5, "AR_M": 5.9, "AR_L": 10.7},
    "partial_lrp": {"AP": 4.7, "AP_M": 8.0, "AP_L": 5.1, "AR": 10.4, "AR_M": 19.9, "AR_L": 8.0},
    "trans_attribution": {"AP": 7.2, "AP_M": 10.4, "AP_L": 12.4, "AR": 13.4, "AR_M": 21.0, "AR_L": 19.4},
    "chefer_ours": {"AP": 13.1, "AP_M": 14.4, "AP_L": 24.6, "AR": 19.3, "AR_M": 23.9, "AR_L": 33.2},
}

COLS = ("AP", "AP_M", "AP_L", "AR", "AR_M", "AR_L")


def _pct(x):
    if x is None:
        return None
    return round(float(x) * 100.0, 1)


def parse_block(text, metric):
    """Extract Table-1 fields from one COCOeval summarize() block."""
    def grab(pat):
        m = re.search(pat, text)
        return float(m.group(1)) if m else None

    iou = r"(?:IoU=0\.20:0\.95|IoU=0\.50:0\.95)"
    return {
        "AP": grab(rf"Average Precision\s+\(AP\) @\[ {iou} \| area=\s+all \| maxDets=100 \] = ([0-9.-]+)"),
        "AP_S": grab(rf"Average Precision\s+\(AP\) @\[ {iou} \| area= small \| maxDets=100 \] = ([0-9.-]+)"),
        "AP_M": grab(rf"Average Precision\s+\(AP\) @\[ {iou} \| area=medium \| maxDets=100 \] = ([0-9.-]+)"),
        "AP_L": grab(rf"Average Precision\s+\(AP\) @\[ {iou} \| area= large \| maxDets=100 \] = ([0-9.-]+)"),
        "AR": grab(rf"Average Recall\s+\(AR\) @\[ {iou} \| area=\s+all \| maxDets=100 \] = ([0-9.-]+)"),
        "AR_S": grab(rf"Average Recall\s+\(AR\) @\[ {iou} \| area= small \| maxDets=100 \] = ([0-9.-]+)"),
        "AR_M": grab(rf"Average Recall\s+\(AR\) @\[ {iou} \| area=medium \| maxDets=100 \] = ([0-9.-]+)"),
        "AR_L": grab(rf"Average Recall\s+\(AR\) @\[ {iou} \| area= large \| maxDets=100 \] = ([0-9.-]+)"),
        "metric": metric,
    }


def parse_log(log_path):
    text = open(log_path).read()
    parts = text.split("IoU metric: ")
    segm = bbox = None
    for p in parts[1:]:
        name, body = p.split("\n", 1)
        name = name.strip()
        if name == "segm":
            segm = parse_block(body, "segm")
        elif name == "bbox":
            bbox = parse_block(body, "bbox")
    return segm, bbox


def table1_row(block):
    if not block:
        return None
    return {k: _pct(block[k]) for k in COLS}


def format_row(name, row):
    if row is None:
        return f"{name:22}  (missing)"
    return (
        f"{name:22}  "
        + "  ".join(f"{row[k]:5.1f}" for k in COLS)
    )


def main(log_path, method, out_path):
    segm, bbox = parse_log(log_path)
    our_segm = table1_row(segm)
    our_bbox = table1_row(bbox)
    out = {
        "method": method,
        "protocol": "Chefer Table 1 — COCO val, IoU 0.20:0.95, skip AP_small",
        "units": "percent (same as paper)",
        "log": log_path,
        "paper": PAPER,
        "ours": {
            "bbox": our_bbox,
            "segm": our_segm,
        },
        "delta_vs_chefer_paper": None,
        "delta_vs_chefer_repro": None,
    }
    if our_segm and our_segm["AP"] is not None:
        out["delta_vs_chefer_paper"] = {
            k: round(our_segm[k] - PAPER["chefer_ours"][k], 1) for k in COLS
        }
    json.dump(out, open(out_path, "w"), indent=2)
    print("Chefer Table 1 columns:  AP  AP_M  AP_L  AR  AR_M  AR_L")
    print(format_row("paper DETR bbox", PAPER["detr_bbox"]))
    print(format_row("paper Chefer Ours", PAPER["chefer_ours"]))
    print(format_row(f"ours bbox ({method})", our_bbox))
    print(format_row(f"ours segm ({method})", our_segm))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
