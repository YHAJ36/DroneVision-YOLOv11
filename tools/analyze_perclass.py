"""per-class 尺寸分布 + 召回率，用于给"哪几类实际可用"下准确结论。

不靠 AP 排名猜，直接测：每一类的目标实际多大、召回多少。
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "ultralytics-cfg"))

import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
W = os.path.join(ROOT, "runs", "yolo11s_896", "weights", "best.pt")
IMG = os.path.join(ROOT, "dataset_visdrone/yolo", "images", "val")
LBL = os.path.join(ROOT, "dataset_visdrone/yolo", "labels", "val")
CONF, IOU_THR = 0.25, 0.5

NAMES = ["pedestrian", "people", "bicycle", "car", "van",
         "truck", "tricycle", "awning-tricycle", "bus", "motor"]
# 之前 val.py 实测的 AP@0.5，用于并列展示
AP50 = {"pedestrian": 0.551, "people": 0.415, "bicycle": 0.246, "car": 0.843,
        "van": 0.516, "truck": 0.451, "tricycle": 0.352,
        "awning-tricycle": 0.173, "bus": 0.639, "motor": 0.562}


def read_gt(path, w, h):
    b, c = [], []
    if not os.path.exists(path):
        return np.zeros((0, 4)), np.zeros((0,), int)
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            p = ln.split()
            if len(p) != 5:
                continue
            cid = int(float(p[0]))
            cx, cy, bw, bh = (float(x) for x in p[1:])
            b.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                      (cx + bw / 2) * w, (cy + bh / 2) * h])
            c.append(cid)
    return np.array(b, float), np.array(c, int)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    u = aa[:, None] + bb[None, :] - inter
    return np.where(u > 0, inter / u, 0)


def main():
    model = YOLO(W)
    files = sorted(f for f in os.listdir(IMG) if f.lower().endswith((".jpg", ".png")))
    hit = defaultdict(int)
    tot = defaultdict(int)
    sides = defaultdict(list)

    for i, fn in enumerate(files, 1):
        stem = os.path.splitext(fn)[0]
        r = model.predict(source=os.path.join(IMG, fn), imgsz=896, conf=CONF, verbose=False)[0]
        h, w = r.orig_shape
        gb, gc = read_gt(os.path.join(LBL, stem + ".txt"), w, h)
        if len(gb) == 0:
            continue
        pb = r.boxes.xyxy.cpu().numpy() if (r.boxes is not None and len(r.boxes)) else np.zeros((0, 4))
        pc = r.boxes.cls.cpu().numpy().astype(int) if (r.boxes is not None and len(r.boxes)) else np.zeros((0,), int)
        M = iou_matrix(gb, pb)
        for gi in range(len(gb)):
            c = gc[gi]
            side = float(np.sqrt(max((gb[gi, 2] - gb[gi, 0]) * (gb[gi, 3] - gb[gi, 1]), 0)))
            sides[c].append(side)
            tot[c] += 1
            same = pc == c
            best = M[gi][same].max() if same.any() else 0.0
            if best >= IOU_THR:
                hit[c] += 1
        if i % 150 == 0:
            print("  %d/%d" % (i, len(files)), flush=True)

    rows = []
    for c in range(10):
        n = tot.get(c, 0)
        if n == 0:
            continue
        s = np.array(sides[c])
        rows.append((NAMES[c], n, float(np.median(s)), float(np.mean(s < 16) * 100),
                     hit[c] * 100.0 / n, AP50.get(NAMES[c], 0)))
    rows.sort(key=lambda x: -x[4])

    print()
    print("=" * 84)
    print("per-class：尺寸 vs 召回 vs AP（按召回率降序）")
    print("=" * 84)
    print("  %-17s %7s %9s %10s %9s %8s %s" % (
        "类别", "GT框数", "尺寸中位", "<16px占比", "召回率", "AP@0.5", "判定"))
    print("  " + "-" * 76)
    for name, n, med, small, rec, ap in rows:
        if rec >= 70:
            verdict = "可用"
        elif rec >= 55:
            verdict = "勉强"
        else:
            verdict = "不可用"
        print("  %-17s %7d %8.1fpx %9.1f%% %8.1f%% %8.3f  %s" % (
            name, n, med, small, rec, ap, verdict))
    print()
    print("  说明：尺寸中位数 = sqrt(宽×高) 的像素值；判据按召回率 >=70/>=55/<55。")


if __name__ == "__main__":
    sys.exit(main())
