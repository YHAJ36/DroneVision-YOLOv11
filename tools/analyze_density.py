"""验证"目标越多越只抓得到大目标"这个假设。

做法：在 val 集上把 GT 按【单图目标数】和【目标尺寸】分桶，逐框做 IoU>=0.5 的
同类匹配，统计各桶的召回率。用数据回答，而不是靠感觉。

输出：
  表1  按单图目标密度分桶的召回率
  表2  按目标尺寸分桶的召回率
  表3  密度 × 尺寸 交叉表（看"密集时是否只剩大目标"）
"""
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "ultralytics-cfg"))

import numpy as np  # noqa: E402
from ultralytics import YOLO  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
WDIR = os.path.join(ROOT, "runs", "yolo11s_896", "weights", "best.pt")
IMG = os.path.join(ROOT, "dataset_visdrone/yolo", "images", "val")
LBL = os.path.join(ROOT, "dataset_visdrone/yolo", "labels", "val")
CONF = 0.25
IOU_THR = 0.5

DENSITY = [(0, 20, "<20"), (20, 50, "20~50"), (50, 100, "50~100"),
           (100, 10 ** 9, ">=100")]
# 按 sqrt(area) 像素分桶
SIZES = [(0, 16, "<16px"), (16, 32, "16~32px"), (32, 96, "32~96px"),
         (96, 10 ** 9, ">=96px")]


def bucket(v, table):
    for lo, hi, name in table:
        if lo <= v < hi:
            return name
    return table[-1][2]


def read_gt(path, w, h):
    boxes, cls = [], []
    if not os.path.exists(path):
        return np.zeros((0, 4)), np.zeros((0,), int)
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            p = ln.split()
            if len(p) != 5:
                continue
            c = int(float(p[0]))
            cx, cy, bw, bh = (float(x) for x in p[1:])
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes.append([x1, y1, x2, y2])
            cls.append(c)
    return np.array(boxes, float), np.array(cls, int)


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.clip(x2 - x1, 0, None)
    ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = aa[:, None] + bb[None, :] - inter
    return np.where(union > 0, inter / union, 0)


def main():
    print("加载模型 ...", flush=True)
    model = YOLO(WDIR)
    files = sorted(f for f in os.listdir(IMG) if f.lower().endswith((".jpg", ".png")))
    print("val 图片 %d 张，conf=%.2f，IoU 阈值=%.2f\n" % (len(files), CONF, IOU_THR), flush=True)

    dens_stat = defaultdict(lambda: [0, 0])          # 桶 -> [命中, 总数]
    size_stat = defaultdict(lambda: [0, 0])
    cross = defaultdict(lambda: [0, 0])              # (密度, 尺寸) -> [命中, 总数]
    per_img_gt = []
    per_img_pred = []
    t0 = time.time()

    for i, fn in enumerate(files, 1):
        stem = os.path.splitext(fn)[0]
        p = os.path.join(IMG, fn)
        r = model.predict(source=p, imgsz=896, conf=CONF, verbose=False)[0]
        h, w = r.orig_shape
        gb, gc = read_gt(os.path.join(LBL, stem + ".txt"), w, h)
        n_gt = len(gb)
        per_img_gt.append(n_gt)

        pb = np.zeros((0, 4))
        pc = np.zeros((0,), int)
        if r.boxes is not None and len(r.boxes):
            pb = r.boxes.xyxy.cpu().numpy()
            pc = r.boxes.cls.cpu().numpy().astype(int)
        per_img_pred.append(len(pb))

        if n_gt == 0:
            continue
        dname = bucket(n_gt, DENSITY)
        M = iou_matrix(gb, pb)
        for gi in range(n_gt):
            same = pc == gc[gi]
            best = M[gi][same].max() if same.any() else 0.0
            hit = 1 if best >= IOU_THR else 0
            bw_ = gb[gi, 2] - gb[gi, 0]
            bh_ = gb[gi, 3] - gb[gi, 1]
            side = float(np.sqrt(max(bw_ * bh_, 0)))
            sname = bucket(side, SIZES)
            dens_stat[dname][0] += hit
            dens_stat[dname][1] += 1
            size_stat[sname][0] += hit
            size_stat[sname][1] += 1
            cross[(dname, sname)][0] += hit
            cross[(dname, sname)][1] += 1

        if i % 100 == 0 or i == len(files):
            print("  %d/%d ..." % (i, len(files)), flush=True)

    dt = time.time() - t0
    print("\n完成，耗时 %.1f s\n" % dt)

    def table(title, stat, order):
        print("=" * 62)
        print(title)
        print("=" * 62)
        print("  %-12s %10s %10s %9s" % ("分组", "命中/总数", "召回率", "占全部"))
        print("  " + "-" * 46)
        total = sum(v[1] for v in stat.values())
        for k in order:
            if k not in stat:
                continue
            hit, n = stat[k]
            if n == 0:
                continue
            print("  %-12s %6d/%-5d %9.1f%% %8.1f%%" % (
                k, hit, n, hit * 100.0 / n, n * 100.0 / max(1, total)))
        print()

    table("表1  按单图目标数分桶 → 召回率", dens_stat, [d[2] for d in DENSITY])
    table("表2  按目标尺寸分桶 → 召回率", size_stat, [s[2] for s in SIZES])

    print("=" * 62)
    print("表3  密度 × 尺寸 交叉召回率（验证\"密集时只剩大目标\"）")
    print("=" * 62)
    print("  %-10s" % "密度\\尺寸" + "".join("%12s" % s[2] for s in SIZES))
    print("  " + "-" * 58)
    for d in DENSITY:
        row = "  %-10s" % d[2]
        any_val = False
        for s in SIZES:
            v = cross.get((d[2], s[2]))
            if v and v[1]:
                row += "%11.1f%%" % (v[0] * 100.0 / v[1])
                any_val = True
            else:
                row += "%12s" % "-"
        if any_val:
            print(row)
    print()

    print("=" * 62)
    print("单图目标数 vs 检出数")
    print("=" * 62)
    print("  GT 平均 %.1f 个/图，检出平均 %.1f 个/图（conf=%.2f）" % (
        np.mean(per_img_gt), np.mean(per_img_pred), CONF))
    print("  GT 最多 %d 个，检出最多 %d 个" % (
        int(np.max(per_img_gt)), int(np.max(per_img_pred))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
