"""补充分析：9 字段异常行的长相、越界框规模、空标注图片数。"""
import os
from collections import Counter
from PIL import Image
from pathlib import Path

# 项目根目录：本脚本位于 tools，向上 1 级即项目根，路径一律基于它拼
ROOT = Path(__file__).resolve().parents[1]

SPLITS = {
    "train": str(ROOT / "dataset_visdrone" / "raw_train" / "VisDrone2019-DET-train"),
    "val": str(ROOT / "dataset_visdrone" / "raw_val" / "VisDrone2019-DET-val"),
}

print("=== 1) 字段数 != 8 的异常行抽样 ===")
shown = 0
for split, base in SPLITS.items():
    adir = os.path.join(base, "annotations")
    for fn in sorted(os.listdir(adir)):
        with open(os.path.join(adir, fn), "r", encoding="utf-8", errors="replace") as fh:
            for i, ln in enumerate(fh, 1):
                ln = ln.rstrip("\n")
                if not ln.strip():
                    continue
                p = ln.split(",")
                if len(p) != 8:
                    if shown < 12:
                        print("  [%s] %s 第 %d 行，`%d` 字段: %s" % (
                            split, fn, i, len(p), ln))
                        shown += 1

print("\n=== 2) 越界框与退化框统计（逐图读真实尺寸）===")
tot_boxes = 0
oob_left = oob_top = oob_right = oob_bottom = 0
oob_any = 0
degenerate = 0
empty_after = 0
imgs_total = 0
dims = Counter()

for split, base in SPLITS.items():
    adir = os.path.join(base, "annotations")
    idir = os.path.join(base, "images")
    n_empty = 0
    for fn in sorted(os.listdir(adir)):
        stem = os.path.splitext(fn)[0]
        imgp = os.path.join(idir, stem + ".jpg")
        if not os.path.exists(imgp):
            continue
        with Image.open(imgp) as im:
            W, H = im.size
        dims[(W, H)] += 1
        imgs_total += 1
        kept = 0
        with open(os.path.join(adir, fn), "r", encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                p = ln.split(",")
                if len(p) != 8:
                    continue
                try:
                    x, y, w, h = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
                    cid = int(float(p[5]))
                except ValueError:
                    continue
                if cid == 0 or cid == 11:
                    continue
                tot_boxes += 1
                if w <= 0 or h <= 0:
                    degenerate += 1
                    continue
                x1, y1, x2, y2 = x, y, x + w, y + h
                bad = False
                if x1 < 0: oob_left += 1; bad = True
                if y1 < 0: oob_top += 1; bad = True
                if x2 > W: oob_right += 1; bad = True
                if y2 > H: oob_bottom += 1; bad = True
                if bad: oob_any += 1
                kept += 1
        if kept == 0:
            n_empty += 1
    print("  [%s] 过滤 cid 0/11 后有效框为 0 的图片: %d" % (split, n_empty))
    empty_after += n_empty

print("\n  参与统计图片: %d" % imgs_total)
print("  有效框(cid 1~10)总数: %d" % tot_boxes)
print("  退化框(w/h<=0，需丢弃): %d" % degenerate)
print("  越界框合计: %d  (左%d 上%d 右%d 下%d)" % (
    oob_any, oob_left, oob_top, oob_right, oob_bottom))
print("  越界框占比: %.2f%%" % (oob_any * 100.0 / tot_boxes if tot_boxes else 0))
print("  过滤后有效框为 0 的图片合计: %d" % empty_after)

print("\n=== 3) 图片尺寸全量分布 ===")
for s, c in dims.most_common(20):
    print("  %-14s %5d 张" % ("%dx%d" % s, c))
