"""VisDrone2019-DET -> YOLO 格式转换

关键处理（均针对实际数据勘察结果）：
  1. 丢弃 class_id 0（ignored regions）与 class_id 11（others，全部 score=0）
     —— 实测 score=0 只出现在这两类上，class 1~10 全为 score=1，
        因此"丢 0/11"与"丢所有 score=0"等价，不会误伤有效目标。
  2. 容错解析：容忍行尾多余逗号（9999974_00000_d_0000053.txt 有 34 行这种）。
  3. 逐图读取真实尺寸归一化（实测图片有 11 种不同分辨率，不可假设固定值）。
  4. 框按图片边界裁剪；裁剪后退化的框（宽或高 <= 0）丢弃。
  5. 空标注文件保留（作为纯背景图）。
"""
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

# 项目根目录：本脚本位于 tools，向上 1 级即项目根，路径一律基于它拼
ROOT = Path(__file__).resolve().parents[1]


from PIL import Image

SRC = {
    "train": str(ROOT / "dataset_visdrone" / "raw_train" / "VisDrone2019-DET-train"),
    "val": str(ROOT / "dataset_visdrone" / "raw_val" / "VisDrone2019-DET-val"),
}
DST = str(ROOT / "dataset_visdrone" / "yolo")

DROP_CLASSES = {0, 11}
CLASS_MAP = {i: i - 1 for i in range(1, 11)}   # 1->0, 2->1, ... 10->9

CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

YAML = "path: " + ROOT.as_posix() + "/dataset_visdrone/yolo" + """
train: images/train
val: images/val

nc: 10
names:
  0: pedestrian
  1: people
  2: bicycle
  3: car
  4: van
  5: truck
  6: tricycle
  7: awning-tricycle
  8: bus
  9: motor
"""


def parse_line(line):
    """解析一行 VisDrone 标注。返回 (x, y, w, h, cid) 或 None。"""
    parts = line.strip().split(",")
    # 容忍行尾多余的逗号/空字段
    while parts and parts[-1].strip() == "":
        parts.pop()
    if len(parts) < 8:
        return None
    try:
        x = float(parts[0])
        y = float(parts[1])
        w = float(parts[2])
        h = float(parts[3])
        score = parts[4].strip()
        cid = int(float(parts[5]))
    except (ValueError, IndexError):
        return None
    return x, y, w, h, cid, score


def convert_split(split):
    base = SRC[split]
    adir = os.path.join(base, "annotations")
    idir = os.path.join(base, "images")

    dst_img = os.path.join(DST, "images", split)
    dst_lbl = os.path.join(DST, "labels", split)
    os.makedirs(dst_img, exist_ok=True)
    os.makedirs(dst_lbl, exist_ok=True)

    stats = Counter()
    per_class = Counter()
    total_boxes = 0
    names = sorted(os.listdir(adir))

    for i, fn in enumerate(names, 1):
        stem = os.path.splitext(fn)[0]
        imgp = os.path.join(idir, stem + ".jpg")
        if not os.path.exists(imgp):
            stats["图片缺失"] += 1
            continue

        with Image.open(imgp) as im:
            W, H = im.size
        if W <= 0 or H <= 0:
            stats["尺寸非法"] += 1
            continue

        out_lines = []
        with open(os.path.join(adir, fn), "r", encoding="utf-8", errors="replace") as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                stats["原始行"] += 1
                parsed = parse_line(raw)
                if parsed is None:
                    stats["解析失败"] += 1
                    continue
                x, y, w, h, cid, score = parsed
                stats["解析成功"] += 1

                # ---- score 护栏（自检"丢 0/11 ≡ 丢所有 score=0"这个等价假设）----
                # 该假设此前是用外部脚本(analyze_annots.py)验证的：score=0 只出现在
                # class 0/11 上。这里内建自检，一旦出现"类有效但 score=0"就计数，
                # 非零即说明等价假设被破坏，需要人工复核。
                if score == "0" and cid not in DROP_CLASSES:
                    stats["★score0_但类有效"] += 1

                if cid in DROP_CLASSES:
                    stats["丢弃_忽略类(%d)" % cid] += 1
                    continue
                if cid not in CLASS_MAP:
                    stats["丢弃_未知类(%d)" % cid] += 1
                    continue

                if w <= 0 or h <= 0:
                    stats["丢弃_退化框"] += 1
                    continue

                # 裁剪到图片边界
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                cx1, cy1 = max(0.0, x1), max(0.0, y1)
                cx2, cy2 = min(float(W), x2), min(float(H), y2)
                if (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
                    stats["已裁剪"] += 1
                bw, bh = cx2 - cx1, cy2 - cy1
                if bw <= 0 or bh <= 0:
                    stats["丢弃_裁剪后退化"] += 1
                    continue

                c = CLASS_MAP[cid]
                ccx = (cx1 + cx2) / 2.0 / W
                ccy = (cy1 + cy2) / 2.0 / H
                nw = bw / W
                nh = bh / H

                # 归一化后必须落在 (0,1)
                if not (0.0 < ccx < 1.0 and 0.0 < ccy < 1.0 and 0.0 < nw <= 1.0 and 0.0 < nh <= 1.0):
                    stats["丢弃_越界归一化"] += 1
                    continue

                out_lines.append("%d %.6f %.6f %.6f %.6f" % (c, ccx, ccy, nw, nh))
                per_class[c] += 1
                total_boxes += 1

        with open(os.path.join(dst_lbl, stem + ".txt"), "w", encoding="utf-8") as fh:
            if out_lines:
                fh.write("\n".join(out_lines) + "\n")
        shutil.copy2(imgp, os.path.join(dst_img, stem + ".jpg"))

        stats["输出图片"] += 1
        if not out_lines:
            stats["空标注图片"] += 1

        if i % 1000 == 0 or i == len(names):
            print("    [%s] %d/%d ..." % (split, i, len(names)), flush=True)

    return stats, per_class, total_boxes


def main():
    if os.path.exists(DST):
        print("[!] 输出目录已存在，将覆盖其中的同名文件: %s" % DST)
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(DST, *sub.split("/")), exist_ok=True)

    grand = Counter()
    all_per_class = Counter()
    summary = {}

    for split in ("train", "val"):
        print("[转换] %s" % split, flush=True)
        stats, per_class, total = convert_split(split)
        grand.update(stats)
        all_per_class.update(per_class)
        summary[split] = (stats["输出图片"], stats["空标注图片"], total, stats)
        print("    完成：图片 %d，标注框 %d，空标注图 %d" % (
            stats["输出图片"], total, stats["空标注图片"]), flush=True)

    with open(os.path.join(DST, "VisDrone.yaml"), "w", encoding="utf-8") as fh:
        fh.write(YAML)

    # 写一份转换报告，便于复核
    with open(os.path.join(DST, "convert_report.txt"), "w", encoding="utf-8") as fh:
        fh.write("VisDrone -> YOLO 转换报告\n" + "=" * 60 + "\n\n")
        for split in ("train", "val"):
            n_img, n_empty, n_box, st = summary[split]
            fh.write("[%s] 输出图片 %d，空标注 %d，有效框 %d\n" % (
                split, n_img, n_empty, n_box))
            for k in sorted(st):
                fh.write("      %-22s %d\n" % (k, st[k]))
            fh.write("\n")
        fh.write("全类别框数（YOLO 编号）:\n")
        for c in sorted(all_per_class):
            fh.write("   %2d %-18s %d\n" % (c, CLASS_NAMES[c], all_per_class[c]))
        fh.write("\n合计有效框: %d\n" % sum(all_per_class.values()))

    print("\n" + "=" * 64)
    print("转换汇总")
    print("=" * 64)
    for split in ("train", "val"):
        n_img, n_empty, n_box, st = summary[split]
        print("  [%s] 图片 %d | 有效框 %d | 空标注图 %d" % (split, n_img, n_box, n_empty))
    print("\n  各类别框数（YOLO 编号）:")
    for c in sorted(all_per_class):
        print("    %2d %-18s %6d" % (c, CLASS_NAMES[c], all_per_class[c]))
    print("\n  丢弃统计:")
    for k in sorted(grand):
        if k.startswith("丢弃") or k in ("解析失败", "图片缺失", "尺寸非法", "已裁剪"):
            print("    %-24s %d" % (k, grand[k]))
    print("\n  已写出: %s" % os.path.join(DST, "VisDrone.yaml"))
    print("  报告:   %s" % os.path.join(DST, "convert_report.txt"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
