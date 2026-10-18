"""校验 YOLO 转换结果：数量一致性、标签格式、坐标系合法性。"""
import os
import sys
from collections import Counter
from pathlib import Path

# 项目根目录：本脚本位于 tools，向上 1 级即项目根，路径一律基于它拼
ROOT = Path(__file__).resolve().parents[1]

ROOT = str(ROOT / "dataset_visdrone" / "yolo")
NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

problems = []

print("=" * 70)
print("一、数量一致性")
print("=" * 70)
overall_ok = True
for split in ("train", "val"):
    idir = os.path.join(ROOT, "images", split)
    ldir = os.path.join(ROOT, "labels", split)
    if not os.path.isdir(idir) or not os.path.isdir(ldir):
        print("  [%s] 目录缺失" % split)
        overall_ok = False
        continue
    imgs = {os.path.splitext(f)[0] for f in os.listdir(idir) if f.lower().endswith(".jpg")}
    lbls = {os.path.splitext(f)[0] for f in os.listdir(ldir) if f.lower().endswith(".txt")}
    only_img = imgs - lbls
    only_lbl = lbls - imgs
    print("  [%s] 图片 %d | 标签 %d | 差集 图片无标签 %d, 标签无图片 %d  -> %s" % (
        split, len(imgs), len(lbls), len(only_img), len(only_lbl),
        "一致 ✓" if not only_img and not only_lbl else "不一致 ✗"))
    if only_img or only_lbl:
        overall_ok = False
        problems.append("%s: 图片/标签不配对 (无标签图 %d, 无图标签 %d)" % (
            split, len(only_img), len(only_lbl)))
        for s in list(only_img)[:5]:
            print("        无标签: %s" % s)
        for s in list(only_lbl)[:5]:
            print("        无图片: %s" % s)

print()
print("=" * 70)
print("二、标签内容校验（逐行）")
print("=" * 70)

for split in ("train", "val"):
    ldir = os.path.join(ROOT, "labels", split)
    if not os.path.isdir(ldir):
        continue
    n_lines = 0
    n_files = 0
    n_empty = 0
    cls_cnt = Counter()
    bad_fields = 0
    bad_cls = 0
    bad_range = 0
    bad_norm = 0
    samples = []
    for fn in sorted(os.listdir(ldir)):
        if not fn.endswith(".txt"):
            continue
        n_files += 1
        p = os.path.join(ldir, fn)
        with open(p, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip()]
        if not lines:
            n_empty += 1
        for ln in lines:
            n_lines += 1
            parts = ln.split()
            # YOLO 检测标签为 5 个字段：class cx cy w h
            if len(parts) != 5:
                bad_fields += 1
                if len(samples) < 3:
                    samples.append((fn, ln))
                continue
            try:
                c = int(parts[0])
                vals = [float(v) for v in parts[1:]]
            except ValueError:
                bad_fields += 1
                if len(samples) < 3:
                    samples.append((fn, ln))
                continue
            if not (0 <= c <= 9):
                bad_cls += 1
            else:
                cls_cnt[c] += 1
            cx, cy, w, h = vals
            for v in (cx, cy, w, h):
                if v < 0.0 or v > 1.0:
                    bad_range += 1
                    break
            if not (0.0 < cx < 1.0 and 0.0 < cy < 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                bad_norm += 1

    print("  [%s] 标签文件 %d（空文件 %d）| 总行数 %d" % (split, n_files, n_empty, n_lines))
    print("       字段数错误 %d | 类别越界 %d | 坐标出 [0,1] %d | 归一化非法(含 w/h<=0 或中心点在边界) %d" % (
        bad_fields, bad_cls, bad_range, bad_norm))
    if bad_fields or bad_cls or bad_range or bad_norm:
        overall_ok = False
        problems.append("%s: 标签格式问题 (字段 %d, 类别 %d, 范围 %d, 归一化 %d)" % (
            split, bad_fields, bad_cls, bad_range, bad_norm))
        for fn, ln in samples:
            print("        样例: %s -> %s" % (fn, ln))
    print("       类别分布: %s" % dict(sorted(cls_cnt.items())))

print()
print("=" * 70)
print("三、VisDrone.yaml")
print("=" * 70)
yp = os.path.join(ROOT, "VisDrone.yaml")
if os.path.exists(yp):
    with open(yp, "r", encoding="utf-8") as fh:
        content = fh.read()
    print(content)
    try:
        import yaml
        cfg = yaml.safe_load(content)
        print("  解析结果: nc=%s, names 条数=%s" % (
            cfg.get("nc"), len(cfg.get("names") or {})))
        if cfg.get("nc") != 10 or len(cfg.get("names") or {}) != 10:
            problems.append("yaml 的 nc 或 names 数量不等于 10")
            overall_ok = False
        names_ok = all((cfg.get("names") or {}).get(i) == NAMES[i] for i in range(10))
        print("  类别名与规格一致: %s" % ("是 ✓" if names_ok else "否 ✗"))
        basef = cfg.get("path")
        for key in ("train", "val"):
            rel = cfg.get(key)
            full = os.path.join(basef, rel)
            print("  %s 路径存在: %s (%s)" % (key, os.path.isdir(full), full))
    except ImportError:
        print("  (无 pyyaml，跳过解析)")
else:
    print("  [!] 找不到 VisDrone.yaml")
    overall_ok = False
    problems.append("缺少 VisDrone.yaml")

print()
print("=" * 70)
if overall_ok:
    print("结论：全部检查通过 ✓  可以开始训练")
else:
    print("结论：发现问题 ✗")
    for p in problems:
        print("  - %s" % p)
sys.exit(0 if overall_ok else 1)
