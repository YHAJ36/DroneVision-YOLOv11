"""分析 VisDrone 标注的真实分布，为转换规则提供依据。

重点回答：
  1. 到底存在哪些 class_id？（规格只定义了 0 和 1~10）
  2. score 字段的真实含义——是否存在 class 有效但 score=0 的框？
  3. 框是否有越界？宽高是否有非正数？
  4. 图片尺寸分布。
"""
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

PIL = None
try:
    from PIL import Image
    PIL = Image
except ImportError:
    pass

ROOT = Path(__file__).resolve().parents[1]
SPLITS = {
    "train": str(ROOT / "dataset_visdrone" / "raw_train" / "VisDrone2019-DET-train"),
    "val": str(ROOT / "dataset_visdrone" / "raw_val" / "VisDrone2019-DET-val"),
}

cls_cnt = Counter()
pair_cnt = Counter()          # (class, score)
fieldcnt = Counter()
sizes = Counter()
n_files = 0
n_lines = 0
bad_lines = 0
oob = 0
nonpos = 0
score0_by_class = Counter()
maxdim = Counter()

for split, base in SPLITS.items():
    adir = os.path.join(base, "annotations")
    idir = os.path.join(base, "images")
    if not os.path.isdir(adir):
        print("[!] 缺少目录: %s" % adir)
        continue
    files = sorted(os.listdir(adir))
    n_files += len(files)
    print("[%s] 标注文件 %d 个" % (split, len(files)))

    # 抽样部分图片统计尺寸（全量太慢）
    if PIL and os.path.isdir(idir):
        imgs = sorted(os.listdir(idir))
        step = max(1, len(imgs) // 300)
        for nm in imgs[::step]:
            try:
                with PIL.open(os.path.join(idir, nm)) as im:
                    sizes[im.size] += 1
            except Exception:
                pass

    for fn in files:
        with open(os.path.join(adir, fn), "r", encoding="utf-8", errors="replace") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                n_lines += 1
                p = ln.split(",")
                fieldcnt[len(p)] += 1
                if len(p) != 8:
                    bad_lines += 1
                    continue
                try:
                    x, y, w, h = (float(p[0]), float(p[1]), float(p[2]), float(p[3]))
                    score = int(float(p[4]))
                    cid = int(float(p[5]))
                except ValueError:
                    bad_lines += 1
                    continue
                cls_cnt[cid] += 1
                pair_cnt[(cid, score)] += 1
                if cid != 0 and score == 0:
                    score0_by_class[cid] += 1
                if w <= 0 or h <= 0:
                    nonpos += 1

print("\n" + "=" * 70)
print("总标注文件 %d，总行数 %d，字段数非 8 的行 %d" % (n_files, n_lines, bad_lines))
print("字段数分布: %s" % dict(fieldcnt))

print("\n--- class_id 分布 ---")
for cid in sorted(cls_cnt):
    print("  class %-3d : %7d 个框" % (cid, cls_cnt[cid]))

print("\n--- (class, score) 联合分布 ---")
for (cid, sc) in sorted(pair_cnt):
    print("  class %-3d score=%d : %7d" % (cid, sc, pair_cnt[(cid, sc)]))

print("\n--- 关键问题：class 1~10 中 score=0 的框 ---")
if score0_by_class:
    tot = sum(score0_by_class.values())
    print("  共 %d 个框属于 class 1~10 但 score=0：" % tot)
    for cid in sorted(score0_by_class):
        print("     class %-3d : %d" % (cid, score0_by_class[cid]))
else:
    print("  没有。score=0 只出现在 class 0 —— 说明 score 与 class 0 强相关。")

print("\n--- 标注里出现但不属于 0~10 的 class ---")
odd = sorted(c for c in cls_cnt if c not in range(0, 11))
print("  %s" % (odd if odd else "无"))
for c in odd:
    print("     class %-3d : %d 个框（规格未定义，需决定如何处理）" % (c, cls_cnt[c]))

print("\n--- 几何异常 ---")
print("  width/height <= 0 的框: %d" % nonpos)

print("\n--- 图片尺寸抽样（最多 300 张/子集）---")
for s, c in sizes.most_common(10):
    print("  %s : %d 张" % (s, c))
