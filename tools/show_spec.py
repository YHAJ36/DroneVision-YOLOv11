"""打印训练将用到的【模型 + 数据】完整清单（以磁盘实际内容为准）。"""
import os
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YOLO = os.path.join(ROOT, "dataset_visdrone/yolo")
NAMES = ["pedestrian", "people", "bicycle", "car", "van",
         "truck", "tricycle", "awning-tricycle", "bus", "motor"]


def human(n):
    n = float(n)
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return "%.2f %s" % (n, u)
        n /= 1024
    return "%.2f TB" % n


def dirsize(p):
    t = 0
    for dp, _, fs in os.walk(p):
        for f in fs:
            try:
                t += os.path.getsize(os.path.join(dp, f))
            except OSError:
                pass
    return t


print("=" * 74)
print("一、模型")
print("=" * 74)
for w in sorted(os.listdir(os.path.join(ROOT, "weights"))):
    p = os.path.join(ROOT, "weights", w)
    print("  %-16s %10s" % (w, human(os.path.getsize(p))))
print("  主用: yolo11s.pt（YOLO11s，9,458,752 参数，COCO 预训练 nc=80 → 迁移到 nc=10）")

print()
print("=" * 74)
print("二、数据集配置 VisDrone.yaml")
print("=" * 74)
yp = os.path.join(YOLO, "VisDrone.yaml")
with open(yp, "r", encoding="utf-8") as fh:
    for line in fh.read().rstrip().splitlines():
        print("  " + line)

print()
print("=" * 74)
print("三、数据规模")
print("=" * 74)
print("  %-8s %8s %10s %12s %12s" % ("子集", "图片", "标签", "标注框", "磁盘占用"))
print("  " + "-" * 58)

grand_boxes = 0
grand_imgs = 0
for split in ("train", "val"):
    idir = os.path.join(YOLO, "images", split)
    ldir = os.path.join(YOLO, "labels", split)
    imgs = [f for f in os.listdir(idir) if f.lower().endswith((".jpg", ".png"))]
    lbls = [f for f in os.listdir(ldir) if f.lower().endswith(".txt")]
    nbox = 0
    for f in lbls:
        with open(os.path.join(ldir, f), "r", encoding="utf-8") as fh:
            nbox += sum(1 for ln in fh if ln.strip())
    sz = dirsize(idir) + dirsize(ldir)
    grand_boxes += nbox
    grand_imgs += len(imgs)
    print("  %-8s %8d %10d %12d %12s" % (split, len(imgs), len(lbls), nbox, human(sz)))
print("  " + "-" * 58)
print("  %-8s %8d %10s %12d" % ("合计", grand_imgs, "-", grand_boxes))

print()
print("=" * 74)
print("四、类别分布（trai+val 合计，YOLO 编号 0~9）")
print("=" * 74)
cnt = Counter()
for split in ("train", "val"):
    ldir = os.path.join(YOLO, "labels", split)
    for f in os.listdir(ldir):
        if not f.endswith(".txt"):
            continue
        with open(os.path.join(ldir, f), "r", encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    try:
                        cnt[int(ln.split()[0])] += 1
                    except (ValueError, IndexError):
                        pass

tot = sum(cnt.values())
print("  %-4s %-18s %9s %8s  %s" % ("编号", "名称", "框数", "占比", "分布"))
mx = max(cnt.values()) if cnt else 1
for i in range(10):
    c = cnt.get(i, 0)
    bar = "#" * max(1, int(28.0 * c / mx))
    print("  %-4d %-18s %9d %7.2f%%  %s" % (i, NAMES[i], c, c * 100.0 / tot if tot else 0, bar))
print()
print("  合计 %d 个框" % tot)
if tot:
    hi = max(cnt.values())
    lo = min(cnt.values())
    print("  最不均衡倍数: %d / %d = %.1f 倍（car vs awning-tricycle）"
          % (hi, lo, hi / float(lo)))

print()
print("=" * 74)
print("五、源数据与原始压缩包（备份保留）")
print("=" * 74)
for d in sorted(os.listdir(ROOT)):
    p = os.path.join(ROOT, d)
    if os.path.isfile(p) and d.endswith(".zip"):
        print("  %-34s %10s" % (d, human(os.path.getsize(p))))
    elif os.path.isdir(p) and d.startswith("VisDrone2019-DET-"):
        n = sum(1 for dp, _, fs in os.walk(p) for f in fs)
        print("  %-34s %10s  (%d 个文件)" % (d + "\\", human(dirsize(p)), n))
