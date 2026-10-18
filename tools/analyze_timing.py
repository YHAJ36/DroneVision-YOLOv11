"""从训练的 results.csv 里提取真实 epoch 耗时，用于估算训练时长。

results.csv 的 'time' 列是累计秒数，相邻行之差即单个 epoch 的实际耗时。
"""
import csv
import glob
import os

RUNS = [
    r"D:\2\exp_upscale\runs\att_v11_896_triplet_s43",
    r"D:\2\exp_upscale\runs\att_v11_896_triplet_s44",
    r"D:\2\exp_upscale\runs\nf_yolo26_896_s43",
    r"D:\2\exp_upscale\runs\nf_yolo26_896_s44",
]

print("%-34s %7s %12s %12s %12s" % ("实验", "epoch", "总耗时", "均/epoch", "中位/epoch"))
print("-" * 84)

for run in RUNS:
    p = os.path.join(run, "results.csv")
    if not os.path.exists(p):
        print("%-34s  (无 results.csv)" % os.path.basename(run))
        continue
    times = []
    epochs = 0
    with open(p, "r", encoding="utf-8", errors="replace") as fh:
        rd = csv.DictReader(fh)
        cols = None
        for row in rd:
            if cols is None:
                cols = [k.strip() for k in row.keys()]
            epochs += 1
            k = None
            for c in row:
                if c and c.strip() == "time":
                    k = c
                    break
            if k is not None:
                try:
                    times.append(float(row[k]))
                except (ValueError, TypeError):
                    pass

    if len(times) < 2:
        print("%-34s %7d  (time 列不足)" % (os.path.basename(run), epochs))
        continue

    deltas = [b - a for a, b in zip(times, times[1:])]
    deltas = [d for d in deltas if d > 0]
    if not deltas:
        continue
    total = (times[-1] - times[0]) / 3600.0
    avg = sum(deltas) / len(deltas)
    srt = sorted(deltas)
    med = srt[len(srt) // 2]
    print("%-34s %7d %10.2f h %10.1f s %10.1f s" % (
        os.path.basename(run), epochs, total, avg, med))

print()
print("说明：'time' 为累计秒，差值即单 epoch 实际耗时（含验证）。")
print("      若该列缺失或为 0，则以文件跨度粗估。")
