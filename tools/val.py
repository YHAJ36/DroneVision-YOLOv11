"""验证脚本 —— 在 val 集上评估权重，打印 overall + per-class AP。

设计要点：
  1. batch=8 / workers=0 —— ultralytics 的 val 默认 batch=16，是训练 batch 的两倍。
     896 + yolo11s + 10 类 + NMS 在 8GB 卡上没理由冒这个险，与训练保持一致。
     workers=0 避免 Windows 上 DataLoader 多进程(spawn)重新导入主模块。
  2. 默认同时评估 best.pt 与 last.pt —— best.pt 是按 val 的 fitness 选出来的，
     再拿它报 val 存在**选择偏置**。两者接近说明偏置小，差距大就必须在报告里承认。

用法：
    envs/yolo26-env/Scripts/python.exe val.py [权重路径]
    不带参数 -> 自动评估 best.pt 和 last.pt 并对比
"""
import os
import sys
from pathlib import Path

# 项目根目录：本脚本位于 tools，向上 1 级即项目根，路径一律基于它拼
ROOT = Path(__file__).resolve().parents[1]


os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(ROOT / "ultralytics-cfg"),
)

from ultralytics import YOLO  # noqa: E402

DATA = str(ROOT / "dataset_visdrone" / "yolo" / "VisDrone.yaml")
RUN_DIR = str(ROOT / "runs" / "yolo11s_896")
CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor",
]

# 与训练保持一致，避免 val 阶段 OOM
VAL_ARGS = dict(imgsz=896, batch=8, workers=0, device=0, split="val", plots=True)


def evaluate(weights):
    """评估单个权重，返回 (metrics, 指标字典)。"""
    print("\n" + "=" * 66)
    print("评估: %s" % weights)
    print("=" * 66)
    model = YOLO(weights)
    m = model.val(data=DATA, **VAL_ARGS)

    res = {
        "map50": float(m.box.map50),
        "map": float(m.box.map),
        "map75": float(m.box.map75),
        "mp": float(m.box.mp),
        "mr": float(m.box.mr),
        "ap50": [float(x) for x in m.box.ap50],
        "ap": [float(x) for x in m.box.ap],
    }

    print("  mAP@0.5      : %.4f" % res["map50"])
    print("  mAP@0.5:0.95 : %.4f" % res["map"])
    print("  mAP@0.75     : %.4f" % res["map75"])
    print("  precision    : %.4f" % res["mp"])
    print("  recall       : %.4f" % res["mr"])

    print("\n  per-class AP（按 AP@0.5 降序；末位通常是小目标/稀有类）")
    print("  %-18s %10s %14s" % ("类别", "AP@0.5", "AP@0.5:0.95"))
    print("  " + "-" * 45)
    order = sorted(range(len(res["ap50"])), key=lambda i: -res["ap50"][i])
    for i in order:
        name = CLASS_NAMES[i] if i < len(CLASS_NAMES) else "class%d" % i
        ap_all = res["ap"][i] if i < len(res["ap"]) else 0.0
        print("  %-18s %10.4f %14.4f" % (name, res["ap50"][i], ap_all))
    return res


def main():
    # 指定了权重就只评它
    if len(sys.argv) > 1:
        w = sys.argv[1]
        if not os.path.exists(w):
            sys.exit("[!] 找不到权重: %s" % w)
        evaluate(w)
        return 0

    best = os.path.join(RUN_DIR, "weights", "best.pt")
    last = os.path.join(RUN_DIR, "weights", "last.pt")

    results = {}
    for tag, w in (("best", best), ("last", last)):
        if os.path.exists(w):
            results[tag] = evaluate(w)
        else:
            print("\n[跳过] %s 不存在: %s" % (tag, w))

    if not results:
        sys.exit("[!] best.pt 与 last.pt 都不存在，请先训练")

    # ---- 选择偏置对比 ----
    if "best" in results and "last" in results:
        b, l = results["best"], results["last"]
        print("\n" + "=" * 66)
        print("选择偏置检查（报告的严谨性依据）")
        print("=" * 66)
        print("  %-16s %10s %10s %10s" % ("指标", "best.pt", "last.pt", "差值"))
        print("  " + "-" * 50)
        for key, label in (("map50", "mAP@0.5"), ("map", "mAP@0.5:0.95"),
                           ("mp", "precision"), ("mr", "recall")):
            d = b[key] - l[key]
            print("  %-16s %10.4f %10.4f %+10.4f" % (label, b[key], l[key], d))

        gap = b["map50"] - l["map50"]
        print()
        if abs(gap) <= 0.01:
            print("  结论: 两者 mAP@0.5 相差 %.4f（<=0.01），选择偏置小。" % gap)
            print("        报告可用 best.pt 的数字，但仍建议注明口径。")
        else:
            print("  结论: 两者 mAP@0.5 相差 %+.4f，选择偏置**不可忽略**。" % gap)
            print("        报告中必须同时给出 last.pt 的数字，并注明 best.pt 按 val 选出。")
        print()
        print("  ★ 口径声明（建议原文写进报告）：")
        print("     \"本报告所有指标均在 VisDrone2019-DET 的 val 集上测得。")
        print("      best.pt 系按 val 的 fitness(0.1*mAP50 + 0.9*mAP50-95) 选出，")
        print("      再以该权重回报 val，故结果含选择偏置，非严格泛化估计。")
        print("      VisDrone2019-DET 的 test-dev/test-challenge 无公开标注，")
        print("      无法在留出集上做无偏评估。\"")

    return 0


if __name__ == "__main__":
    # Windows 上 DataLoader 子进程会重新导入本模块，需要此保护
    sys.exit(main())
