"""YOLO11s 训练脚本 —— VisDrone2019-DET，896 分辨率，RTX 4060 Laptop 8GB

用法（必须用本项目自带的解释器）：
    envs/yolo26-env/Scripts/python.exe train.py [选项]

常用：
    --time 5.5            墙钟时间上限（小时）。到点自动停并保存，用于硬性截止场景
    --epochs 100          epoch 上限（与 --time 谁先到算谁）
    --resume              从 runs/yolo11s_896/weights/last.pt 续训

例（今晚 7 小时窗口）：
    train.py --epochs 100 --time 5.5

超参借鉴同机已验证配置：YOLO11s + batch8 + imgsz896 在同一块 RTX 4060 Laptop 上
实测 108.2 s/epoch（1428 张训练集），batch=8 显存安全。

OOM 应急顺序：batch 8 -> 4 -> 2；仍不行再降 imgsz 640。
"""
import argparse
import os
import sys
from pathlib import Path

# 项目根目录：本脚本位于 tools，向上 1 级即项目根，路径一律基于它拼
ROOT = Path(__file__).resolve().parents[1]


# 把 ultralytics 配置目录固定到项目内。**必须在 import ultralytics 之前设置。**
#
# 指向项目内也更干净，多个项目之间不会互相污染配置。
os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(ROOT / "ultralytics-cfg"),
)

from ultralytics import YOLO  # noqa: E402

DATA = str(ROOT / "dataset_visdrone" / "yolo" / "VisDrone.yaml")
WEIGHTS = str(ROOT / "models" / "yolo11s.pt")
PROJECT = str(ROOT / "runs")
NAME = "yolo11s_896"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=100, help="epoch 上限")
    ap.add_argument("--time", type=float, default=None,
                    help="墙钟时间上限（小时）；与 epochs 谁先到算谁")
    ap.add_argument("--batch", type=int, default=8, help="OOM 则 4 -> 2")
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--save-period", type=int, default=5, dest="save_period",
                    help="每 N 个 epoch 存一次权重；短窗口用 5")
    ap.add_argument("--name", default=NAME)
    ap.add_argument("--resume", action="store_true",
                    help="从 runs/<name>/weights/last.pt 续训")
    return ap.parse_args()


def main():
    args = parse_args()

    for label, path in (("数据集配置", DATA), ("权重", WEIGHTS)):
        if not os.path.exists(path):
            sys.exit("[!] 找不到%s: %s" % (label, path))

    print("=" * 66)
    print("训练配置")
    print("=" * 66)
    print("  data      : %s" % DATA)
    print("  model     : %s" % WEIGHTS)
    print("  epochs    : %s" % args.epochs)
    print("  time 上限 : %s 小时%s" % (
        args.time, "" if args.time else "（无限制）"))
    print("  imgsz     : %s" % args.imgsz)
    print("  batch     : %s" % args.batch)
    print("  save_period:%s" % args.save_period)
    print("  输出      : %s/%s" % (PROJECT, args.name))
    print()
    if args.time:
        import datetime
        end = datetime.datetime.now() + datetime.timedelta(hours=args.time)
        print("  ★ 预计结束（墙钟）: %s" % end.strftime("%Y-%m-%d %H:%M:%S"))
        print()

    if args.resume:
        last = os.path.join(PROJECT, args.name, "weights", "last.pt")
        if not os.path.exists(last):
            sys.exit("[!] 找不到可续训的权重: %s" % last)
        print("续训自: %s" % last)
        model = YOLO(last)
        model.train(resume=True)
    else:
        model = YOLO(WEIGHTS)
        model.train(
            data=DATA,
            epochs=args.epochs,
            time=args.time,          # None = 不限制；单位小时
            imgsz=args.imgsz,
            batch=args.batch,
            device=0,
            amp=True,
            workers=args.workers,
            patience=args.patience,
            project=PROJECT,
            name=args.name,
            exist_ok=True,

            # ---- 以下借鉴同机已验证配置 ----
            optimizer="SGD",
            cos_lr=True,
            lr0=0.01,
            lrf=0.005,
            close_mosaic=5,
            mosaic=0.8,
            mixup=0.1,
            copy_paste=0.1,
            scale=0.5,
            hsv_h=0.02,
            hsv_s=0.8,
            hsv_v=0.5,
            save_period=args.save_period,
            deterministic=True,
            cache=False,
            plots=True,
            val=True,
            verbose=True,
        )

    print("\n训练结束。最佳权重：%s" % os.path.join(PROJECT, args.name, "weights", "best.pt"))


if __name__ == "__main__":
    # Windows 上 DataLoader 多进程(spawn)会重新导入本模块，
    # 没有这个保护会导致子进程递归重启训练。必须保留。
    main()
