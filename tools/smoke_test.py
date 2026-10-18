"""烧机 / 首轮标定 —— 用 3% 数据跑 1 个 epoch，一次性回答三个问题：

  1. batch=8 + imgsz=896 在 8GB 卡上会不会 OOM？（看峰值显存）
  2. 每个 iteration 实际耗多少秒？（据此外推满数据集单 epoch 耗时）
  3. 整条流水线（数据加载 / workers / AMP / 验证）能不能跑通？

【重要】全部逻辑放在 _run() 里，由 __main__ 保护调用。
Windows 上 DataLoader 多进程(spawn)会重新导入主模块，平铺脚本会导致子进程递归重启训练
——这正是上一次卡住的原因。

所有进度都 flush 到 stdout，便于实时观察。
"""
import json
import os
import sys
import time
from pathlib import Path

# 项目根目录：本脚本位于 tools，向上 1 级即项目根，路径一律基于它拼
ROOT = Path(__file__).resolve().parents[1]


os.environ.setdefault(
    "YOLO_CONFIG_DIR",
    str(ROOT / "ultralytics-cfg"),
)

import torch  # noqa: E402
from ultralytics import YOLO  # noqa: E402

DATA = str(ROOT / "dataset_visdrone" / "yolo" / "VisDrone.yaml")
WEIGHTS = str(ROOT / "models" / "yolo11s.pt")
PROJECT = str(ROOT / "runs")
NAME = "_smoke_test"

FRACTION = 0.03
TRAIN_TOTAL = 6471
VAL_TOTAL = 548
BATCH = 8
IMGSZ = 896
WORKERS = int(os.environ.get("SMOKE_WORKERS", "2"))


def say(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def _run():
    say("烧机开始：fraction=%s batch=%d imgsz=%d workers=%d" % (
        FRACTION, BATCH, IMGSZ, WORKERS))
    say("权重: %s" % WEIGHTS)
    say("数据: %s" % DATA)

    say("加载权重中 ...")
    torch.cuda.reset_peak_memory_stats()
    model = YOLO(WEIGHTS)
    say("权重加载完成")

    t0 = time.time()
    say("调用 train() ...")
    model.train(
        data=DATA,
        epochs=1,
        fraction=FRACTION,
        imgsz=IMGSZ,
        batch=BATCH,
        device=0,
        amp=True,
        workers=WORKERS,
        project=PROJECT,
        name=NAME,
        exist_ok=True,
        optimizer="SGD",
        cos_lr=True,
        lr0=0.01,
        lrf=0.005,
        mosaic=0.8,
        mixup=0.1,
        copy_paste=0.1,
        scale=0.5,
        save=False,
        plots=False,
        val=True,
        verbose=True,
    )
    wall = time.time() - t0
    say("train() 返回，墙钟 %.1f s" % wall)

    peak_reserved = torch.cuda.max_memory_reserved() / 2 ** 30
    peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 30
    _free, total = torch.cuda.mem_get_info()
    total_gb = total / 2 ** 30

    say("=" * 60)
    say("峰值显存 reserved: %.2f GB / 总 %.2f GB" % (peak_reserved, total_gb))
    say("峰值显存 alloc   : %.2f GB" % peak_alloc)
    say("OOM 判定: %s" % ("未 OOM，余量 %.2f GB" % (total_gb - peak_reserved)
                          if peak_reserved < total_gb * 0.97 else "接近/已超显存"))

    result = {
        "fraction": FRACTION,
        "workers": WORKERS,
        "wall_sec": wall,
        "peak_reserved_gb": peak_reserved,
        "peak_alloc_gb": peak_alloc,
        "total_gb": total_gb,
        "oom": peak_reserved >= total_gb * 0.97,
    }

    csv_path = os.path.join(PROJECT, NAME, "results.csv")
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as fh:
            rows = [r for r in fh.read().strip().splitlines() if r.strip()]
        if len(rows) >= 2:
            cols = [c.strip() for c in rows[0].split(",")]
            if "time" in cols:
                epoch_sec = float(rows[-1].split(",")[cols.index("time")])
                sub_imgs = max(1, int(TRAIN_TOTAL * FRACTION))
                per_img = epoch_sec / sub_imgs
                est_train = per_img * TRAIN_TOTAL
                val_share = epoch_sec * 0.12
                est_val = val_share / max(1, int(VAL_TOTAL * FRACTION)) * VAL_TOTAL
                est_epoch = est_train + est_val
                say("3%% 数据单 epoch : %.1f s" % epoch_sec)
                say("每图训练耗时    : %.4f s" % per_img)
                say("外推训练部分    : %.0f s" % est_train)
                say("外推验证部分    : %.0f s" % est_val)
                say("★ 外推单 epoch  : %.0f s = %.2f 分钟" % (est_epoch, est_epoch / 60.0))
                result.update({
                    "subset_epoch_sec": epoch_sec,
                    "est_epoch_sec": est_epoch,
                    "est_epoch_min": est_epoch / 60.0,
                })
        else:
            say("[!] results.csv 行数不足")
    else:
        say("[!] 找不到 %s" % csv_path)

    out = os.path.join(PROJECT, "smoke_result.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    say("已写: %s" % out)
    say("烧机结束")
    return 0


if __name__ == "__main__":
    sys.exit(_run())
