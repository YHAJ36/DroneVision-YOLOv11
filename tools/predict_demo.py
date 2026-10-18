"""推理测试工具 —— 用训练好的 VisDrone 模型做推理，适合自己验证效果。

用法：
    # 单张图片
    python predict_demo.py 某个图片.jpg

    # 整个文件夹
    python predict_demo.py dataset_visdrone/yolo/images/val

    # 视频文件
    python predict_demo.py 测试.mp4

    # 摄像头（实时）
    python predict_demo.py 0

可选参数：
    --conf 0.25        置信度阈值（VisDrone 目标密集，可试 0.35~0.4 减少误检）
    --imgsz 896        推理分辨率，默认与训练一致
    --weights 路径     换用其它权重（默认 best.pt）
    --no-show          不弹窗显示，仅保存结果

输出：
    标注后的图片/视频保存到 runs/predict/<时间戳>/
"""
import argparse
import os
import sys
import time
from pathlib import Path

# 项目根目录：本脚本位于 tools，向上 1 级即项目根，路径一律基于它拼
ROOT = Path(__file__).resolve().parents[1]


os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "ultralytics-cfg"))

from ultralytics import YOLO  # noqa: E402

DEFAULT_WEIGHTS = str(ROOT / "runs" / "yolo11s_896" / "weights" / "best.pt")
OUT_DIR = str(ROOT / "runs" / "predict")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="图片 / 文件夹 / 视频 / 摄像头编号(如 0)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=896)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        sys.exit("[!] 找不到权重: %s" % args.weights)

    print("=" * 66)
    print("模型  : %s" % args.weights)
    print("来源  : %s" % args.source)
    print("imgsz : %d    conf : %.2f" % (args.imgsz, args.conf))
    print("=" * 66, flush=True)

    model = YOLO(args.weights)
    print("类别  : %s" % list(model.names.values()), flush=True)

    t0 = time.time()
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        save=True,
        project=OUT_DIR,
        name=time.strftime("%Y%m%d-%H%M%S"),
        exist_ok=True,
        show=not args.no_show,
        verbose=True,
    )
    dt = time.time() - t0

    from collections import Counter
    cnt = Counter()
    files = 0
    total_objs = 0
    for r in results:
        files += 1
        boxes = r.boxes
        n = 0 if boxes is None else len(boxes)
        total_objs += n
        if boxes is not None:
            for b in boxes:
                cnt[model.names[int(b.cls[0])]] += 1

    print("\n" + "=" * 66)
    print("结果汇总")
    print("=" * 66)
    print("  处理 %d 个输入，耗时 %.2f s，均速 %.1f ms/项" % (
        files, dt, dt * 1000.0 / max(1, files)))
    print("  共检出 %d 个目标" % total_objs)
    if cnt:
        print("\n  按类别（降序）：")
        for name, c in cnt.most_common():
            print("    %-18s %5d" % (name, c))
    print("\n  标注结果已保存到: %s" % OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
