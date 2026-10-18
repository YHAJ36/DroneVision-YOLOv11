"""用训练好的模型做推理检测"""

import os
import sys
from pathlib import Path

# 项目根目录：本脚本位于 code/train，向上 2 级即项目根，路径一律基于它拼
ROOT = Path(__file__).resolve().parents[2]


os.environ.setdefault("YOLO_CONFIG_DIR",
                      str(ROOT / "ultralytics-cfg"))

from ultralytics import YOLO


# 训练产出的最佳权重
WEIGHTS = str(ROOT / "runs" / "yolo11s_896" / "weights" / "best.pt")


def detect_one(image_path, conf=0.25):
    """对单张图片做检测，返回 [(类名, 置信度, [x1,y1,x2,y2]), ...]"""

    # 加载模型
    model = YOLO(WEIGHTS)

    # 推理：只传一张图，所以结果列表取第 0 个
    results = model.predict(
        source=image_path,
        imgsz=896,       # 必须和训练时一致
        conf=conf,       # 置信度阈值
        verbose=False,
    )

    r = results[0]

    # 一个目标都没检出时 boxes 为 None
    if r.boxes is None or len(r.boxes) == 0:
        return []

    detections = []
    for i in range(len(r.boxes)):
        cls_id = int(r.boxes.cls[i])          # 类别编号
        confidence = float(r.boxes.conf[i])   # 置信度
        xyxy = r.boxes.xyxy[i].tolist()       # 坐标，张量转列表
        name = model.names[cls_id]            # 编号转类名
        detections.append((name, confidence, xyxy))

    return detections


def main():
    if len(sys.argv) < 2:
        print("用法: python detect.py <图片路径>")
        return

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print("[!] 图片不存在: %s" % image_path)
        return

    print("模型: %s" % WEIGHTS)
    print("图片: %s" % image_path)

    detections = detect_one(image_path)
    print("\n检出 %d 个目标" % len(detections))

    # 按类别计数
    counts = {}
    for name, confidence, xyxy in detections:
        counts[name] = counts.get(name, 0) + 1

    # 降序打印
    for name, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        print("  %-18s %d" % (name, cnt))


if __name__ == "__main__":
    main()
