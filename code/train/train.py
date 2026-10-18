"""YOLO11s 训练脚本--VisDrone2019-DET,896分辨率"""

import os
from pathlib import Path

# 项目根目录：本脚本位于 code/train，向上 2 级即项目根，路径一律基于它拼
ROOT = Path(__file__).resolve().parents[2]


os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "ultralytics-cfg"))

from ultralytics import YOLO

#路径配置
DATA  = str(ROOT / "dataset_visdrone" / "yolo" / "VisDrone.yaml")
WEIGHTS = str(ROOT / "models" / "yolo11s.pt")
PROJECT = str(ROOT / "runs")
NAME = "yolo11s_896"


def main():
    model = YOLO(WEIGHTS)
    
    model.train(data=DATA,
                 epochs=100,
                 imgsz=896,
                 batch=8,
                 device=0,
                 amp=True,
                 workers=0,
                 patience=30,
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
                 close_mosaic=5,
                 save_period=5,
    )

    print("训练完成，最佳权重：%s" % os.path.join(PROJECT,NAME,"weights","best.pt"))

if __name__ == "__main__":
    main()