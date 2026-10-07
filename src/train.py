"""
Week 4 — Model Fine-tuning（Transfer Learning）
================================================
用 Week 3 的資料集對 YOLOv8n 做 Fine-tuning，產生「自己的」交通 Object Detection 模型。

關鍵觀念（大一也能懂）：
- Transfer Learning（遷移學習）：別人已經用百萬張圖訓練好的 YOLOv8n 當起點，
  我們只在自己的小資料上「微調」，省時間也省資料。
- Epoch（輪）：把整個訓練集看過幾遍。
- Batch（批次）：一次送幾張圖進去算梯度。
- Learning Rate（lr0）：模型「每一步調整多大」的速度。
- Training Loss：預測與標註的差距，越訓練應該越低。
- 訓練完會產生 best.pt（驗證集表現最好的權重），我們複製到 models/best.pt。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

from common import resolve_data_yaml


def main() -> int:
    ap = argparse.ArgumentParser(description="Week 4: Fine-tune YOLO on traffic dataset")
    ap.add_argument("--data", default="data/dataset.yaml")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--name", default="week4_traffic")
    args = ap.parse_args()

    model = YOLO(args.weights)
    data_yaml = resolve_data_yaml(args.data)
    try:
        results = model.train(
            data=data_yaml,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            name=args.name,
            exist_ok=True,
            verbose=True,
        )
    finally:
        Path(data_yaml).unlink(missing_ok=True)

    # 把表現最好的權重複製到專案約定的 models/best.pt
    best = Path(results.save_dir) / "weights" / "best.pt"
    dst = Path("models/best.pt")
    dst.parent.mkdir(exist_ok=True)
    if best.exists():
        shutil.copy(best, dst)
        print(f"已複製 best.pt -> {dst}")
    else:
        print("警告：找不到 best.pt，請檢查訓練輸出。", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
