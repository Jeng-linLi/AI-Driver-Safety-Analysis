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
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO


def resolve_data_yaml(path: str) -> str:
    """讀取 dataset.yaml，把相對 path 解析成絕對路徑，寫成「臨時 yaml」並回傳其路徑。

    為什麼要這樣做：
    1) ultralytics 預設會把 dataset.yaml 裡的相對 path 當成 settings 中
       datasets_dir 下的子路徑，導致找不到我們放在 data/ 下的資料集。
       → 所以這裡改成「相對於 dataset.yaml 所在目錄」解析成絕對路徑。
    2) ultralytics 的 model.train(data=...) 只接受「yaml 檔路徑字串」，
       不接受 dict。→ 所以寫成一個臨時 yaml 再傳路徑。
    """
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str((p.parent / cfg["path"]).resolve())
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return tmp.name


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
