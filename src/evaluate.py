"""
Week 5 — Model Evaluation
=========================
用 Week 4 訓練出的 models/best.pt 在驗證集上評估，量化模型表現。

指標（大一也能懂）：
- Precision（精確率）：模型「說是某類」的預測裡，有多少真的對。
- Recall（召回率）：所有「真正的某類物件」，模型抓到多少。
- F1 Score：Precision 與 Recall 的調和平均（兼顧兩者的綜合指標）。
- mAP@.5：IoU 重疊門檻 0.5 時的平均精確率（物件偵測主流指標）。
- mAP@.5-.95：更嚴格（多個 IoU 門檻平均），數值通常比 mAP@.5 低。
- Confusion Matrix（混淆矩陣）：哪些類別常被互相搞混。

注意：目前模型是在「合成資料」上訓練/評估，指標會偏高且不代表真實道路表現。
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO


def resolve_data_yaml(path: str) -> str:
    """同 train.py：把 dataset.yaml 的相對 path 解析成絕對路徑，寫成臨時 yaml。"""
    p = Path(path)
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["path"] = str((p.parent / cfg["path"]).resolve())
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(cfg, tmp)
    tmp.close()
    return tmp.name


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Week 5: Evaluate trained model")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--data", default="data/dataset.yaml")
    ap.add_argument("--name", default="week5_eval")
    args = ap.parse_args()

    data_yaml = resolve_data_yaml(args.data)
    try:
        model = YOLO(args.weights)
        metrics = model.val(data=data_yaml, name=args.name, verbose=True)
    finally:
        Path(data_yaml).unlink(missing_ok=True)

    rd = metrics.results_dict
    box = metrics.box
    names = model.names  # {id: name}
    # 每類 mAP@.5-.95（ultralytics 內部 box.ap 為 list）
    try:
        ap_list = list(box.ap)
        per_class = {
            names[i]: _to_float(ap_list[i]) for i in range(min(len(names), len(ap_list)))
        }
    except Exception:
        per_class = {}

    # 整體指標：優先從 results_dict 取，取不到再退回 box 屬性
    summary = {
        "weights": args.weights,
        "precision": _to_float(rd.get("metrics/precision(B)", getattr(box, "p", None))),
        "recall": _to_float(rd.get("metrics/recall(B)", getattr(box, "r", None))),
        "mAP50": _to_float(rd.get("metrics/mAP_0.5", getattr(box, "map50", None))),
        "mAP50-95": _to_float(rd.get("metrics/mAP_0.5:0.95", getattr(box, "map", None))),
        "per_class_mAP50-95": per_class,
    }

    out = Path("results") / "evaluation_metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n指標已存：{out}")
    print("混淆矩陣圖：runs/detect/week5_eval/confusion_matrix.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
