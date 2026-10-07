"""
Week 2 — Object Detection with YOLO
====================================
用「預訓練 YOLO」模型偵測影片中的道路物件（car / person / truck / bus ...），
把 Bounding Box + 類別標籤 + Confidence Score 畫到畫面上，並存成新影片。

概念（大一也能懂）：
- YOLO = You Only Look Once，一種 Object Detection 模型。
  它「一次看完整張圖」，直接預測：框在哪裡 + 是什麼 + 有多有信心。
- Bounding Box = 用矩形框把偵測到的物件框起來（座標 x1,y1,x2,y2）。
- Confidence Score = 模型對這個預測的把握度（0~1，越大越可信）。
- Pretrained（預訓練）= 別人已經用大規模資料（COCO，80 類）訓練好的模型，
  我們先把這個現成模型拿來用，之後 Week 4 才會用自己的資料微調（Fine-tuning）。

關於類別：COCO 有 80 類，我們只挑跟「道路安全」相關的類別來畫框與統計。
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import cv2
from ultralytics import YOLO

# 只關注與道路安全相關的 COCO 類別（其餘不畫框、不統計）
TARGET_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
}


@dataclass
class Detection:
    """單一偵測結果：類別名稱、Bounding Box (x1,y1,x2,y2)、Confidence。"""

    name: str
    bbox: tuple[int, int, int, int]
    score: float


def detect_and_draw(model: YOLO, frame, conf: float = 0.25, imgsz: int = 640):
    """對單一畫面做 YOLO 推論，並直接在畫面上畫綠框 + 標籤。

    Args:
        imgsz: 推理解析度（越小越快；320 在 CPU 上約比 640 快一倍，精度略降）。
    Returns:
        (frame, detections)：畫好框的畫面，與 Detection 清單（供 Risk / 統計使用）。
    """
    results = model(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    dets: list[Detection] = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        name = model.names[cls_id]
        if name not in TARGET_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        score = float(box.conf[0])
        dets.append(Detection(name, (x1, y1, x2, y2), score))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{name} {score:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return frame, dets


def load_model(weights: str = "yolov8n.pt") -> YOLO:
    """載入 YOLO 模型；首次執行會自動下載預訓練權重（約 6MB）。"""
    return YOLO(weights)


def detect_video(
    model: YOLO,
    input_path: str,
    output_path: str,
    conf: float = 0.25,
    max_frames: int = 600,
    imgsz: int = 640,
) -> dict:
    """對影片逐幀做 Object Detection：畫框 → 寫入新影片。

    Args:
        conf: 只保留 Confidence >= conf 的預測（門檻越高越嚴格）。
        imgsz: 推理解析度（越小越快；見 detect_and_draw）。
    Returns:
        {"frames": 處理幀數, "detections": 各類別被偵測到的總次數}
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"開啟影片失敗：{input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    counts: dict[str, int] = {}
    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        # 推論 + 畫框（共用 detect_and_draw）
        frame, dets = detect_and_draw(model, frame, conf, imgsz)
        for d in dets:
            counts[d.name] = counts.get(d.name, 0) + 1

        writer.write(frame)
        n += 1

    cap.release()
    writer.release()
    return {"frames": n, "detections": counts}


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 2: YOLO Object Detection")
    parser.add_argument("input", help="輸入影片路徑")
    parser.add_argument("output", help="輸出影片路徑")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLO 權重")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence 門檻")
    parser.add_argument("--imgsz", type=int, default=640, help="推理解析度（越小越快）")
    parser.add_argument("--max-frames", type=int, default=600)
    args = parser.parse_args()

    model = load_model(args.weights)
    info = detect_video(model, args.input, args.output, args.conf, args.max_frames, args.imgsz)
    print(f"完成：{args.output}")
    print("處理幀數：", info["frames"])
    print("各類別偵測次數：", info["detections"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
