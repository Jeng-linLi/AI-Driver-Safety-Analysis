"""
Week 8 — Real-Time Integration（即時整合系統）
=============================================
把前面幾週的模組串成一條「即時管線」，對影片（或 webcam）逐幀輸出：
- Object Detection 的 Bounding Box + 類別 + Confidence（綠框）
- Lane Detection 的車道線（黃線）
- Risk Analysis 的風險等級橫幅（綠/橘/紅）
- FPS（每秒幀數）與幀數計數，放在左上角

這是一個「近即時（near-real-time）」的 prototype：在 CPU 上 YOLO 每幀推論
會有延遲，所以用 FPS 誠實呈現效能，而不是假裝能 30 FPS 即時跑。

用法：
    # 對影片做即時整合分析
    python src/realtime.py data/raw/your_driving_video.mp4 results/videos/realtime.mp4

    # 用 webcam（裝置編號 0）
    python src/realtime.py 0 results/videos/webcam_out.mp4
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
from ultralytics import YOLO

from detection import TARGET_CLASSES, load_model
from lane_detection import detect_lanes
from risk_analysis import COLOR, Detection, assess_risk

FONT = cv2.FONT_HERSHEY_SIMPLEX


def process_video(
    model: YOLO,
    input_path: str,
    output_path: str,
    conf: float = 0.25,
    max_frames: int = 600,
    use_lane: bool = True,
) -> dict:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"開啟影片/裝置失敗：{input_path}")

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps_src, (w, h))

    n = 0
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    fps_smooth = 0.0
    prev_t = time.time()

    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        # 1) Object Detection
        results = model(frame, conf=conf, verbose=False)[0]
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
            cv2.putText(frame, f"{name} {score:.2f}", (x1, max(0, y1 - 6)),
                        FONT, 0.5, (0, 255, 0), 2)

        # 2) Lane Detection（疊加黃線）
        if use_lane:
            frame = detect_lanes(frame)

        # 3) Risk Analysis（頂部橫幅）
        risk = assess_risk(dets, h, w)
        risk_counts[risk["level"]] += 1
        cv2.rectangle(frame, (0, 0), (w, 40), COLOR[risk["level"]], -1)
        cv2.putText(frame, f"RISK: {risk['level']} ({risk['score']:.2f})",
                    (10, 27), FONT, 0.7, (255, 255, 255), 2)

        # 4) FPS + 幀數
        now = time.time()
        dt = now - prev_t
        prev_t = now
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        fps_smooth = 0.9 * fps_smooth + 0.1 * inst_fps if fps_smooth else inst_fps
        cv2.putText(frame, f"FPS: {fps_smooth:.1f}  frame: {n}",
                    (w - 230, 27), FONT, 0.6, (255, 255, 255), 2)

        writer.write(frame)
        n += 1

    cap.release()
    writer.release()
    return {"frames": n, "risk_counts": risk_counts, "avg_fps": round(fps_smooth, 2)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 8: Real-Time Integration")
    parser.add_argument("input", help="輸入影片路徑或 webcam 編號（如 0）")
    parser.add_argument("output", help="輸出影片路徑")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--no-lane", action="store_true", help="關閉車道線")
    args = parser.parse_args()

    # webcam 編號直接當 int 傳入 VideoCapture
    try:
        src = int(args.input)
    except ValueError:
        src = args.input

    model = load_model(args.weights)
    info = process_video(model, src, args.output, args.conf, args.max_frames,
                         use_lane=not args.no_lane)
    print(f"完成：{args.output}")
    print("處理幀數：", info["frames"])
    print("平均 FPS：", info["avg_fps"])
    print("風險統計：", info["risk_counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
