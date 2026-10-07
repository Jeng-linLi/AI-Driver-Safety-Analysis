"""
Week 7 — Risk Analysis（風險分析）
=================================
把 Week 2 的「物件偵測」與 Week 6 的「車道線」結果，合併成一個簡單的
駕駛風險等級：LOW / MEDIUM / HIGH。

這是一套「啟發式（heuristic）」規則，不是神經網絡，目的是：
- 讓大一學生理解「怎麼把偵測結果轉成有意義的警示」。
- 作為 Week 8 即時系統的決策模組。

風險規則（可解釋、可調）：
- HIGH：畫面中出現「行人 / 自行車」這類弱勢用路人，且距離近（框大、貼近車頭）。
- MEDIUM：出現車輛且距離偏近；或出現行人但距離較遠。
- LOW：沒有偵測到相關物件，或物件都很遠。

⚠️ 這只是「學習用」的簡化規則，絕對不能當作真實自駕車的安全判斷。
真實系統需要距離估測（depth）、追蹤（tracking）、軌跡預測等。
"""
from __future__ import annotations

import argparse
import sys

import cv2
from ultralytics import YOLO

from detection import TARGET_CLASSES, Detection, detect_and_draw, load_model

# 「弱勢用路人」：一出現就大幅提高風險
VULNERABLE = {"person", "bicycle"}

# 風險等級顏色（BGR）
COLOR = {"LOW": (0, 200, 0), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255)}


def _area_fraction(bbox: tuple[int, int, int, int], frame_h: int, frame_w: int) -> float:
    x1, y1, x2, y2 = bbox
    return ((x2 - x1) * (y2 - y1)) / (frame_w * frame_h)


def _is_close(bbox: tuple[int, int, int, int], frame_h: int, frame_w: int) -> bool:
    """用「框佔畫面比例」與「是否貼近車頭（畫面下方）」粗略判斷距離近。"""
    x1, y1, x2, y2 = bbox
    close_by_area = _area_fraction(bbox, frame_h, frame_w) > 0.05
    close_by_bottom = y2 > frame_h * 0.75  # 框的底部落在最下 1/4 → 離車近
    return close_by_area or close_by_bottom


def assess_risk(detections: list[Detection], frame_h: int, frame_w: int) -> dict:
    """根據偵測結果評估風險等級與原因。

    Returns:
        {"level": "LOW"|"MEDIUM"|"HIGH", "reasons": [str, ...], "score": float}
        score 為 0~1 的風險分數，方便 Week 8 畫進度條或排序。
    """
    reasons: list[str] = []
    level = "LOW"
    score = 0.0

    for d in detections:
        name = d.name
        close = _is_close(d.bbox, frame_h, frame_w)
        if name in VULNERABLE:
            if close:
                level = "HIGH"
                score = max(score, 0.9)
                reasons.append(f"近距離弱勢用路人：{name}（conf {d.score:.2f}）")
            else:
                if level != "HIGH":
                    level = "MEDIUM"
                score = max(score, 0.6)
                reasons.append(f"場景中有弱勢用路人：{name}（較遠）")
        elif name in {"car", "truck", "bus", "motorcycle"}:
            if close:
                if level == "LOW":
                    level = "MEDIUM"
                score = max(score, 0.5)
                reasons.append(f"近距離車輛：{name}（conf {d.score:.2f}）")

    if not reasons:
        reasons.append("未偵測到相關道路物件")

    return {"level": level, "reasons": reasons, "score": round(score, 2)}


def process_video(
    model: YOLO,
    input_path: str,
    output_path: str,
    conf: float = 0.25,
    max_frames: int = 600,
    imgsz: int = 640,
) -> dict:
    """對影片逐幀做偵測 + 風險分析：畫框 + 頂部風險橫幅 → 寫入新影片。"""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"開啟影片失敗：{input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    counts: dict[str, int] = {}
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    n = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break

        # 推論 + 畫框（共用 detection.detect_and_draw）
        frame, dets = detect_and_draw(model, frame, conf, imgsz)
        for d in dets:
            counts[d.name] = counts.get(d.name, 0) + 1

        risk = assess_risk(dets, h, w)
        risk_counts[risk["level"]] += 1

        # 頂部風險橫幅
        color = COLOR[risk["level"]]
        cv2.rectangle(frame, (0, 0), (w, 40), color, -1)
        text = f"RISK: {risk['level']}  ({risk['score']:.2f})  | " + "; ".join(risk["reasons"][:2])
        cv2.putText(frame, text, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)

        writer.write(frame)
        n += 1

    cap.release()
    writer.release()
    return {"frames": n, "detections": counts, "risk_counts": risk_counts}


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 7: Risk Analysis")
    parser.add_argument("input", help="輸入影片路徑")
    parser.add_argument("output", help="輸出影片路徑")
    parser.add_argument("--weights", default="yolov8n.pt", help="YOLO 權重")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640, help="推理解析度（越小越快）")
    parser.add_argument("--max-frames", type=int, default=600)
    args = parser.parse_args()

    model = YOLO(args.weights)
    info = process_video(model, args.input, args.output, args.conf, args.max_frames, args.imgsz)
    print(f"完成：{args.output}")
    print("處理幀數：", info["frames"])
    print("各類別偵測次數：", info["detections"])
    print("風險統計：", info["risk_counts"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
