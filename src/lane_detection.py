"""
Week 6 — Lane Detection（傳統 Computer Vision）
================================================
用「傳統影像處理」方法找出車道線，並把線疊加回畫面上：
- Canny Edge Detection：先找畫面中的邊緣（亮度變化大的地方）。
- Region of Interest (ROI)：只保留畫面下半部的「梯形區域」，
  因為車道線通常出現在車子前方的地面，不在天空。
- Hough Transform：把邊緣點組合成一條條直線（車道線近似直線）。

概念（大一也能懂）：
- 我們「不」用 AI 模型來做這週，而是用 OpenCV 的經典演算法。
  這能讓你理解：在深度學習之前，電腦視覺是怎麼處理「線」這種幾何特徵的。
- 這是 Self-Driving 教科書裡最經典的入門做法（Udacity / OpenCV 範例）。
- 缺點：對光線、路面、彎道很敏感；真實系統通常會換成深度學習（如 LaneNet）。

注意：這套方法在「合成 / 清晰」的車道影片上效果最好；
真實行車影片（雨天、Shadow、斑馬線）需要更多前處理（透視變換、曲線擬合）。
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np


def _region_of_interest(edges: np.ndarray) -> np.ndarray:
    """只保留畫面下半部的梯形 ROI（車道線通常出現在這裡）。"""
    h, w = edges.shape
    mask = np.zeros_like(edges)
    # 梯形：左、右、上、下四點，把天空與遠景切掉
    polygon = np.array(
        [
            [
                (0, h),
                (int(w * 0.45), int(h * 0.55)),
                (int(w * 0.55), int(h * 0.55)),
                (w, h),
            ]
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


def detect_lanes(
    frame: np.ndarray,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 50,
    min_line_length: int = 40,
    max_line_gap: int = 150,
    use_roi: bool = True,
) -> np.ndarray:
    """對單一畫面做車道線偵測，回傳「疊加黃色車道線」的畫面。

    Args:
        frame: BGR 彩色影像（來自 cv2.VideoCapture.read）。
    Returns:
        疊加車道線後的 BGR 影像。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 1) 模糊：先降噪，避免 Canny 抓到太多雜訊邊緣
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # 2) Canny 邊緣偵測
    edges = cv2.Canny(blur, canny_low, canny_high)
    # 3) ROI：只看車道區域
    if use_roi:
        edges = _region_of_interest(edges)
    # 4) Hough Transform：從邊緣點找出直線
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )

    output = frame.copy()
    if lines is not None:
        for line in lines:
            # OpenCV 5 與部分版本回傳 (N,1,4)，舊版回傳 (N,4)：統一攤平
            x1, y1, x2, y2 = line.reshape(-1)[:4].astype(int)
            # BGR 黃色 (0,255,255)，線寬 4
            cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 4)
    return output


def process_video(
    input_path: str,
    output_path: str,
    max_frames: int = 600,
    **kwargs,
) -> dict:
    """對影片逐幀做車道線偵測：畫線 → 寫入新影片。

    Returns:
        {"frames": 處理幀數, "frames_with_lanes": 偵測到線的幀數}
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"開啟影片失敗：{input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    n = 0
    frames_with_lanes = 0
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        out = detect_lanes(frame, **kwargs)
        # 粗略統計：這一幀有沒有畫到黃線（BGR 中 B==0 且 G==R==255）
        yellow = cv2.inRange(out, (0, 255, 255), (0, 255, 255))
        if int(cv2.countNonZero(yellow)) > 0:
            frames_with_lanes += 1
        writer.write(out)
        n += 1

    cap.release()
    writer.release()
    return {"frames": n, "frames_with_lanes": frames_with_lanes}


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 6: Lane Detection (Canny + Hough)")
    parser.add_argument("input", help="輸入影片路徑")
    parser.add_argument("output", help="輸出影片路徑")
    parser.add_argument("--max-frames", type=int, default=600)
    parser.add_argument("--no-roi", action="store_true", help="關閉 ROI 遮罩")
    args = parser.parse_args()

    info = process_video(
        args.input,
        args.output,
        max_frames=args.max_frames,
        use_roi=not args.no_roi,
    )
    print(f"完成：{args.output}")
    print("處理幀數：", info["frames"])
    print("偵測到車道線的幀數：", info["frames_with_lanes"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
