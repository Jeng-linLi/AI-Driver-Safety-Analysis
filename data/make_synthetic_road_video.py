"""
產生一段「合成車道」測試影片，供 Week 6 Lane Detection 演示與測試用。
（因為這個 portfolio 沒有真實行車影片，我們用程式畫出會收斂到消失點的
兩條白色車道線 + 路面，讓 Canny + Hough 真的能偵測到線。）

執行：
    python data/make_synthetic_road_video.py --out results/videos/road_demo.mp4 --frames 120
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def make_frame(w: int, h: int, shift: float) -> np.ndarray:
    """畫一幀合成道路：天空（上）+ 路面（下）+ 兩條收斂車道線。"""
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    # 天空（上 55%）：淺灰藍
    frame[: int(h * 0.55), :] = (120, 140, 160)
    # 路面（下 45%）：深灰
    frame[int(h * 0.55):, :] = (60, 60, 60)

    # 消失點（畫面中央偏上），隨 shift 左右微小移動模擬方向盤修正
    vx = int(w * 0.5 + shift * w)
    vy = int(h * 0.55)

    # 兩條車道線：從畫面底部兩側收斂到消失點
    # 左線
    cv2.line(frame, (int(w * 0.15), h - 1), (vx - int(w * 0.05), vy), (235, 235, 235), 6)
    # 右線
    cv2.line(frame, (int(w * 0.85), h - 1), (vx + int(w * 0.05), vy), (235, 235, 235), 6)
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="產生合成車道測試影片")
    parser.add_argument("--out", default="results/videos/road_demo.mp4")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--w", type=int, default=640)
    parser.add_argument("--h", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (args.w, args.h))

    for i in range(args.frames):
        # 讓車道線左右小幅搖擺，模擬車輛微幅偏移
        shift = 0.06 * np.sin(i / 12.0)
        writer.write(make_frame(args.w, args.h, shift))

    writer.release()
    print(f"已產生：{args.out}（{args.frames} 幀, {args.w}x{args.h}）")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
