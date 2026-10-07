"""Week 6 — Lane Detection 冒煙測試。"""
import sys
from pathlib import Path

import cv2
import numpy as np

# 讓 test 能 import src/ 下的模組
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from lane_detection import detect_lanes  # noqa: E402
from make_synthetic_road_video import make_frame  # noqa: E402


def test_detect_lanes_finds_lines():
    """在合成道路畫面上，應能偵測並疊加黃色車道線。"""
    frame = make_frame(640, 360, shift=0.0)
    out = detect_lanes(frame)
    assert out.shape == frame.shape
    # 黃線在 BGR 中為 (0,255,255)
    yellow = cv2.inRange(out, (0, 255, 255), (0, 255, 255))
    assert cv2.countNonZero(yellow) > 0


def test_detect_lanes_returns_rgb_image():
    """輸入與輸出都應是 3 通道 BGR 影像。"""
    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    out = detect_lanes(frame)
    assert out.ndim == 3 and out.shape[2] == 3
