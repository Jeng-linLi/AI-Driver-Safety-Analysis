"""
Week 1 基本測試：驗證「讀圖 / 讀影片 / 存影片」這條 I/O 管線可以跑通。
執行：pytest tests/
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

# 讓 test 能 import src/video_io.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import video_io  # noqa: E402


def _make_dummy_video(path: str, frames: int = 10, w: int = 64, h: int = 64) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(frames):
        writer.write(np.full((h, w, 3), (i * 20) % 255, dtype=np.uint8))
    writer.release()


def test_read_video_info(tmp_path):
    p = tmp_path / "dummy.mp4"
    _make_dummy_video(str(p))
    info = video_io.read_video_info(str(p))
    assert info["frame_count"] >= 1
    assert info["width"] == 64
    assert info["height"] == 64


def test_process_video(tmp_path):
    inp = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _make_dummy_video(str(inp))
    n = video_io.process_video(str(inp), str(out), max_frames=5)
    assert n == 5
    assert os.path.exists(out)


def test_read_image(tmp_path):
    p = tmp_path / "img.png"
    cv2.imwrite(str(p), np.zeros((32, 32, 3), dtype=np.uint8))
    img = video_io.read_image(str(p))
    assert img.shape == (32, 32, 3)
