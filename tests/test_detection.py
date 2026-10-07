"""
Week 2 基本測試（smoke test）：
驗證 YOLO 模型能載入、detect_video 能對影片逐幀推論並存檔。
注意：首次執行會自動下載 yolov8n.pt（約 6MB），需要網路。
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import detection  # noqa: E402


def _dummy_video(path: str, frames: int = 5, w: int = 160, h: int = 120) -> None:
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(frames):
        writer.write(np.full((h, w, 3), (i * 40) % 255, dtype=np.uint8))
    writer.release()


def test_detect_video_runs(tmp_path):
    model = detection.load_model("yolov8n.pt")  # 首次下載權重
    inp = tmp_path / "in.mp4"
    out = tmp_path / "out.mp4"
    _dummy_video(str(inp))
    info = detection.detect_video(model, str(inp), str(out), max_frames=5)
    assert info["frames"] == 5
    assert os.path.exists(out)
