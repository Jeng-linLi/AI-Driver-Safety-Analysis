"""Week 8 — Real-Time Integration 整合冒煙測試。"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "data"))

from detection import load_model  # noqa: E402
from realtime import process_video  # noqa: E402
from make_synthetic_road_video import make_frame  # noqa: E402


def _make_clip(path: Path, frames: int = 5):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30, (640, 360))
    for i in range(frames):
        writer.write(make_frame(640, 360, shift=0.05 * i))
    writer.release()


def test_realtime_pipeline_runs():
    clip = ROOT / "results" / "videos" / "_rt_smoke.mp4"
    out = ROOT / "results" / "videos" / "_rt_out.mp4"
    _make_clip(clip, frames=5)

    model = load_model("yolov8n.pt")
    info = process_video(model, str(clip), str(out), max_frames=5)

    assert info["frames"] == 5
    assert set(info["risk_counts"]) == {"LOW", "MEDIUM", "HIGH"}
    assert out.exists() and out.stat().st_size > 0

    clip.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
