"""Week 7 — Risk Analysis 單元測試（不依賴 YOLO，直接測 assess_risk 規則）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from risk_analysis import Detection, assess_risk  # noqa: E402

H, W = 360, 640


def test_low_when_empty():
    r = assess_risk([], H, W)
    assert r["level"] == "LOW"
    assert r["score"] == 0.0


def test_high_when_close_pedestrian():
    # 貼近車頭（底部在下 1/4）的行人 → HIGH
    det = Detection("person", (280, 300, 360, 350), 0.9)
    r = assess_risk([det], H, W)
    assert r["level"] == "HIGH"
    assert r["score"] >= 0.9


def test_medium_when_far_pedestrian():
    # 上方小框的行人（較遠） → MEDIUM
    det = Detection("person", (300, 60, 330, 110), 0.8)
    r = assess_risk([det], H, W)
    assert r["level"] == "MEDIUM"


def test_medium_when_close_vehicle():
    # 近距離車輛（框大） → 至少 MEDIUM
    det = Detection("car", (100, 250, 540, 350), 0.7)
    r = assess_risk([det], H, W)
    assert r["level"] in {"MEDIUM", "HIGH"}


def test_reasons_explained():
    det = Detection("bicycle", (200, 280, 260, 345), 0.85)
    r = assess_risk([det], H, W)
    assert len(r["reasons"]) > 0
    assert any("bicycle" in reason or "自行車" in reason for reason in r["reasons"])
