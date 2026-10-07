"""
Week 3 — 建立 YOLO 格式交通 Dataset（合成、可重現）
==================================================
這支 script 會生成一個「合成交通場景」資料集，用來示範 YOLO 的
Dataset / Annotation / Train-Validation split 流程。

為什麼用「合成」資料？
- 本機沒有大份量真實交通影片/標註，且要保證可重現、可測試。
- 此資料集用來把「訓練管線」跑通並產出 models/best.pt。
- 真正上場前，請把它換成真實交通資料集（見 README / 下方註解）。

YOLO 標註格式（每張圖一個 .txt，每行一個物件）：
  <class_id> <x_center> <y_center> <width> <height>
  座標都「歸一化」到 0~1（相對於圖片寬高）。

如何換成真實資料集：
  1) 收集行車影片/圖片，用 LabelImg / Roboflow 標註成相同 YOLO 格式。
  2) 放到 data/traffic_dataset/images/{train,val} 與 labels/{train,val}。
  3) 修改 data/dataset.yaml 的 names（類別名稱）即可。
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

CLASSES = ["car", "person", "truck", "bus"]

# 每類的「典型外觀」（BGR 顏色 / 寬度範圍），僅用於合成，不代表真實物體。
COLORS = {
    "car": ((30, 30, 200), (40, 70)),    # 紅色車
    "person": ((200, 180, 30), (12, 28)),  # 黃色人
    "truck": ((200, 30, 30), (60, 110)),   # 藍色卡車
    "bus": ((30, 200, 30), (70, 130)),     # 綠色巴士
}


def make_scene(w: int, h: int, rng: random.Random):
    """生成一張合成交通場景 + 對應 YOLO 標註字串清單。"""
    img = np.full((h, w, 3), 50, np.uint8)            # 柏油路灰
    cv2.rectangle(img, (0, h // 2), (w, h), (80, 80, 80), -1)  # 下半部亮一點（路面）
    anns: list[str] = []
    n_obj = rng.randint(2, 6)
    for _ in range(n_obj):
        cls = rng.choice(CLASSES)
        color, (min_s, max_s) = COLORS[cls]
        bw = rng.randint(min_s, max_s)
        bh = int(bw * rng.uniform(0.5, 0.9))
        x1 = rng.randint(0, max(1, w - bw))
        y1 = rng.randint(h // 2, max(h // 2 + 1, h - bh))
        x2, y2 = x1 + bw, y1 + bh
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        # 轉成 YOLO 格式：歸一化中心點 + 寬高
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        nw, nh = bw / w, bh / h
        anns.append(f"{CLASSES.index(cls)} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return img, anns


def main() -> None:
    ap = argparse.ArgumentParser(description="Week 3: 生成合成交通 Dataset")
    ap.add_argument("--out", default="data/traffic_dataset")
    ap.add_argument("--num", type=int, default=120, help="圖片總數")
    ap.add_argument("--size", type=int, default=416, help="圖片邊長")
    ap.add_argument("--seed", type=int, default=42, help="隨機種子（可重現）")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    base = Path(args.out)
    for split in ("train", "val"):
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)

    w = h = args.size
    for i in range(args.num):
        split = "val" if i % 5 == 0 else "train"  # 80 / 20 分割
        img, anns = make_scene(w, h, rng)
        imgp = base / "images" / split / f"img_{i:04d}.png"
        lblp = base / "labels" / split / f"img_{i:04d}.txt"
        cv2.imwrite(str(imgp), img)
        lblp.write_text(("\n".join(anns) + "\n") if anns else "")
    print(f"資料集已生成：{base}（{args.num} 張，train/val = 80/20）")


if __name__ == "__main__":
    main()
