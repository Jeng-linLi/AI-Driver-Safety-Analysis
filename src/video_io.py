"""
Week 1 — 基礎 Computer Vision I/O
=================================
這是專案的第一支可執行 script。

目標：確認 Python + OpenCV 環境可以正確「讀取圖片、讀取影片、儲存影片」。

概念（大一也能懂）：
- 圖片在電腦裡其實是一個「數字矩陣」（numpy array）。
  OpenCV 讀進來的矩陣形狀是 (高度, 寬度, 3)，3 代表 B/G/R 三個顏色通道。
- 影片就是「很多張圖片（frame）依照順序快速播放」。
  我們用 VideoCapture 一張一張讀，用 VideoWriter 一張一張寫。
"""
from __future__ import annotations

import argparse
import sys

import cv2
import numpy as np


def read_image(path: str) -> np.ndarray:
    """讀取一張圖片，回傳 BGR 格式的 numpy 陣列。

    Raises:
        FileNotFoundError: 檔案不存在或不是圖片格式。
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"讀不到圖片：{path}")
    return img


def read_video_info(path: str) -> dict:
    """開啟影片並回傳基本資訊（幀數 / FPS / 寬高）。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"開啟影片失敗：{path}")
    info = {
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return info


def process_video(
    input_path: str,
    output_path: str,
    max_frames: int = 300,
    grayscale: bool = False,
) -> int:
    """讀取影片、逐幀處理、寫出新的影片檔。

    這裡的「處理」只是示範（轉灰階）；之後的週次會換成
    Object Detection / Lane Detection 等真正的 AI 分析。

    Returns:
        實際寫出的幀數。
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
    while n < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if grayscale:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        writer.write(frame)
        n += 1

    cap.release()
    writer.release()
    return n


def show_video(path: str, max_frames: int = 300) -> None:
    """播放影片。需要圖形介面（你自己的電腦），headless 環境會自動跳過。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"開啟影片失敗：{path}")
    try:
        n = 0
        while n < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("AI Driver Safety — preview", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            n += 1
    except cv2.error as e:
        print(f"[跳過] 這個環境沒有圖形介面，無法顯示視窗：{e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main() -> int:
    parser = argparse.ArgumentParser(description="Week 1: 影像 / 影片 I/O 練習")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_img = sub.add_parser("image", help="讀取並顯示圖片資訊")
    p_img.add_argument("path")

    p_vid = sub.add_parser("video", help="讀取並顯示影片資訊")
    p_vid.add_argument("path")

    p_proc = sub.add_parser("process", help="處理影片並存檔")
    p_proc.add_argument("input")
    p_proc.add_argument("output")
    p_proc.add_argument("--max-frames", type=int, default=300)
    p_proc.add_argument("--grayscale", action="store_true")

    p_show = sub.add_parser("show", help="播放影片（需圖形介面）")
    p_show.add_argument("path")
    p_show.add_argument("--max-frames", type=int, default=300)

    args = parser.parse_args()

    if args.cmd == "image":
        img = read_image(args.path)
        print(f"圖片讀取成功：shape={img.shape}（高, 寬, 通道）")
    elif args.cmd == "video":
        info = read_video_info(args.path)
        print("影片資訊：", info)
    elif args.cmd == "process":
        n = process_video(args.input, args.output, args.max_frames, args.grayscale)
        print(f"已處理並存檔：{args.output}（{n} 幀）")
    elif args.cmd == "show":
        show_video(args.path, args.max_frames)
    return 0


if __name__ == "__main__":
    sys.exit(main())
