# AI Driver Safety Analysis — Real-Time Road and Vehicle Detection

> 狀態：**WIP（進行中）— 目前進度 Week 8 / 12（接續 Week 9–12）**
> 這是一個 AI 大一學生的 Portfolio Project：建立一套完整的 AI workflow prototype
> （Data → Model → Training → Evaluation → Computer Vision → Risk Analysis → System Integration → Demo → GitHub）。
>
> ⚠️ **Disclaimer**：本專案是「學習用 prototype」，**不是**可上路的自駕車系統，
> 也**不宣稱**達到任何 Autonomous Driving 安全認證標準。

---

## 1. Project Overview

一套基於 Computer Vision 與 Object Detection 的駕駛安全分析系統：
從行車影片（或 webcam）中偵測道路物件、分析車道與基本道路風險，
並以即時 / 近即時方式呈現結果（Bounding Box、標籤、Confidence、車道線、Risk Level、FPS）。

## 2. Motivation

對 AI 初學者而言，最好的學習方式就是「邊做邊學」一個真實、能跑、能放上 GitHub 的專案。
本專案串起一條完整的 AI 工程鏈：從讀資料、訓練模型、評估、到做成 demo 與技術報告。

## 3. Features

- [x] 讀取圖片 / 影片 / 儲存影片（Week 1）
- [x] Object Detection（car / person / truck / bus ...）— Week 2 ✅
- [x] 自建交通 Dataset（YOLO 格式 + dataset.yaml）— Week 3 ✅
- [x] Model Fine-tuning（YOLOv8n + 合成資料，mAP50 0.958）— Week 4 ✅
- [x] Model Evaluation（mAP50 0.958, Confusion Matrix）— Week 5 ✅
- [x] Lane Detection（Canny + Hough + ROI）— Week 6 ✅
- [x] Risk Analysis（LOW / MEDIUM / HIGH，啟發式）— Week 7 ✅
- [x] Real-Time Integration（Detection + Lane + Risk + FPS）— Week 8 ✅
- [ ] Risk Analysis（LOW / MEDIUM / HIGH）— Week 7
- [ ] Real-Time Pipeline — Week 8
- [ ] Experiments Log — Week 9
- [ ] Web Demo（Streamlit）— Week 10
- [ ] Engineering（Dockerfile / testing）— Week 11
- [ ] Portfolio（README / Report / Demo）— Week 12

## 4. System Architecture

```
Video/Webcam
   └─> [1] Object Detection (YOLOv8n)  → 綠框 + 類別 + Confidence
   └─> [2] Lane Detection (Canny+Hough) → 黃線
   └─> [3] Risk Analysis (heuristic)    → 風險橫幅 LOW/MEDIUM/HIGH
   └─> [4] Visualization (FPS + 幀數)   → 輸出影片 / webcam
（Week 8 已把 [1][2][3][4] 串成單一即時管線 src/realtime.py）
```

## 5. Dataset

目前使用「合成交通場景」資料集（Week 3 生成，YOLO 格式）：
- 4 類：car / person / truck / bus
- 120 張（train 96 / val 24），416×416
- 由 `data/make_synthetic_dataset.py` 重新產生（已 gitignore，不進 Git）

> 這是為了把「訓練管線」跑通的可重現資料。
> 正式版請替換為真實交通資料集（用 LabelImg / Roboflow 標註成相同 YOLO 格式即可）。

## 6. Model

- 骨幹：YOLOv8n（nano，約 3.0M 參數，適合 prototype / CPU）
- 方法：Transfer Learning — 以官方預訓練 YOLOv8n 為起點，在自有資料上 Fine-tuning
- 類別（4）：car / person / truck / bus
- 權重：models/best.pt（由 src/train.py 訓練產生，已 gitignore）

> 目前權重在「合成交通資料集」上訓練，mAP 很高是因為合成資料單純；
> 換成真實交通資料後指標會更貼近實際，也更具有意義。

## 7. Training

```bash
python src/train.py --epochs 20 --batch 16 --imgsz 416
```

- Epochs: 20 | Batch: 16 | Img size: 416 | Optimizer: auto | Device: CPU
- 資料：data/dataset.yaml（合成資料集，120 張，80/20 split）
- 產出：runs/detect/week4_traffic/（含曲線圖、驗證結果），並複製 best.pt → models/best.pt

驗證集結果（best.pt）：

| 類別 | Precision | Recall | mAP@.5 | mAP@.5-.95 |
|------|-----------|--------|--------|------------|
| all  | 0.953 | 0.927 | 0.958 | 0.867 |
| car  | 1.000 | 0.939 | 0.955 | 0.916 |
| person | 1.000 | 0.889 | 0.952 | 0.741 |
| truck | 0.959 | 0.880 | 0.932 | 0.859 |
| bus  | 0.854 | 1.000 | 0.992 | 0.950 |

## 8. Evaluation

用 `src/evaluate.py` 在驗證集（24 張）上評估 models/best.pt：

| 指標 | 數值 |
|------|------|
| Precision | 0.953 |
| Recall | 0.927 |
| mAP@.5 | 0.958 |
| mAP@.5-.95 | 0.867 |

每類 mAP@.5-.95：car 0.916 / person 0.741 / truck 0.859 / bus 0.950

- Confusion Matrix：`runs/detect/week5_eval/confusion_matrix.png`
- 完整指標：`results/evaluation_metrics.json`

> 這些數字是在「合成資料」上得到的。因為合成場景單純（純色矩形），
> 模型很容易學，所以指標偏高；這不代表它能在真實道路上同樣準確。
> 換成真實交通資料後，指標會更貼近實際，弱點也會更明顯
> （例如 person 的 mAP 明顯低於車輛，是常見的難點）。

## 9. Demo

規劃中（Week 8 / Week 10）。

## 10. Installation

```bash
# 1. 建立虛擬環境（推薦，避免污染系統 Python）
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. 安裝依賴
pip install -r requirements.txt
```

## 11. Usage（Week 1）

```bash
# 讀取並顯示影片資訊（幀數 / FPS / 解析度）
python src/video_io.py video data/raw/your_driving_video.mp4

# 處理影片並存檔（這裡只是示範轉灰階）
python src/video_io.py process data/raw/your_driving_video.mp4 results/videos/out.mp4 --grayscale

# 播放影片（需要你自己的電腦有圖形介面）
python src/video_io.py show data/raw/your_driving_video.mp4
```

> 還沒有行車影片？把任何一段 mp4 放進 `data/raw/` 就能先試跑。
> （注意：`data/raw/*.mp4` 已被 .gitignore 忽略，不會進 Git。）

### Week 2 — Object Detection

```bash
# 對行車影片做 YOLO 偵測，輸出帶 Bounding Box + 標籤 + Confidence 的影片
python src/detection.py data/raw/your_driving_video.mp4 results/videos/detected.mp4

# 提高嚴格度（只保留信心 >= 0.5 的預測）
python src/detection.py data/raw/your_driving_video.mp4 results/videos/detected.mp4 --conf 0.5
```

> 僅關注道路安全相關類別：person / bicycle / car / motorcycle / bus / truck / traffic light / stop sign。

### Week 5 — Evaluation

```bash
python src/evaluate.py --weights models/best.pt
```

> 輸出指標到 results/evaluation_metrics.json，混淆矩陣到 runs/detect/week5_eval/confusion_matrix.png。

### Week 6 — Lane Detection

```bash
# 用傳統 CV（Canny + Hough + ROI）找出車道線，輸出疊加黃線的影片
python src/lane_detection.py data/raw/your_driving_video.mp4 results/videos/lane_detected.mp4

# 沒有行車影片？先產一段合成車道影片來試跑
python data/make_synthetic_road_video.py --out results/videos/road_demo.mp4 --frames 120
python src/lane_detection.py results/videos/road_demo.mp4 results/videos/lane_detected.mp4
```

> 這週「不用 AI 模型」，純 OpenCV 經典演算法：先找邊緣、只看車道區域、再把邊緣點連成直線。
> 對清晰 / 合成路面效果最好；真實雨天、Shadow、彎道需要更多前處理。

### Week 7 — Risk Analysis

```bash
# 對行車影片做偵測 + 風險分析，輸出帶「風險橫幅」的影片
python src/risk_analysis.py data/raw/your_driving_video.mp4 results/videos/risk_demo.mp4 --weights yolov8n.pt
```

風險規則（啟發式、可解釋）：

| 等級 | 觸發條件 |
|------|----------|
| 🔴 HIGH | 近距離出現弱勢用路人（person / bicycle） |
| 🟠 MEDIUM | 近距離車輛，或場景中有（較遠的）弱勢用路人 |
| 🟢 LOW | 未偵測到相關道路物件 |

> ⚠️ 這是「學習用」簡化規則，絕非真實自駕車安全判斷。真實系統還需要距離估測、追蹤、軌跡預測。

### Week 8 — Real-Time Integration

```bash
# 把 Detection + Lane + Risk 串成即時管線，輸出帶 FPS / 風險橫幅的影片
python src/realtime.py data/raw/your_driving_video.mp4 results/videos/realtime.mp4 --weights yolov8n.pt

# 用 webcam 即時分析（裝置編號 0）
python src/realtime.py 0 results/videos/webcam_out.mp4

# 效能優化：降推理解析度 + 跳幀（CPU 上 FPS 明顯提升）
python src/realtime.py data/raw/your_driving_video.mp4 results/videos/realtime.mp4 \
    --imgsz 320 --detect-every 2
```

實測效能（CPU、合成 640×360 影片，30 幀平均）：

| 配置 | 平均 FPS |
|------|----------|
| imgsz=640, detect-every=1（原版） | ~21 |
| imgsz=320, detect-every=1 | ~32（≈1.5×） |
| imgsz=320, detect-every=2 | ~108* |

> \* 合成片段幾乎無物件，跳幀主要省下「推論」開銷，因此 FPS 很高；真實繁忙畫面會介於兩者之間，但仍明顯快於原版。
> 這是「近即時」prototype：合成畫面簡單所以 FPS 偏高；真實影片物件多、解析度高時會變慢。要真正即時需上 GPU / 更小模型 / 或只對關鍵幀偵測。
>
> 程式碼層面已重構：`detect_and_draw()`（偵測+畫框）與 `resolve_data_yaml()` 抽出到 `src/detection.py` / `src/common.py` 共用，消除 risk_analysis / realtime / train / evaluate 之間的重複碼。

## 12. Results

陸續補上（Week 5 / Week 9）。

## 13. Limitations

- 僅為學習用 prototype，不具備任何安全認證。
- 依賴公開 pretrained model，準確度受限於訓練資料。

## 14. Future Work

- 加入更精準的距離估測（depth estimation）
- 擴充 Dataset 與類別
- 部署到邊緣裝置

## 15. Author

Johnny, Jeng-lin Li — AI 大一學生 / Portfolio Project

---

### 📅 12-Week Roadmap

| Week | 主題 | 狀態 |
|------|------|------|
| 1 | Python / OpenCV / Git | ✅ |
| 2 | Object Detection (YOLO) | ✅ |
| 3 | Dataset | ✅ |
| 4 | Model Fine-tuning | ✅ |
| 5 | Model Evaluation | ✅ |
| 6 | Lane Detection | ✅ |
| 7 | Risk Analysis | ✅ |
| 8 | Real-Time Integration | ✅ |
| 9 | Experiments | ⬜ |
| 10 | Web Demo (Streamlit) | ⬜ |
| 11 | Engineering | ⬜ |
| 12 | Portfolio | ⬜ |
