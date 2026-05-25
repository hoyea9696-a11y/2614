# Propose Spec - System Design

## 1. 介面設計 (UI/UX)
- **風格**: 科技感與玻璃擬態 (Glassmorphism)。背景為極深色，卡片與導覽列採用 **科技淺色 (銀白/淡藍半透明)** 以增強對比度。
- **左側邊欄**: 包含以下獨立分頁：
    - `單張影像驗證 (Single Verify)`
    - `批量資料夾驗證 (Batch Verify)`
    - `模型訓練 (Training)`
    - **主螢幕**: 動態過渡效果，根據選擇的分頁顯示對應內容。
- **互動細節**:
    - 按鈕懸停有光輝效果 (Glow effect)。
    - 上傳區塊支援拖放，並伴隨掃描線動畫。
    - 訓練進度條具備流光動畫效果。

## 2. 功能模組設計

### A. 檔案管理系統
- 掃描並過濾 `.jpg`, `.bmp`, `.png`。
- 資料集驗證：檢查 `images` 與 `labels` 是否成對，自動掃描目錄中的 `.yaml`。

- **參數設定**: 
    - **基礎**: Epochs, Batch size, Image size, Optimizer。
    - **推理參數**: 
        - **TTA**: 測試時增強，提升精度。
        - **過濾強度 (Conf)**: 信心值閾值。
        - **敏感度 (IoU)**: 重疊區域閾值。
- **圖表連結**: 利用 WebSocket 或後端 Poll 機制實時更新損失函數 (Loss) 與精準度 (mAP)。

### C. 推理引擎 (Inference)
- 內建模型載入。驗證頁面支援 S, M, C, E 型號即時切換。
- **偵測模式**: 
    - **單張分頁**: 獨立佈局，重點展示大圖預覽與檢測結果表。
    - **批量分頁**: 獨立佈局，包含 20 張縮圖分頁與批量控制項。
- **結果數據區塊**: 預覽圖下方展示列表，包含：名稱、信心值、X1, Y1, X2, Y2 座標。
- **智能硬體偵測**: 訓練前自動執行 `torch.cuda.is_available()` 檢查。
    - **GPU 模式**: 偵測到 CUDA 時，強制指定 `device=0` 以釋放最大性能。
    - **CPU 模式**: 偵測失敗時，自動回退至 `device=cpu`，確保訓練流程不因驅動問題卡死。
- **UUID 命名機制**: 全面取代原始檔名，解決 Windows 環境下的文件鎖定與併發讀寫報錯。
- **集中式事件管理**: 移除冗餘監聽器，確保單一入口。
- **高對比度數據展示**: 
    - 使用深色背景與螢光色系指標。
    - 狀態標籤統一採用**白色字體**搭配亮色背景，優化對比。
- **UI 設計系統**: 
    - 整合代碼高亮預覽區，模擬專業級工業監控儀表板。
    - 側邊欄底部內建「硬體狀態監控」，具備非同步容錯邏輯，確保資訊獲取不影響訓練啟動。

## 3. API 端點定義 (FastAPI)
- `POST /upload/image`: 上傳單張圖片並返回推理結果。
- `POST /api/verify/single`: 處理單張或批次循環推理。
- `POST /api/train/start`: 啟動背景訓練任務。
- `GET /api/train/status`: 獲取訓練即時數據。
- `POST /api/clear_data`: 清除 `uploads` 與 `outputs` 下的所有實體檔案與狀態。
- `POST /inference/verify`: 匯入模型並對圖片進行驗證。

## 4. 目錄結構
```text
/
├── spec/               # SDD 規格文件
├── server/             # FastAPI 後端
│   ├── main.py
│   ├── trainer.py
│   └── inferencer.py
├── web/                # React 前端
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── App.jsx
├── scripts/            # 輔助腳本
│   └── create_shortcut.ps1 # 建立桌面捷徑
├── models/             # 存放訓練好的模型與預訓練模型
├── data/               # 存放上傳的資料集與驗證圖片
├── setup.bat           # 建立虛擬環境、安裝依賴與建立捷徑 (Windows)
└── run.py              # 啟動腳本 (同時啟動後端並開啟瀏覽器)
```
