# AI Detector 實作計畫 (Implementation Plan)

## 1. 專案概述
本專案為一個 AI 影像平台 (AI Detector)，整合 YOLO 推理能力與直覺管理介面，達成「數據管理 -> 精確標記 -> 模型訓練 -> 驗證佈署」的完整自動化生命週期。支援用戶上傳資料集、AI 輔助自動標記、模型訓練與深度效能診斷。

## 2. 系統架構
*   **後端框架**: FastAPI (Python)
*   **AI 引擎**: Ultralytics YOLO (訓練與推理) + LLM (Gemini/Groq 用於 AI 助理)
*   **前端介面**: HTML + CSS + JS (無框架，原生 JavaScript 搭配 Chart.js)
*   **伺服器架構**: 多埠口設計 (Port 8000 主入口，8001-8003 負責各子系統 API，透過 `window.postMessage` 進行跨頁面狀態通訊)

## 3. 模組實作細節

### 3.1 資料集管理模組 (Dataset Hub)
*   **介面**: 卡片式網格展示，即時渲染標記進度條，支援上傳與刪除功能。
*   **邏輯**: 遞迴掃描目錄，維持 YOLO 原生目錄結構 (`images/` 與 `labels/`)。上傳時自動建立並歸檔至 `data/datasets/[資料集名稱]/`。

### 3.2 影像標記任務 (Annotation Task)
*   **介面**: 高階標記畫布，支援滑鼠框選、微調、智慧跳轉未標記圖片。
*   **邏輯**: 
    *   標籤轉換：接收前端座標轉為 YOLO 格式寫入。
    *   防鎖定機制：實作 Windows/OneDrive 寫入重試邏輯，避免同步衝突。
    *   AI 輔助：支援透過自然語言下達「自動標記」指令，背景執行影像辨識。

### 3.3 模型訓練中心 (Training Center)
*   **介面**: 動態配置參數 (模型、輪次、批次等)，即時查看訓練進度日誌。
*   **邏輯**: 
    *   隔離執行緒：自動生成 `data.yaml` 並在獨立 threading 中執行，避免阻塞主執行緒。
    *   自動佈署：訓練完成後自動將 `best.pt` 複製至 `models/` 並重新命名。

### 3.4 驗證與診斷系統 (Verify & Diagnosis)
*   **介面**: 
    *   驗證區：支援單張/批量驗證，即時調整 Confidence、IoU、TTA 參數。
    *   診斷區：渲染混淆矩陣、P-R 曲線，提供健康分數儀表板。
*   **邏輯**: 讀取 `runs/` 內的訓練日誌與 CSV 報表，由 `ai_analyzer.py` 產出數據，交由 `ai_service.py` 產生「行動處方箋」(AI 分析建議)。

## 4. 檔案與目錄結構規劃
```text
AI-project/
├── run_multi_port.py             # 系統多埠口啟動器 (根目錄入口)
├── start_server.bat              # 一鍵啟動腳本
├── server/                       # 後端核心引擎
│   ├── main.py                   # API 網關與全域狀態管理
│   ├── inferencer.py             # YOLO 影像辨識與推理
│   ├── trainer.py                # YOLO 模型訓練調度
│   ├── ai_service.py             # LLM 大腦接口 (Gemini/Groq)
│   └── ai_analyzer.py            # 資料分析與健康度診斷
├── server/static/                # 前端靜態資源
│   ├── index_v3.html             # 主面板 (訓練、驗證、診斷)
│   ├── datasets.html             # 資料集管理介面
│   ├── labeling.html             # 標記工作檯介面
│   └── app.js, style.css         # 互動邏輯與設計樣式
├── models/                       # 已佈署之模型權重 (.pt)
├── runs/                         # 訓練日誌與生成之圖表
├── data/                         
│   ├── datasets/                 # 核心資料庫 (圖檔與標籤)
│   ├── uploads/                  # 網頁上傳暫存區
│   └── outputs/                  # 畫好標記框之推理結果圖
└── .env                          # API 金鑰與敏感設定檔
```

## 5. 待執行與優化項目 (Next Steps)
1. **驗證介面拆分**: 將驗證頁面明確拆分為「單張驗證」與「批量驗證」獨立側邊欄分頁。
2. **批量處理優化**: 批次圖片預覽加入分頁功能 (每頁 20 張)，提升前端效能。
3. **資料清理機制**: 實作「清空資料」功能，連動刪除 `uploads` 與 `outputs` 目錄下的實體暫存檔案。
4. **結果清單顯示**: 預覽圖下方即時顯示檢測結果清單 (包含類別、信心值、精確座標)。
