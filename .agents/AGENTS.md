# AI Detector — Agent 協作規範 (AGENTS.md)

此檔案為本專案所有 AI Agent（包含 Antigravity、Claude Code、GitHub Copilot Workspace、Codex CLI 等）的**主要操作規範與上下文說明**。所有 Agent 在協作前必須先讀取本文件。

---

## 1. 專案概述

本專案為一個基於 **YOLO** 的 AI 物件辨識平台，提供資料集管理、圖片標記、模型訓練、推理辨識與模型評估的完整功能。

- **技術棧**：Python 3.10+、FastAPI、Ultralytics YOLO、PyTorch
- **作業系統**：Windows（路徑分隔符請使用 `/` 或 `\\`）
- **Python 虛擬環境**：`venv/`（請在執行所有 python 指令前先啟動）
- **啟動虛擬環境指令**：`.\venv\Scripts\activate`（PowerShell）

---

## 2. 專案目錄結構

```
AI-project/
├── .agents/                    # AI Agent 協作定義（本目錄）
│   ├── AGENTS.md               # 本檔案：主要規範
│   ├── SKILLS.md               # 所有 Skill 的 CLI 執行規範
│   ├── WORKFLOW.md             # Multi-Agent 工作流程定義
│   └── rules/                  # 額外規則
├── data/
│   └── datasets/               # 資料集存放目錄（每個子資料夾即一個 Dataset）
├── models/                     # 預訓練 YOLO 權重（如 yolov9m.pt）
├── runs/
│   └── train/                  # 訓練輸出（子資料夾即實驗名稱，含 best.pt）
├── server/
│   ├── main.py                 # FastAPI 後端入口
│   ├── trainer.py              # YOLOTrainer 模型訓練類別
│   ├── inferencer.py           # YOLOInferencer 推理類別
│   └── ai_analyzer.py          # AIAnalyzer 健康度與評估分析類別
├── workspace_state.json        # 專案目前狀態（必讀，用於判斷跳過哪些步驟）
└── requirements.txt
```

---

## 3. 核心規則

1. **讀取狀態優先**：執行任何任務前，必須先讀取 `workspace_state.json` 以了解目前的專案進度，避免重複執行已完成的步驟。
2. **修改後更新狀態**：每個 Skill 執行完畢後，必須更新 `workspace_state.json` 中對應的欄位。
3. **不得修改核心業務邏輯**：除非使用者明確要求，否則不得修改 `server/` 目錄底下的現有 Python 邏輯。Agent 的職責是**呼叫**這些功能，而非重寫它們。
4. **錯誤回報格式**：遇到錯誤時，必須按照以下格式回報：
   ```
   [ERROR]
   Status: FAILED
   Root Cause: <具體原因>
   Suggested Fix: <建議修正步驟>
   ```
5. **虛擬環境必須啟動**：所有 `python` 指令執行前，確認虛擬環境已啟動（`venv/`）。
6. **語言規範**：對使用者的輸出使用繁體中文；代碼注釋與技術日誌使用英文。

---

## 4. Skills 索引

詳細規範請參閱 `.agents/SKILLS.md`。

| Skill 名稱 | 功能描述 | 所屬 Agent |
| :--- | :--- | :--- |
| `dataset_import` | 掃描並匯入資料集目錄，建立 Dataset 結構 | Data Agent |
| `dataset_health` | 分析資料集健康度，產出評分與報告 | Data Agent |
| `annotation_check` | 檢查未標記圖片清單，驗證標籤格式 | Data Agent |
| `model_train` | 執行 YOLO 模型訓練 | Model Agent |
| `model_infer` | 對指定圖片執行推理辨識 | Model Agent |
| `model_eval` | 評估訓練完成的模型，讀取 mAP 等指標 | Model Agent |

---

## 5. Workflow 索引

詳細規範請參閱 `.agents/WORKFLOW.md`。

| 情境 | 觸發條件 | 工作流 |
| :--- | :--- | :--- |
| 建立新模型 | 無 Dataset 或指定建立全新模型 | Import → Health → Train → Eval |
| 重新訓練模型 | Dataset 已存在 | Train → Eval |
| 快速推理辨識 | 已有模型，使用者提供圖片 | Infer |
| 模型效果不佳 | 使用者反映準確率低 | Eval → 診斷 → Annotate 或 Train |
