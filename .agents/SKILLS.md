# AI Detector — Skills 與 Tools 規範 (SKILLS.md)

本文件定義專案中對外暴露的 **Skills**。每個 Skill 封裝了多個底層的 **Tools（原子操作）**、決策邏輯與自動化 Gate，代表一個使用者會重複要求完成的完整工作。

---

## 1. SKILL: `train` (訓練模型)

- **所屬 Agent**: Model Agent（前置需 Data Agent 配合）
- **關聯 Jobs**: `JOB-1` (建立新模型), `JOB-2` (改善準確率), `JOB-4` (擴充資料集後重訓)
- **為什麼值得做成 Skill**: 訓練並非單純呼叫 `train` 指令。它整合了「資料未標記率檢查」、「資料品質健康度 Gate」、「超參數自適應調整」、「CUDA OOM 記憶體不足自動降級重試」等一連串使用者不想手動處理的邏輯。

### 內部包含的 Tools (底層原子操作)
1. `list_unlabeled`：掃描並列出未標記的圖片。
2. `analyze_health`：計算資料集健康評分與類別分佈。
3. `prepare_yaml`：自動生成 YOLO 訓練所必須的 `dataset.yaml` 結構定義。
4. `run_training`：調用底層 `ultralytics` 執行訓練並回傳進度。

### 執行與決策邏輯
1. **未標記檢查**：呼叫 `list_unlabeled`。若 `unlabeled_count / total_images > 0.2` (超過 20% 未標記)，中斷流程，回報使用者要求先進行標記。
2. **品質 Gate 驗證**：呼叫 `analyze_health`。若 `health_score < 70`，中斷流程並產出品質報告，引導使用者補充偏斜類別的資料。
3. **環境準備**：呼叫 `prepare_yaml` 生成 YAML 設定檔。
4. **訓練與自動 Retry (OOM 處理)**：
   - 執行 `run_training` (預設 `batch=16`)。
   - 若發生 `CUDA out of memory`：將 `batch` 減半（16 -> 8 -> 4），自動重試（最多 2 次）。
   - 其他嚴重錯誤：輸出 Log 最後 20 行，終止工作。
5. **輸出驗證**：確認 `runs/train/<EXP_NAME>/weights/best.pt` 存在且檔案大小大於 0。

### 呼叫指令 (CLI)
```powershell
python -c "
import sys
sys.path.insert(0, 'server')
from trainer import YOLOTrainer
from ai_analyzer import ai_analyzer
import json

ds_name = '<DATASET_NAME>'
epochs = <EPOCHS>
batch = <BATCH_SIZE>
exp_name = '<EXP_NAME>'

# 1. 執行內部 Tools 檢查
health = ai_analyzer.analyze_dataset_health(ds_name)
if health.get('score', 0) < 70:
    print(json.dumps({'status': 'FAILED', 'reason': 'Dataset health score below 70', 'details': health}, ensure_ascii=False))
    sys.exit(1)

# 2. 準備與執行
trainer = YOLOTrainer()
yaml_path = trainer.prepare_dataset('data/datasets/' + ds_name)
trainer.train(variant='yolov9m.pt', data_yaml=yaml_path, epochs=epochs, batch=batch, model_name=exp_name)
print(json.dumps({'status': 'SUCCESS', 'model_path': f'runs/train/{exp_name}/weights/best.pt'}, ensure_ascii=False))
"
```

---

## 2. SKILL: `infer` (辨識圖片)

- **所屬 Agent**: Model Agent
- **關聯 Jobs**: `JOB-3` (辨識指定圖片)
- **為什麼值得做成 Skill**: 封裝了「模型選擇邏輯」與「路徑無效的自動 Fallback 機制」，使用者只需傳入圖片，AI 自行決定最適合的模型。

### 內部包含的 Tools (底層原子操作)
1. `select_model`：根據專案狀態或使用者指定，挑選合適的模型路徑。
2. `run_inference`：調用底層載入模型，進行 YOLO 推論並產出標記後圖片。

### 執行與決策邏輯
1. **模型選擇**：
   - 若使用者明確指定 `model_path`，直接使用。
   - 若無指定，讀取 `workspace_state.json` 的 `model.best_model_path`。
   - 若皆無，Fallback 使用預設的 `models/yolov9m.pt`。
2. **圖片驗證**：若輸入圖片路徑不存在，直接終止回報錯誤。
3. **推理與儲存**：執行 `run_inference`，將辨識標記後的圖片輸出至 `data/outputs/`。

### 呼叫指令 (CLI)
```powershell
python -c "
import sys
sys.path.insert(0, 'server')
from inferencer import YOLOInferencer
import json

model_path = '<MODEL_PATH>'  # 依選擇邏輯帶入，例如 'models/yolov9m.pt'
image_path = '<IMAGE_PATH>'

try:
    inf = YOLOInferencer(model_path=model_path)
    results, annotated = inf.predict_and_save(image_path, conf=0.25)
    print(json.dumps({'status': 'SUCCESS', 'results': results, 'annotated_image': annotated}, ensure_ascii=False))
except Exception as e:
    print(json.dumps({'status': 'FAILED', 'reason': str(e)}))
"
```

---

## 3. SKILL: `eval` (評估模型效果)

- **所屬 Agent**: Model Agent
- **關聯 Jobs**: `JOB-2` (改善現有模型準確率)
- **為什麼值得做成 Skill**: 評估模型不只是回傳 mAP 等數字，而是將這些數字轉化為「可被 Agent 或使用者讀懂的具體改進建議（如：是否過擬合、哪些類別資料不夠）」。

### 內部包含的 Tools (底層原子操作)
1. `read_metrics`：讀取 YOLO 訓練輸出的 `results.csv`。
2. `summarize_metrics`：計算 mAP50、最終 Loss 等關鍵數字。

### 執行與決策邏輯
1. **讀取指標**：獲取最新一次訓練的 CSV 指標。
2. **問題診斷 (Gatekeeper)**：
   - **檢查過擬合**：若驗證集損失比前幾輪平均上升超過 5%，標註為 `OVERFITTING`。
   - **檢查標籤傾斜**：若有特定類別在 `results.csv` 中的 Precision/Recall 異常低，標註為 `DATA_SKEW`。
3. **產出下一步建議**：
   - 若為 `OVERFITTING`，輸出建議：「調降 Epochs 或增加訓練資料增強（Augmentation）」。
   - 若為 `DATA_SKEW`，輸出建議：「補充該特定類別的標記圖片」。

### 呼叫指令 (CLI)
```powershell
python -c "
import sys
sys.path.insert(0, 'server')
from ai_analyzer import AIAnalyzer
import json

exp_name = '<EXP_NAME>'
analyzer = AIAnalyzer()
df = analyzer.get_latest_results(exp_name)
summary = analyzer.summarize_metrics(df)
print(json.dumps(summary, ensure_ascii=False, indent=2))
"
```
