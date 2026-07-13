# AI Detector 更新紀錄 (Update Log)

## [2026-05-29] 更新 Gemini API 模型版本

### 修正問題
- **模型已停用**：Gemini 1.5 系列 (Flash & Pro) 已經移除且停止服務。

### 變更內容
- **[.env](file:///c:/Users/freddy_huang/OneDrive%20-%20ASUS/Desktop/AI-project/.env)**:
  - 將 `GEMINI_FLASH_MODEL` 更新為 `gemini-2.5-flash`
  - 將 `GEMINI_PRO_MODEL` 更新為 `gemini-2.5-pro`
- **[server/static/index_v3.html](file:///c:/Users/freddy_huang/OneDrive%20-%20ASUS/Desktop/AI-project/server/static/index_v3.html)**:
  - 將 AI 助手對話窗的 LLM 下拉選單 (`#ai-model-select`) 中的選項從 Gemini 1.5 系列同步更新為 Gemini 2.0。
- **[server/ai_service.py](file:///c:/Users/freddy_huang/OneDrive%20-%20ASUS/Desktop/AI-project/server/ai_service.py)**:
  - 修正後端直接寫死 `gemini-1.5-pro-latest` 以及無法正確讀取 `.env` 中 `GEMINI_FLASH_MODEL` 與 `GEMINI_PRO_MODEL` 變數的問題。

## [2026-05-21] 影像驗證中心修復

### 修正問題
- **推理參數錯誤**：修正單張/批量資料夾驗證時 YOLO 推理參數傳入無效的 `model_path` 導致的 `TypeError`。
- **解包與解析錯誤**：修正 `YOLOInferencer.predict` 返回的二元組被誤當成字典處理導致的 `AttributeError`。
- **影像存檔缺失**：修正推理標記後的影像未被存入實體硬碟、且前端加載無效 URL 的問題。

### 變更內容
- **[server/main.py](file:///c:/Users/freddy_huang/OneDrive%20-%20ASUS/Desktop/AI-project/server/main.py)**:
  - **`verify_files` 路由**：載入用戶選擇的模型權重，修正 `predict` 簽名，解包獲取 `detections` 與 `annotated_img`。使用 OpenCV 安全的 `imencode` 寫入機制，將圖像儲存於 `outputs/verify_{filename}` 中，回傳其訪問 URL `/outputs/verify_{filename}` 及其 `objects` 列表。
  - **`verify_folder` 路由**：同上，修正對應資料夾內各圖片的 YOLO 推理與影像存檔邏輯。
  - **`verify_single` 路由**：同步重構其 API 推理調用及安全存檔邏輯，以保證代碼的一致性與穩定性。

## [2026-05-21] 驗證中心前端 UI/UX 優化與資料夾上傳支援

### 修正問題
- **路徑輸入不便**：移除原先的伺服器路徑輸入（Prompt/Alert），改為瀏覽器原生的資料夾上傳，免除手動輸入伺服器檔案路徑的繁瑣步驟。
- **偵測詳情缺乏**：前端結果預覽未列出各圖片的偵測目標清單（類別名稱、置信度、四角坐標）。

### 變更內容
- **[server/static/index_v3.html](file:///c:/Users/freddy_huang/OneDrive%20-%20ASUS/Desktop/AI-project/server/static/index_v3.html)**:
  - 調整驗證中心控制列，加入「選擇影像上傳」與「選擇資料夾上傳」按鈕，並配置 `webkitdirectory` 與 `directory` 屬性，使用戶可直接選取整本資料夾。
  - 新增狀態提示文字 `#verify-status-text` 以即時顯示目前已選擇的圖片張數。
- **[server/static/app.js](file:///c:/Users/freddy_huang/OneDrive%20-%20ASUS/Desktop/AI-project/server/static/app.js)**:
  - 實作 `handleVerifyUpload` 邏輯，自動篩選選定資料夾中的圖片格式，並更新狀態提示。
  - 調整 `startUnifiedVerification` 將選擇的本地資料夾內圖片統一以 `FormData` 多檔案形式傳送至 `/api/verify/files` 端點推理。
  - 優化 `renderVerifyResults` 渲染卡片：在預覽圖下方即時渲染出每個物件的類別名稱、Confidence（百分比）以及精確的 `[x1, y1, x2, y2]` 坐標清單。

## [2026-05-21] 驗證中心子目錄上傳錯誤與服務更新重啟

### 修正問題
- **資料夾上傳子目錄寫入錯誤**：修正 `webkitdirectory` 上傳時 `file.filename` 帶有相對子路徑（如 `images/1.jpg`）導致儲存至 `temp_verify` 實體檔案時因找不到目錄報 `FileNotFoundError` (Errno 2) 的問題。
- **後端 YOLO 推理舊代碼殘留**：重啟多埠混合動力伺服器，完全解決記憶體內舊進程殘留呼叫已廢棄 `model_path` 參數之舊 YOLO 預測方法的問題。

### 變更內容
- **[server/main.py](file:///c:/Users/freddy_huang/OneDrive%20-%20ASUS/Desktop/AI-project/server/main.py)**:
  - 在 `verify_files` 中使用 `Path(file.filename).name` 將上傳的圖片檔名強制「扁平化」，去除任何相對路徑前綴，確保在 `temp_verify` 與 `outputs` 下寫入實體圖片均能百分之百成功。
- **伺服器重啟**：
  - 清理背景佔用 `8000 ~ 8003` 埠口之所有舊 python 進程，並重新執行 `run_multi_port.py`，載入全新的代碼實體。
