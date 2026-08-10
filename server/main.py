from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
import shutil
import os
import json
import sys
import threading
import uuid
import platform
import psutil
import torch
from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel
from inferencer import YOLOInferencer
from ai_service import ai_engine
from optimization_api import router as optimization_router
from ai_analyzer import ai_analyzer

from fastapi.middleware.cors import CORSMiddleware

# 自定義靜態檔案類別，強制不快取
class NoCacheStaticFiles(StaticFiles):
    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False
    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

app = FastAPI()

# 允許跨 Port 通訊 (8000 ~ 8005)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 註冊新模組
app.include_router(optimization_router)

# 取得絕對路徑
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "server" / "static"
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
DATASET_DIR = DATA_DIR / "datasets"

# 確保目錄存在
for d in [UPLOAD_DIR, OUTPUT_DIR, MODEL_DIR, DATASET_DIR]:
    d.mkdir(parents=True, exist_ok=True)

inferencer = YOLOInferencer()
# 跟 agent_tools.train_yolo_model 共用同一個 YOLOTrainer 實例（見 agent_tools.py
# 的 shared_trainer 說明）——不能各自 new 一個，否則網頁表單按「終止」設的
# stop_event，跟 Agent 真正在跑訓練用的那個 trainer 是兩個不同物件，永遠打不到。
from agent_tools import shared_trainer as trainer

# 全域自動標記狀態
auto_label_status = {
    "status": "idle",
    "dataset": "",
    "processed": 0,
    "total": 0,
    "labeled_count": 0,
    "error": ""
}

# ------------------------------
# 資料集 API (務必在 mount 之前)
# ------------------------------
@app.post("/api/datasets/upload_v2")
async def upload_dataset(name: str = Form(...), file: UploadFile = File(...)):
    try:
        ds_path = DATASET_DIR / name
        ds_path.mkdir(parents=True, exist_ok=True)
        
        # 取得檔案路徑 (包含可能的子目錄)
        file_path = ds_path / file.filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 串流存檔
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        await file.close()
        return {"status": "success", "message": f"Saved {file.filename}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/datasets/list")
async def list_datasets():
    try:
        datasets = []
        if DATASET_DIR.exists():
            for d in DATASET_DIR.iterdir():
                if d.is_dir():
                    # 使用 rglob 進行遞迴搜尋，確保能抓到子資料夾裡的圖片
                    images = []
                    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
                        images.extend(list(d.rglob(ext)))
                    labels = list(d.rglob("*.txt"))
                    # 排除 classes.txt 等非標記檔案，且必須檔案大小 > 0 才算真正標記過
                    valid_labels = []
                    for l in labels:
                        if l.name.lower() not in ['classes.txt', 'notes.txt', 'obj.names', '_classes.txt']:
                            try:
                                if l.stat().st_size > 0:
                                    valid_labels.append(l)
                            except:
                                pass
                    
                    datasets.append({
                        "id": str(d.stat().st_mtime),
                        "name": d.name,
                        "path": str(d.absolute()),
                        "imgCount": len(images),
                        "labelCount": len(valid_labels)
                    })
        return {"status": "success", "datasets": datasets}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/datasets/health")
async def get_dataset_health(name: str):
    try:
        return ai_analyzer.analyze_dataset_health(name)
    except Exception as e:
        return {"status": "error", "message": str(e)}



@app.delete("/api/datasets/{name}")
async def delete_dataset(name: str):
    try:
        ds_path = DATASET_DIR / name
        if ds_path.exists() and ds_path.is_dir():
            import shutil
            import os
            import stat

            def remove_readonly(func, path, _):
                "Clear the readonly bit and reattempt the removal"
                os.chmod(path, stat.S_IWRITE)
                func(path)

            shutil.rmtree(ds_path, onerror=remove_readonly)
            return {"status": "success", "message": f"Dataset {name} deleted"}
        else:
            return {"status": "error", "message": "Directory not found"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/datasets/files")
async def get_dataset_files(name: str):
    try:
        ds_path = DATASET_DIR / name
        if not ds_path.exists():
            return {"status": "error", "message": "Dataset not found"}
        
        # 遞迴搜尋所有圖片並檢查標記狀態
        image_files = []
        for ext in ["**/*.jpg", "**/*.jpeg", "**/*.png", "**/*.bmp", "**/*.webp"]:
            image_files.extend(list(ds_path.glob(ext)))
        
        files_info = []
        labeled_count = 0
        for img_path in sorted(image_files):
            rel_img = str(img_path.relative_to(ds_path)).replace('\\', '/')
            
            # Python 版邏輯：取得不含副檔名的路徑
            base_name = os.path.splitext(rel_img)[0]
            label_rel = base_name + '.txt'
            
            if 'images/' in rel_img:
                label_rel = rel_img.replace('images/', 'labels/')
                label_rel = os.path.splitext(label_rel)[0] + '.txt'
            
            label_abs = ds_path / label_rel
            is_labeled = label_abs.exists() and label_abs.stat().st_size > 0
            
            if is_labeled: labeled_count += 1
            files_info.append({
                "name": rel_img,
                "labeled": is_labeled
            })
            
        # 遞迴搜尋標籤定義檔
        class_files = ["classes.txt", "_classes.txt", "obj.names", "data.yaml"]
        found_class_file = None
        class_names = []
        for cf in class_files:
            matches = list(ds_path.glob(f"**/{cf}"))
            if matches:
                found_class_file = str(matches[0].relative_to(ds_path)).replace('\\', '/')
                try:
                    if cf.endswith(".yaml"):
                        import yaml
                        with open(matches[0], 'r', encoding='utf-8') as f:
                            yd = yaml.safe_load(f)
                            if 'names' in yd:
                                if isinstance(yd['names'], dict):
                                    class_names = [yd['names'][k] for k in sorted(yd['names'].keys())]
                                else:
                                    class_names = yd['names']
                    else:
                        with open(matches[0], 'r', encoding='utf-8') as f:
                            class_names = [l.strip() for l in f.readlines() if l.strip()]
                except:
                    pass
                break
            
        return {
            "status": "success", 
            "files": files_info, 
            "labeled_count": labeled_count,
            "total_count": len(files_info),
            "has_classes": bool(class_names), 
            "classes": class_names, 
            "class_file": found_class_file
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/models")
async def list_models():
    models = [f.name for f in MODEL_DIR.glob("*.pt")]
    return {"models": models}

@app.post("/api/models/delete")
async def delete_model(model_name: str = Form(...)):
    """刪除訓練產出的模型檔。只允許刪 *_best.pt（訓練產物），預訓練權重不給刪；
    路徑鎖死在 models/ 資料夾內，防止用 ../ 之類的名稱越權刪檔。"""
    try:
        target = (MODEL_DIR / model_name).resolve()
        if target.parent != MODEL_DIR.resolve() or target.suffix != ".pt":
            return {"status": "error", "message": "不合法的模型名稱"}
        if not model_name.endswith("_best.pt"):
            return {"status": "error", "message": "只能刪除訓練產出的模型（*_best.pt），預訓練權重不可刪除"}
        if not target.exists():
            return {"status": "error", "message": "找不到這個模型檔"}
        target.unlink()
        return {"status": "success", "message": f"已刪除模型 {model_name}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

from fastapi.responses import FileResponse

@app.get("/api/model/evaluation/charts/{chart_name}")
async def get_evaluation_chart(chart_name: str, model: str = ""):
    from fastapi.responses import JSONResponse
    # 這裡的 model 參數是前端傳過來的訓練名稱 (例如: 花-訓練)
    if not model:
        model = "exp"
    
    chart_path = BASE_DIR / "runs" / "train" / model / chart_name
    
    if chart_path.exists() and chart_path.is_file():
        return FileResponse(str(chart_path))
    else:
        return JSONResponse(status_code=404, content={"status": "error", "message": f"Chart {chart_name} not found in {model}"})

@app.post("/api/verify/files")
async def verify_files(files: list[UploadFile] = File(...), model: str = Form(...), imgsz: int = Form(640), conf: float = Form(0.25), iou: float = Form(0.45), augment: bool = Form(False)):
    try:
        from inferencer import YOLOInferencer
        import shutil
        import cv2
        temp_dir = Path("temp_verify")
        temp_dir.mkdir(exist_ok=True)
        results = []
        inf = YOLOInferencer(model_path=str(MODEL_DIR / model))
        for file in files:
            filename_flat = Path(file.filename).name
            file_path = temp_dir / filename_flat
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            detections, annotated_img = inf.predict(
                str(file_path),
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                augment=augment
            )
            
            # Save the annotated image to outputs/ directory
            ext = os.path.splitext(filename_flat)[1] or ".jpg"
            output_file = OUTPUT_DIR / f"verify_{filename_flat}"
            _, img_encoded = cv2.imencode(ext, annotated_img)
            with open(output_file, "wb") as f:
                f.write(img_encoded.tobytes())
                
            results.append({
                "name": filename_flat,
                "url": f"/outputs/verify_{filename_flat}",
                "objects": detections
            })
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/verify/folder")
async def verify_folder(request: Request):
    try:
        data = await request.json()
        folder_path = Path(data.get("path"))
        model = data.get("model")
        imgsz = data.get("imgsz", 640)
        conf = data.get("conf", 0.25)
        iou = data.get("iou", 0.45)
        augment = data.get("augment", False)
        if not folder_path.exists():
            return {"status": "error", "message": "路徑不存在"}
        from inferencer import YOLOInferencer
        import cv2
        inf = YOLOInferencer(model_path=str(MODEL_DIR / model))
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        results = []
        for img_p in folder_path.rglob("*"):
            if img_p.suffix.lower() in image_extensions:
                detections, annotated_img = inf.predict(
                    str(img_p),
                    imgsz=imgsz,
                    conf=conf,
                    iou=iou,
                    augment=augment
                )
                
                # Save the annotated image to outputs/ directory
                output_file = OUTPUT_DIR / f"verify_{img_p.name}"
                _, img_encoded = cv2.imencode(img_p.suffix or ".jpg", annotated_img)
                with open(output_file, "wb") as f:
                    f.write(img_encoded.tobytes())
                    
                results.append({
                    "name": img_p.name,
                    "url": f"/outputs/verify_{img_p.name}",
                    "objects": detections
                })
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/datasets/image/{ds_name}/{img_name:path}")
async def get_dataset_image(ds_name: str, img_name: str):
    file_path = DATASET_DIR / ds_name / img_name
    if file_path.exists():
        return FileResponse(file_path)
    raise HTTPException(status_code=404)

@app.get("/api/datasets/label/{ds_name}/{label_path:path}")
async def get_dataset_label(ds_name: str, label_path: str):
    try:
        # label_path 可能是 train/labels/1.txt
        file_path = DATASET_DIR / ds_name / label_path
        if not file_path.exists():
            return {"status": "error", "message": "Label file not found"}
        
        with file_path.open("r", encoding="utf-8") as f:
            content = f.read()
        return {"status": "success", "content": content}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/datasets/save_classes")
async def save_dataset_classes(ds_name: str = Form(...), classes: str = Form(...)):
    try:
        class_list = json.loads(classes)
        ds_path = DATASET_DIR / ds_name
        class_file = ds_path / "classes.txt"
        
        with open(class_file, "w", encoding="utf-8") as f:
            f.write("\n".join(class_list))
            
        return {"status": "success", "message": "Classes saved"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/datasets/save_label")
async def save_dataset_label(ds_name: str = Form(...), label_path: str = Form(...), content: str = Form("")):
    try:
        file_path = DATASET_DIR / ds_name / label_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 增加重試機制 (針對 OneDrive 忙碌鎖定)
        for attempt in range(3):
            try:
                with file_path.open("w", encoding="utf-8") as f:
                    f.write(content)
                return {"status": "success", "message": "Label saved"}
            except PermissionError:
                if attempt < 2:
                    import time
                    time.sleep(0.5)
                    continue
                raise
        return {"status": "error", "message": "File busy"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/datasets/delete")
async def delete_dataset_post(name: str = Form(...)):
    try:
        ds_path = DATASET_DIR / name
        if ds_path.exists():
            shutil.rmtree(ds_path)
        return {"status": "success"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ------------------------------
# 其他 API 保持不變
# ------------------------------
@app.get("/api/system/info")
async def get_system_info():
    info = {"cpu": "未知", "ram": "-", "gpu": "無", "cuda": False}
    try:
        info["cpu"] = platform.processor() or platform.machine()
        ram = psutil.virtual_memory()
        info["ram"] = f"{round(ram.total / (1024**3), 1)} GB"
        info["cuda"] = torch.cuda.is_available()
        if info["cuda"]:
            info["gpu"] = torch.cuda.get_device_name(0)
        return info
    except:
        return info

@app.post("/api/verify/single")
async def verify_single(
    model_name: str = Form(...),
    imgsz: int = Form(...),
    conf: float = Form(...),
    iou: float = Form(...),
    augment: bool = Form(...),
    file: UploadFile = File(...)
):
    try:
        import cv2
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        save_path = UPLOAD_DIR / f"input{ext}"
        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 執行推理
        inferencer.switch_model(str(MODEL_DIR / model_name))
        detections, annotated_img = inferencer.predict(
            str(save_path),
            imgsz=imgsz,
            conf=conf,
            iou=iou,
            augment=augment
        )
        
        # Save output image
        output_file = OUTPUT_DIR / f"verify_{file.filename}"
        _, img_encoded = cv2.imencode(ext, annotated_img)
        with open(output_file, "wb") as f:
            f.write(img_encoded.tobytes())
            
        return {"status": "success", "image_url": f"/outputs/verify_{file.filename}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/api/clear")
async def clear_cache():
    for d in [UPLOAD_DIR, OUTPUT_DIR]:
        for f in d.iterdir():
            if f.is_file(): f.unlink()
    return {"status": "success"}

# ------------------------------
# 訓練 API
# ------------------------------
@app.post("/api/train/start")
async def start_training(
    dataset_name: str = Form(...),
    model_name: str = Form("exp"),
    variant: str = Form("yolov9m.pt"),
    epochs: int = Form(50),
    batch: int = Form(16),
    imgsz: int = Form(640),
    patience: int = Form(50)
):
    try:
        ds_path = DATASET_DIR / dataset_name
        if not ds_path.exists():
            return {"status": "error", "message": f"資料集 {dataset_name} 不存在"}

        if trainer.progress.get("status") == "training":
            return {"status": "error", "message": "已有訓練正在進行中"}

        # 準備 data.yaml
        data_yaml = trainer.prepare_dataset(str(ds_path))

        def run():
            trainer.train(
                variant=variant,
                data_yaml=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                patience=patience,
                model_name=model_name
            )
            # 訓練完成後，使用絕對路徑自動複製 best.pt 到 models/
            project_root = Path(__file__).resolve().parent.parent
            best_pt = project_root / "runs" / "train" / model_name / "weights" / "best.pt"
            
            if best_pt.exists():
                # 使用輸入的名稱作為檔名
                dest = MODEL_DIR / f"{model_name}_best.pt"
                shutil.copy2(str(best_pt), str(dest))
                print(f"Model deployed to {dest}")
            else:
                print(f"Warning: best.pt not found at {best_pt}")

        t = threading.Thread(target=run, daemon=True)
        t.start()

        return {"status": "success", "message": "訓練已啟動"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/train/status")
async def get_train_status():
    return trainer.progress

@app.post("/api/train/stop")
async def stop_training():
    trainer.stop()
    return {"status": "success"}

# ------------------------------
# AI 服務 API
# ------------------------------
class ChatRequest(BaseModel):
    messages: list
    context_data: dict = None
    model_type: str = "gemini"

@app.post("/api/ai/chat")
async def ai_chat(request: ChatRequest):
    try:
        # 自動收集精簡的系統狀態
        datasets_info = ai_analyzer.get_all_datasets_info()
        summarized_datasets = [d['name'] for d in datasets_info] # 只傳名稱，不傳細節
        
        auto_context = {
            "current_training_status": trainer.progress,
            "available_datasets": summarized_datasets
        }
        
        final_context = request.context_data or {}
        final_context.update(auto_context)
        # 訓練授權只認前端在本次頁面送來的原始選取值。後面健康度分析可能為了回答
        # 問題而更新 final_context["activeDataset"]，不能讓那個補充資料變成開訓依據。
        selected_dataset_for_training = final_context.get("activeDataset", "")
        
        print(f">>> [AI Chat Request] Platform: {request.model_type}")
        user_msg = request.messages[-1]["content"] if request.messages else ""
        # --- 意圖偵測：健康度 vs 模型診斷（可共存） ---
        # 健康度觸發：訊息中出現「健康」「品質」「體檢」等資料集相關字眼
        health_keywords = ["健康", "品質", "體檢", "覆蓋率", "標註", "資料集", "dataset", "狀況", "狀態"]
        is_health_request = any(k in user_msg for k in health_keywords)

        # 使用者訊息中明確點名的資料集（長名優先，防止「測」搶先誤匹配「測試」）。
        # 單字元的資料集名稱（如「2」「花」）太容易撞到一般文字，必須連著「資料集」
        # 一起出現才算點名；比對前先拿掉括號、引號等修飾符號，讓「[2]資料集」
        # 「「花」資料集」這類寫法也能配到。
        import re as _re
        stripped_msg = _re.sub(r"[\[\]（）()「」『』【】<>\"'`＊*\s]", "", user_msg)
        mentioned_ds = None
        for name in sorted(summarized_datasets, key=len, reverse=True):
            if not name:
                continue
            if len(name) >= 2 and name.lower() in user_msg.lower():
                mentioned_ds = name
                break
            if len(name) == 1 and f"{name}資料集" in stripped_msg:
                mentioned_ds = name
                break

        # 有點名資料集就視為在問它的狀態（這個旗標之後還要用來決定模型診斷的優先順序）
        if mentioned_ds:
            is_health_request = True
        
        # 模型診斷觸發：訊息中出現「模型」「訓練」「效能」等字眼，但排除純健康度請求
        model_exclusive_keywords = ["模型", "訓練", "效能", "表現", "mAP", "Loss"]
        model_general_keywords = ["診斷", "分析", "結果", "成果"]
        # 只有當訊息包含「模型專屬詞」，或包含「通用詞但不包含健康詞」時，才觸發模型分析
        has_model_specific = any(k in user_msg for k in model_exclusive_keywords)
        has_general = any(k in user_msg for k in model_general_keywords)
        is_model_request = has_model_specific or (has_general and not is_health_request)
        # 訓練請求不需要先評估舊模型；兩種意圖都含有「模型／訓練」字眼，必須先區分。
        train_keywords = ["訓練模型", "訓練一個", "開始訓練", "幫我訓練", "重新訓練", "train a model"]
        is_train_request = any(kw in user_msg for kw in train_keywords)

        
        # 只要判斷得出使用者在講哪個資料集（訊息點名的優先，否則用前端選取的），就一律
        # 注入真實健康數據——不再用關鍵字當閘門。原本要命中「健康／品質／體檢」才注入，
        # 使用者換個說法（例如「現在的資料集狀況如何」）就注入不到，模型手上沒有任何
        # 數字，只好整段編造圖片數／類別數／標註率，是最容易被誤信的失效模式。掃一次
        # 資料夾的成本遠低於一次幻覺。
        ds_name = mentioned_ds or final_context.get("activeDataset", "")
        if ds_name and ds_name != "None":
            print(f">>> [Health] 正在分析資料集: {ds_name}")
            health_result = ai_analyzer.analyze_dataset_health(ds_name)
            if health_result.get("status") == "success":
                final_context["health_data"] = health_result
                # 讓系統提示詞的「當前活動資料集」與診斷摘要標頭顯示正確名稱，
                # 而不是前端還沒選資料集時的 "None"
                final_context["activeDataset"] = ds_name
                print(f">>> [Health] 分析成功: score={health_result['score']}")
                    
        # --- [模型效能診斷] 根據雙軌制判斷結果注入數據 ---
        missing_model_record = None
        auto_selected_model = None
        no_train_record_at_all = False
        if is_model_request and not is_train_request:
            try:
                import pandas as pd
                runs_dir = BASE_DIR / "runs" / "train"
                if runs_dir.exists():
                    # 直接比對 runs/train 底下的實驗名稱（長名優先，避免「1」搶先誤配到
                    # 「訓練-1」）。以前只認方括號寫法「[花-訓練]」，使用者直接打
                    # 「評估模型 花-訓練」就配不到，程式會默默改用「最新的一次訓練」，
                    # 於是回答的其實是別的模型的數據。stripped_msg 已去掉括號與空白，
                    # 所以加不加括號都能配到。
                    # 比對範圍同時包含 runs/train 的實驗名稱與 models/ 的模型檔名——
                    # 使用者講的可能是模型庫裡但沒有訓練紀錄的模型（例如預訓練權重），
                    # 那種情況要明確回報「沒有訓練紀錄」，不能悄悄改用別人的數據。
                    candidates = {d.name for d in runs_dir.iterdir() if d.is_dir()}
                    for f in MODEL_DIR.glob("*.pt"):
                        stem = f.stem
                        candidates.add(stem[:-5] if stem.endswith("_best") else stem)

                    # 「評估剛剛訓練完的模型」這種時間指涉沒有模型名稱可比對，直接視為
                    # 要用最近一次訓練；先判斷這個意圖，才不會被句子裡的字碎片誤配到
                    # 某個舊實驗名稱。
                    latest_ref_keywords = ["剛剛", "剛才", "剛訓練", "剛練", "剛完成",
                                           "最新", "最近", "上一次", "上次", "這次", "新訓練"]
                    wants_latest = any(k in user_msg for k in latest_ref_keywords)

                    mentioned_model = None
                    if not wants_latest:
                        for name in sorted(candidates, key=len, reverse=True):
                            # 單字元名稱（如「1」）太容易撞到句子裡的數字，只認完整檔名寫法
                            if len(name) >= 2 and name in stripped_msg:
                                mentioned_model = name
                                break
                            if f"{name}_best.pt" in stripped_msg:
                                mentioned_model = name
                                break

                    target_exp = None
                    if mentioned_model:
                        possible_dir = runs_dir / mentioned_model
                        if (possible_dir / "results.csv").exists():
                            target_exp = possible_dir
                        else:
                            # 有明確點名卻沒有訓練紀錄：要講清楚，不能悄悄換成別的模型
                            missing_model_record = mentioned_model
                    else:
                        # 沒點名模型（或講的是「剛剛訓練完的」）就取最近一次有紀錄的訓練。
                        # 用 results.csv 的修改時間排序：訓練過程中每個 epoch 都會寫入這個
                        # 檔案，比資料夾的 mtime 更貼近「最後一次真的在訓練」的時間。
                        exps = [d for d in runs_dir.iterdir() if d.is_dir() and (d / "results.csv").exists()]
                        if exps:
                            target_exp = max(exps, key=lambda d: (d / "results.csv").stat().st_mtime)
                            auto_selected_model = target_exp.name
                        else:
                            no_train_record_at_all = True

                    if target_exp:
                        csv_path = target_exp / "results.csv"
                        if csv_path.exists():
                            df = pd.read_csv(csv_path)
                            df.columns = df.columns.str.strip()
                            last_row = df.iloc[-1]
                            final_context["model_eval_data"] = {
                                "model_name": target_exp.name,
                                "epoch": int(last_row.get("epoch", 0)),
                                "train_loss": float(last_row.get("train/box_loss", 0) + last_row.get("train/cls_loss", 0)),
                                "val_loss": float(last_row.get("val/box_loss", 0) + last_row.get("val/cls_loss", 0)),
                                "map50": float(last_row.get("metrics/mAP50(B)", 0)),
                                "map50_95": float(last_row.get("metrics/mAP50-95(B)", 0))
                            }
                            print(f">>> [ModelEval] 擷取模型數據成功: name={target_exp.name}, mAP={final_context['model_eval_data']['map50']}")
            except Exception as e:
                print(f">>> [ModelEval] 擷取訓練數據失敗: {e}")

        
        # --- [自動標記] 本地指令解析 ---
        local_trigger_ds = None
        
        label_keywords = ["自動標記", "執行自動標註", "開始自動標記", "auto label", "啟動自動辨識"]
        is_label_request = any(kw in user_msg for kw in label_keywords)
        
        # 偵測是否明確要求使用自訓模型
        custom_model_keywords = ["訓練好的模型", "自訓模型", "_best.pt", "best.pt", "已訓練", "訓練完成的"]
        use_custom_model = any(kw in user_msg for kw in custom_model_keywords)
        
        if is_label_request:
            datasets_info = ai_analyzer.get_all_datasets_info()
            ds_names = [d['name'] for d in datasets_info]
            
            # [關鍵修正] 依據名稱長度由長到短排序，防止短名字（如「測」）搶先誤匹配長名字（如「測試」）
            ds_names_sorted = sorted(ds_names, key=len, reverse=True)
            
            # 嚴格比對：使用者訊息中是否明確包含資料集名稱
            for ds_name in ds_names_sorted:
                if ds_name.lower() in user_msg.lower() or ds_name in user_msg:
                    local_trigger_ds = ds_name
                    break
            
            # [修正] 不再自動 fallback 到第一個資料集，改為回報錯誤
            if not local_trigger_ds:
                print(f">>> [Auto Label] 找不到匹配的資料集名稱，可用列表: {ds_names}")

        
        # 呼叫 AI（獨立 try/except，確保 429 不會中斷觸發流程）
        try:
            reply = ai_engine.chat(request.messages, final_context, request.model_type)
        except Exception as chat_err:
            reply = f"[AI 暫時不可用]: {str(chat_err)}"

        # 健康圖卡的數據一律由後端附上：以前是叫 LLM 自己把 health JSON 抄一遍進
        # [HEALTH_DATA: ...]，它抄錯格式（甚至憑空編數字）就會讓整段標籤裸露在畫面上、
        # 或畫出假圖卡。這裡先清掉 LLM 自己寫的標籤，再貼上後端的真實數據。
        reply = _re.sub(r"\[HEALTH_DATA:.*?\]", "", reply, flags=_re.S).rstrip()

        # 使用者點名了某個模型但它沒有訓練紀錄：明講原因，不要讓使用者以為是 AI 壞了
        if missing_model_record:
            reply += (f"\n\n⚙️ [系統提示]：找不到模型「{missing_model_record}」的訓練紀錄"
                      f"（runs/train/{missing_model_record}/results.csv 不存在），因此無法提供效能數據。")

        # 自動挑了「最近一次訓練」時要說是哪一個，使用者才不用猜評估的是哪個模型
        if auto_selected_model:
            reply += f"\n\n⚙️ [系統提示]：已自動選用最近一次訓練「{auto_selected_model}」的紀錄進行評估。"
        elif no_train_record_at_all:
            reply += "\n\n⚙️ [系統提示]：目前 runs/train 底下沒有任何含 results.csv 的訓練紀錄，無法評估。"

        if is_health_request and final_context.get("health_data"):
            health_tag = json.dumps(final_context["health_data"], ensure_ascii=False)
            reply += f"\n\n[HEALTH_DATA: {health_tag}]"
        
        # --- 自動標記觸發邏輯 (雙重驗證) ---
        trigger_ds = None
        
        # 只有在使用者訊息確實包含標記意圖時，才解析 AI 的標籤
        if is_label_request:
            # 方法 1：從 AI 回覆中解析
            if "[AUTO_LABEL:" in reply:
                import re
                match = re.search(r"\[AUTO_LABEL:\s*(.*?)\]", reply)
                if match:
                    trigger_ds = match.group(1).strip(" '\"`*.")
            
            # 方法 2：本地解析（AI 未包含標籤但 user 有要求時）
            if not trigger_ds and local_trigger_ds:
                trigger_ds = local_trigger_ds
                reply = f"系統已啟動自動標記，正在為「{trigger_ds}」進行標記。"
            
            # [修正] 找不到資料集時，回傳明確錯誤，不靜默執行
            if not trigger_ds and not local_trigger_ds:
                reply = f"找不到你指定的資料集，目前可用的資料集有：{', '.join([d['name'] for d in ai_analyzer.get_all_datasets_info()])}，請再確認名稱後重試。"
        
        if trigger_ds:
            ds_path = DATASET_DIR / trigger_ds
            if ds_path.exists():
                ds_classes = ai_analyzer.get_dataset_classes(trigger_ds)
                
                def run_auto_label():
                    global auto_label_status
                    auto_label_status.update({
                        "status": "running",
                        "dataset": trigger_ds,
                        "processed": 0,
                        "total": 0,
                        "labeled_count": 0,
                        "error": ""
                    })
                    
                    def progress_cb(proc, tot, lab):
                        auto_label_status["processed"] = proc
                        auto_label_status["total"] = tot
                        auto_label_status["labeled_count"] = lab

                    try:
                        # [修正] 只有使用者明確要求時，才切換成自訓模型
                        if use_custom_model:
                            models_dir = BASE_DIR / "models"
                            custom_models = list(models_dir.glob("*_best.pt"))
                            if custom_models:
                                best_model = str(max(custom_models, key=lambda x: x.stat().st_mtime))
                                print(f"--- [Auto Label] 使用者指定自訓模型: {best_model} ---")
                                inferencer.switch_model(best_model)
                            else:
                                print(f"--- [Auto Label] 找不到自訓模型，回退使用預設 YOLOv9 模型 ---")
                        else:
                            print(f"--- [Auto Label] 使用預設 YOLOv9 模型（如需自訓模型請明確說明）---")

                        
                        count = inferencer.auto_label(str(ds_path), target_classes=ds_classes, progress_callback=progress_cb)

                        auto_label_status["status"] = "finished"
                        print(f"--- [Auto Label Success] {count} images labeled in {trigger_ds} ---")
                    except Exception as e:
                        auto_label_status["status"] = "error"
                        auto_label_status["error"] = str(e)
                        print(f"--- [Auto Label Failed] {e} ---")

                threading.Thread(target=run_auto_label, daemon=True).start()
                reply += f"\n\n⚙️ **[系統提示]**：已啟動自動標記引擎，正在為「{trigger_ds}」進行標記..."

        # --- [直接訓練] 舊式關鍵字觸發 ---
        # 這是「沒有 LangGraph」的行為：訊息中出現訓練意圖＋資料集名稱就直接開始
        # 訓練——不詢問參數、不跳確認視窗、無視使用者指定的參數，一律吃寫死的預
        # 設值，練完還會自動把模型塞進模型庫。跟 Manager Agent 的「提案參數→使用
        # 者確認→參數上鎖→詢問是否存入模型庫」形成對照。
        if is_train_request and not is_label_request:
            # 一般模式只能使用使用者在本次專案頁面明確點選的資料集。
            # 不採用訊息中猜到的名稱，也不使用瀏覽器上一次留下的選取值，避免誤訓練。
            train_ds = None
            ctx_ds = selected_dataset_for_training
            if ctx_ds and ctx_ds != "None" and ctx_ds in summarized_datasets:
                train_ds = ctx_ds

            if not train_ds:
                reply += ("\n\n⚙️ [系統提示]：偵測到訓練指令，但目前尚未選擇資料集，未啟動訓練。"
                          "請先到「資料集管理」點選要使用的資料集，再重新送出訓練要求。")
            elif trainer.progress.get("status") == "training":
                reply += "\n\n⚙️ [系統提示]：已有訓練正在進行中，未重複啟動。"
            else:
                try:
                    data_yaml = trainer.prepare_dataset(str(DATASET_DIR / train_ds))

                    # 實驗名稱帶上資料集與時間，避免每次聊天觸發的訓練都覆蓋掉同一份
                    # runs/train 紀錄與同一個模型檔（原本固定叫 chat_direct，第二次訓練
                    # 就會把第一次的成果蓋掉，事後也分不出是哪個資料集練的）。
                    from datetime import datetime
                    safe_ds = "".join(c for c in train_ds if c.isalnum() or c in "-_") or "ds"
                    exp_name = f"chat_{safe_ds}_{datetime.now().strftime('%m%d_%H%M')}"
                    deployed_name = f"{exp_name}_best.pt"

                    def run_direct_train():
                        trainer.train(
                            variant="yolov9s.pt",
                            data_yaml=data_yaml,
                            epochs=20,
                            imgsz=320,
                            batch=4,
                            patience=50,
                            model_name=exp_name
                        )
                        # 跟 /api/train/start 一樣，練完直接部署到 models/——不問使用者
                        project_root = Path(__file__).resolve().parent.parent
                        best_pt = project_root / "runs" / "train" / exp_name / "weights" / "best.pt"
                        if best_pt.exists():
                            shutil.copy2(str(best_pt), str(MODEL_DIR / deployed_name))
                            print(f"Model deployed to models/{deployed_name}")
                        else:
                            print(f"Warning: best.pt not found at {best_pt}")

                    threading.Thread(target=run_direct_train, daemon=True).start()
                    reply += (f"\n\n⚙️ [系統提示]：已直接啟動模型訓練（資料集：{train_ds}；"
                              "參數採系統預設 yolov9s.pt / epochs 20 / imgsz 320 / batch 4，未經確認）。"
                              f"完成後會自動存入模型庫，名稱為「{deployed_name}」。"
                              "進度請到「模型訓練中心」查看，也可在該頁停止訓練。")
                except Exception as train_err:
                    reply += f"\n\n⚙️ [系統提示]：訓練啟動失敗：{train_err}"

        return {"status": "success", "reply": reply}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/ai/auto_label_status")
async def get_auto_label_status():
    return auto_label_status

# ------------------------------
# Manager Agent API（LangGraph，獨立於既有 AI 助手對話窗，不影響原本功能）
# ------------------------------
class ManagerAgentChatRequest(BaseModel):
    message: str
    model_type: str = "gemini"
    thread_id: str

class ManagerAgentResumeRequest(BaseModel):
    thread_id: str
    model_type: str = "gemini"
    approved: bool
    interrupt_id: str
    operation_id: Optional[str] = None


# 這個伺服器行程啟動時產生一次的識別碼——重開終端機（重啟 server）會換一組新的，
# 同一個行程一直活著（前端重新整理頁面）則不變。前端拿它來區分「重新整理」跟
# 「重開終端」：boot_id 沒變就沿用同一個 thread_id、把對話畫回來；變了就開新對話。
SERVER_BOOT_ID = uuid.uuid4().hex

# 依 model_type 各自快取一份 Graph，換模型（gemini/groq/gemma/local）不用重啟服務
_manager_agent_graphs = {}
_manager_agent_thread_locks = {}
_manager_agent_thread_locks_guard = threading.Lock()


def _try_acquire_agent_thread(thread_id: str):
    """同一個 LangGraph thread 同一時間只允許一條 stream 推進 checkpoint。"""
    with _manager_agent_thread_locks_guard:
        lock = _manager_agent_thread_locks.setdefault(thread_id, threading.Lock())
    return lock if lock.acquire(blocking=False) else None

@app.get("/api/agent/boot")
async def agent_boot():
    """回傳這個伺服器行程的啟動識別碼，讓前端判斷是「重新整理」還是「重開終端」。"""
    return {"status": "success", "boot_id": SERVER_BOOT_ID}


# Agent 要做推論時需要一個「圖片的絕對路徑」（skill_infer 收的是路徑，不是檔案本身）。
# 聊天視窗原本沒有上傳功能，使用者只能自己去檔案總管複製路徑貼進來。這個端點單純把
# 上傳的圖片存進 data/uploads/ 並回傳絕對路徑，前端拿到後填進輸入框即可。
AGENT_UPLOAD_ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@app.post("/api/agent/upload")
async def agent_upload(file: UploadFile = File(...)):
    try:
        ext = Path(file.filename or "").suffix.lower()
        if ext not in AGENT_UPLOAD_ALLOWED_EXTS:
            return {
                "status": "error",
                "message": f"不支援的檔案類型 {ext or '（無副檔名）'}，請上傳圖片（{', '.join(sorted(AGENT_UPLOAD_ALLOWED_EXTS))}）",
            }

        # 檔名只保留安全字元，並加上時間戳避免互相覆蓋；路徑鎖死在 uploads/ 內
        stem = "".join(c for c in Path(file.filename or "image").stem if c.isalnum() or c in "-_") or "image"
        from datetime import datetime
        safe_name = f"agent_{datetime.now().strftime('%m%d_%H%M%S')}_{stem}{ext}"
        save_path = (UPLOAD_DIR / safe_name).resolve()
        if save_path.parent != UPLOAD_DIR.resolve():
            return {"status": "error", "message": "不合法的檔名"}

        with save_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        return {
            "status": "success",
            "filename": safe_name,
            "path": str(save_path),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/api/agent/roboflow/readiness")
async def roboflow_readiness(dataset_name: Optional[str] = None):
    """回傳 Roboflow 補資料功能的目前資格與原因。

    這是純讀取的診斷 API：不會呼叫 Roboflow、建立 Agent graph 或修改資料集，
    讓 UI 能在使用者真正觸發訓練前清楚呈現為何會／不會出現補資料提示。
    """
    from agent_tools import (
        _count_images,
        _load_roboflow_key,
        check_dataset_health,
        get_latest_roboflow_operation,
    )
    from multi_agent import _active_dataset_name
    from skills import COVERAGE_GATE

    # 面板必須反映前端目前點選的資料集；若沒有傳入名稱，才沿用訓練流程最後
    # 寫入 workspace_state.json 的資料集，保持既有 Agent 流程相容。
    dataset_name = dataset_name or _active_dataset_name()
    if not dataset_name:
        return {
            "status": "success",
            "eligible": False,
            "reason_code": "no_active_dataset",
            "reason": "尚未選擇目前資料集；Roboflow 下載只會在訓練流程找不到資料集時由 Data Agent 處理。",
            "api_key_configured": bool(_load_roboflow_key()),
        }

    dataset_path = (DATASET_DIR / dataset_name).resolve()
    if dataset_path.parent != DATASET_DIR.resolve() or not dataset_path.is_dir():
        return {
            "status": "error",
            "message": "找不到指定的資料集",
        }
    image_count = _count_images(dataset_path)
    health = check_dataset_health.invoke({"dataset_name": dataset_name})
    health_ok = health.get("status") == "success" and (health.get("coverage") or 0) >= COVERAGE_GATE
    class_distribution = health.get("class_distribution") or {}
    used_classes = [item for item in class_distribution.values() if (item.get("count") or 0) > 0]
    single_class = len(used_classes) == 1

    state = {}
    state_path = BASE_DIR / "workspace_state.json"
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    evaluation = (state.get("model") or {}).get("evaluation") or {}

    api_key_configured = bool(_load_roboflow_key())
    if not api_key_configured:
        eligible, reason_code = False, "api_key_missing"
        reason = "尚未設定 Roboflow API Key，Agent 無法搜尋 Universe 候選。"
    elif not health_ok:
        eligible, reason_code = True, "health_gate_failed"
        reason = "資料集健康檢查未通過；Agent 會先搜尋並在暫存區評估候選，最後才請你確認。"
    elif evaluation.get("passed") is False:
        eligible, reason_code = True, "eval_gate_failed"
        reason = "最近一次模型評估未通過；Agent 可自動尋找並評估適合補強的 Roboflow 資料。"
    else:
        eligible, reason_code = True, "ready_on_request"
        reason = "可以直接請 Agent 搜尋與評估 Roboflow 候選；確認前不會修改正式資料集。"

    return {
        "status": "success",
        "dataset_name": dataset_name,
        "image_count": image_count,
        "health_score": health.get("score"),
        "coverage": health.get("coverage"),
        "single_class": single_class,
        "evaluation_passed": evaluation.get("passed"),
        "last_map50": ((state.get("model") or {}).get("last_training_metrics") or {}).get("mAP50"),
        "api_key_configured": api_key_configured,
        "eligible": eligible,
        "reason_code": reason_code,
        "reason": reason,
        "supports_class_mapping": True,
        "latest_operation": get_latest_roboflow_operation(dataset_name),
    }


def _get_manager_agent_graph(model_type: str = "gemini"):
    if model_type not in _manager_agent_graphs:
        from manager_agent import get_default_chat_model
        from multi_agent import build_multi_agent
        _manager_agent_graphs[model_type] = build_multi_agent(get_default_chat_model(model_type))
    return _manager_agent_graphs[model_type]

@app.post("/api/agent/chat")
async def agent_chat(request: ManagerAgentChatRequest):
    from multi_agent import stream_trace
    from agent_tools import shared_trainer, is_stop_training_message

    # 訓練中途要終止時，不能走一般流程：Agent 這時卡在前一輪 skill_train.invoke()
    # 裡（同步呼叫，要等訓練真的結束/停止才會回傳），同一個 thread_id 的
    # graph.stream() 沒辦法插進去處理這一則新訊息。所以終止指令在這裡直接用關鍵字
    # 短路掉，跳過 LLM／Graph，直接對 shared_trainer 呼叫 stop()——這樣才能在
    # 前一個請求還卡著的情況下，立刻送出終止訊號讓訓練迴圈在下一個 batch/epoch
    # 邊界真的停下來。
    if shared_trainer.progress.get("status") == "training" and is_stop_training_message(request.message):
        shared_trainer.stop()

        def stop_stream():
            reply = "已送出終止指令，訓練會在目前批次結束後停止（不是立即中斷，請稍候幾秒）。"
            yield f"data: {json.dumps({'type': 'final', 'reply': reply, 'usage': {}, 'llm_calls': 0}, ensure_ascii=False)}\n\n"

        return StreamingResponse(stop_stream(), media_type="text/event-stream")

    graph = _get_manager_agent_graph(request.model_type)
    thread_lock = _try_acquire_agent_thread(request.thread_id)
    if thread_lock is None:
        raise HTTPException(status_code=409, detail="這個 Agent 任務仍在執行中，請等待目前操作完成。")

    config = {"configurable": {"thread_id": request.thread_id}}
    try:
        snapshot = graph.get_state(config)
        if snapshot.interrupts:
            raise HTTPException(
                status_code=409,
                detail="這個 Agent 任務正在等待確認，請先完成目前的確認操作。",
            )
    except Exception:
        thread_lock.release()
        raise

    def event_stream():
        try:
            for entry in stream_trace(graph, request.thread_id, user_message=request.message):
                yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
        finally:
            thread_lock.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.post("/api/agent/resume")
async def agent_resume(request: ManagerAgentResumeRequest):
    """訓練前 interrupt() 暫停後，使用者按確認/取消時呼叫這個端點接著跑，
    跟 /api/agent/chat 一樣是 SSE 串流，差別只在不是送新訊息、是帶著使用者的
    決定（approved）從暫停處恢復（見 multi_agent.stream_trace 的 resume_value）。"""
    from multi_agent import stream_trace
    graph = _get_manager_agent_graph(request.model_type)
    thread_lock = _try_acquire_agent_thread(request.thread_id)
    if thread_lock is None:
        raise HTTPException(status_code=409, detail="這個 Agent 任務仍在執行中，請勿重複送出確認。")

    config = {"configurable": {"thread_id": request.thread_id}}
    try:
        snapshot = graph.get_state(config)
        if not snapshot.interrupts:
            raise HTTPException(status_code=409, detail="這個確認已處理或已失效，請重新整理狀態。")
        pending_interrupt = snapshot.interrupts[0]
        if pending_interrupt.id != request.interrupt_id:
            raise HTTPException(status_code=409, detail="確認識別碼已過期，未執行任何操作。")
        payload = pending_interrupt.value if isinstance(pending_interrupt.value, dict) else {}
        expected_operation_id = payload.get("operation_id")
        if expected_operation_id and request.operation_id != expected_operation_id:
            raise HTTPException(status_code=409, detail="操作識別碼不一致，未啟動訓練。")
    except Exception:
        thread_lock.release()
        raise

    def event_stream():
        try:
            for entry in stream_trace(graph, request.thread_id, resume_value={"approved": request.approved}):
                yield f"data: {json.dumps(entry, ensure_ascii=False)}\n\n"
        finally:
            thread_lock.release()

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.get("/api/agent/pending")
async def agent_pending(thread_id: str, model_type: str = "gemini"):
    """終端機關閉／伺服器重啟會讓前端 SSE 連線中斷，使用者當下彈出的
    confirm() 也會隨之消失，但 SqliteSaver 其實還留著暫停在 interrupt() 的
    checkpoint。前端重新載入頁面時打這個端點，帶著同一個 thread_id 檢查
    是否還卡在某個 interrupt 上，卡著的話把當時的 payload 原樣吐回去，
    讓前端能重新問一次使用者、再送 /api/agent/resume 接續下去。"""
    graph = _get_manager_agent_graph(model_type)
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = graph.get_state(config)
    if snapshot.interrupts:
        pending_interrupt = snapshot.interrupts[0]
        return {
            "status": "success",
            "pending": True,
            "interrupt_id": pending_interrupt.id,
            "payload": pending_interrupt.value,
        }
    return {"status": "success", "pending": False}

@app.get("/api/agent/graph")
async def agent_graph(model_type: str = "gemini"):
    """觀測層：回傳 Multi-Agent 的靜態拓樸（Mermaid 語法），可以貼到
    https://mermaid.live 或前端的 mermaid renderer 看 Manager / Data Agent /
    Model Agent / Evaluation Agent 之間的連線關係。"""
    try:
        graph = _get_manager_agent_graph(model_type)
        return {"status": "success", "mermaid": graph.get_graph().draw_mermaid()}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ------------------------------
# 靜態檔案掛載 (使用絕對路徑以確保多埠啟動正常)
# ------------------------------
app.mount("/static", NoCacheStaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/outputs", NoCacheStaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

@app.get("/")
async def read_index():
    return FileResponse(
        str(STATIC_DIR / "index_v3.html"),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=False, log_level="warning")
