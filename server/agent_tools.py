"""把既有的 YOLOTrainer / YOLOInferencer / AIAnalyzer 包成 LangChain Tool。

Step 2（本檔案的範圍）：只做「整形」——讓這些既有能力具備
Annotated + docstring 的 Tool 格式，未來 Manager Agent 才能用
function-calling 呼叫它們。不修改 trainer.py / inferencer.py /
ai_analyzer.py 的既有邏輯，也不涉及任何 LLM 呼叫。

Manager Agent 本身（何時呼叫、要不要呼叫）留給 Step 3 的 PR。
"""
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Annotated, Optional

import cv2
import requests
from pydantic import Field
from langchain_core.tools import tool

SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from trainer import YOLOTrainer
from inferencer import YOLOInferencer
from ai_analyzer import ai_analyzer

DATASET_DIR = PROJECT_ROOT / "data" / "datasets"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"

ROBOFLOW_EXPORT_FORMAT = "yolov5pytorch"  # Roboflow 沒有獨立的 yolov9 匯出格式，但輸出的 YOLO txt 格式跟 ultralytics 相容

# 全系統唯一一個 YOLOTrainer 實例——train_yolo_model（Agent 用）跟 main.py 的
# /api/train/start /api/train/stop（網頁表單用）都共用這一個物件，兩邊的
# 「終止」才能真的作用在同一份 _stop_event 上。改之前 train_yolo_model 每次呼叫
# 都自己 new 一個 YOLOTrainer()，導致從網頁按「終止」呼叫的是 main.py 自己那個
# 從沒訓練過的 trainer，跟 Agent 真正在跑訓練的那個是兩個不同物件，設 stop_event
# 完全打不到正在跑的那個，看起來就像「按了終止但 CMD 還是繼續跑」。
shared_trainer = YOLOTrainer(data_root=str(DATASET_DIR))

# 訓練是同步、會跑好幾分鐘的呼叫，使用者想中途取消時沒辦法真的插進同一個
# graph.stream() 裡面（Agent 正卡在 skill_train.invoke() 裡）。因此中止指令不透過
# Agent 的推理迴圈，而是在 API 層（main.py 的 /api/agent/chat）用關鍵字比對直接
# 短路掉，跳過 LLM／Graph，直接呼叫 shared_trainer.stop()——這樣才能在訓練還在
# 跑的那個請求卡住時，另一個請求依然能立刻送出終止訊號。
STOP_TRAINING_KEYWORDS = (
    "停止訓練", "終止訓練", "取消訓練", "別訓練了", "stop training", "cancel training",
    # 呼叫端已經先確認「目前真的有訓練在跑」才會拿這份清單來比對，這個前提下使用者
    # 打「終止」「停止」「取消」單獨一個詞，指的幾乎一定是眼前這個訓練，不用擔心
    # 跟其他情境的取消/停止搞混。
    "終止", "停止", "取消", "stop", "cancel",
)


def is_stop_training_message(text: str) -> bool:
    """判斷這句話是不是在要求終止訓練——只在真的有訓練在跑的時候才會被拿來判斷
    （呼叫端要自己先確認 shared_trainer.progress['status'] == 'training'），
    所以這裡的關鍵字可以放寬，不用擔心跟其他情境的「取消」「停止」搞混。"""
    if not text:
        return False
    lowered = text.lower()
    return any(k in text or k in lowered for k in STOP_TRAINING_KEYWORDS)


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    return sum(len(list(path.rglob(ext))) for ext in exts)


def _sanitize_name(name: str) -> str:
    """只取檔名部分並擋掉「.」「..」，避免 dataset_name/model_name（來自 LLM，
    間接來自使用者訊息）被拿來做路徑遍歷，跳出 DATASET_DIR/runs 之外讀寫檔案。"""
    cleaned = Path(str(name)).name
    if not cleaned or cleaned in (".", ".."):
        raise ValueError(f"不合法的名稱：{name!r}")
    return cleaned


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    """解壓前先確認每個成員的目標路徑都落在 dest 底下，防止惡意/被竄改的 zip
    用 ../../ 這種路徑跳出去覆寫任意檔案（zip slip）。"""
    dest_root = dest.resolve()
    for member in zf.infolist():
        target = (dest / member.filename).resolve()
        if target != dest_root and dest_root not in target.parents:
            raise ValueError(f"壓縮檔內含不安全的路徑，已中止解壓：{member.filename}")
    zf.extractall(dest)


def _load_roboflow_key() -> Optional[str]:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if key:
        return key
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("ROBOFLOW_API_KEY="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


@tool
def check_dataset_health(
    dataset_name: Annotated[str, Field(description="資料集名稱，對應 data/datasets/ 底下的資料夾名")],
) -> dict:
    """檢查資料集的健康度（標記覆蓋率、類別平衡、圖片與標籤總數），回傳 0-100 的健康分數。"""
    try:
        dataset_name = _sanitize_name(dataset_name)
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    return ai_analyzer.analyze_dataset_health(dataset_name)


@tool
def import_images_to_dataset(
    source_path: Annotated[str, Field(description="要匯入的圖片路徑，可以是單一圖片檔案，也可以是一個資料夾（例如推論頁面暫存上傳圖片的 data/uploads/ 或 temp_verify/）")],
    dataset_name: Annotated[str, Field(description="要匯入到的目標資料集名稱，對應 data/datasets/ 底下的資料夾名")],
) -> dict:
    """把本機任意路徑（例如使用者已經上傳的圖片）匯入成資料集，存到 data/datasets/<dataset_name>/images/，
    之後就能被 check_dataset_health / train_yolo_model 使用。

    這個工具只搬圖片，不會產生任何標記檔（.txt），匯入後 check_dataset_health 會如實顯示
    0% 標記率，這是正常情況，不代表匯入失敗——用於「先驗證看看、之後再標記」這種流程。"""
    try:
        dataset_name = _sanitize_name(dataset_name)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    src = Path(source_path)
    if not src.exists():
        return {"status": "error", "message": f"來源路徑不存在：{source_path}"}

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if src.is_file():
        candidates = [src] if src.suffix.lower() in exts else []
    else:
        candidates = [p for p in src.rglob("*") if p.is_file() and p.suffix.lower() in exts]

    if not candidates:
        return {"status": "error", "message": f"來源路徑底下沒有找到任何圖片：{source_path}"}

    dest_dir = DATASET_DIR / dataset_name / "images"
    dest_dir.mkdir(parents=True, exist_ok=True)

    imported_paths = []
    for img in candidates:
        dest = dest_dir / img.name
        shutil.copy2(img, dest)
        imported_paths.append(str(dest.resolve()))

    return {
        "status": "success",
        "dataset_name": dataset_name,
        "imported_count": len(imported_paths),
        "imported_paths": imported_paths,
        "total_images_now": _count_images(DATASET_DIR / dataset_name),
    }


@tool
def train_yolo_model(
    dataset_name: Annotated[str, Field(description="要用來訓練的資料集名稱")],
    variant: Annotated[str, Field(description="YOLO 權重檔名，例如 yolov9m.pt")] = "yolov9m.pt",
    epochs: Annotated[int, Field(description="訓練輪數")] = 50,
    imgsz: Annotated[int, Field(description="輸入影像尺寸")] = 640,
    batch: Annotated[int, Field(description="batch size")] = 16,
    model_name: Annotated[str, Field(description="輸出的實驗名稱，會存在 runs/train/<model_name> 底下")] = "exp",
) -> dict:
    """使用指定資料集訓練 YOLO 模型，回傳最終狀態（completed/error/stopped）。

    若本機找不到這個資料集（沒有圖片），會回傳 status="dataset_not_found"，
    這時應該呼叫 download_roboflow_dataset 從 Roboflow Universe 補齊資料後再重試，
    不要直接放棄。"""
    try:
        dataset_name = _sanitize_name(dataset_name)
        model_name = _sanitize_name(model_name)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    ds_path = DATASET_DIR / dataset_name
    if _count_images(ds_path) == 0:
        return {
            "status": "dataset_not_found",
            "message": f"本機找不到資料集 {dataset_name}（沒有圖片），"
                       f"請呼叫 download_roboflow_dataset 從 Roboflow Universe 下載後再重試",
        }

    if shared_trainer.progress.get("status") == "training":
        return {
            "status": "already_training",
            "message": "已有一個訓練正在進行中（可能是網頁表單或前一輪 Agent 啟動的），"
                       "請等它結束或先要求終止，不要同時啟動第二個訓練",
        }

    data_yaml = shared_trainer.prepare_dataset(str(ds_path))
    shared_trainer.train(
        variant=variant,
        data_yaml=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        model_name=model_name,
    )
    return {
        "status": shared_trainer.progress.get("status"),
        "message": shared_trainer.progress.get("message"),
        "model_name": model_name,
        "epochs_run": shared_trainer.progress.get("current_epoch"),
    }


@tool
def evaluate_yolo_model(
    model_name: Annotated[str, Field(description="要評估的實驗名稱，對應 runs/train/<model_name>")] = "exp",
) -> dict:
    """讀取指定實驗的訓練結果 CSV，回傳 mAP、Loss 與是否有過擬合警告的摘要。"""
    df = ai_analyzer.get_latest_results(model_name)
    summary = ai_analyzer.summarize_metrics(df)
    if isinstance(summary, str):
        return {"status": "no_data", "message": summary}
    return {"status": "success", **summary}


@tool
def run_yolo_inference(
    image_path: Annotated[str, Field(description="要偵測的圖片絕對路徑")],
    model_path: Annotated[Optional[str], Field(description="模型權重路徑，留空則用預設的 models/yolov9m.pt")] = None,
    conf: Annotated[float, Field(description="信心值門檻")] = 0.25,
    iou: Annotated[float, Field(description="IoU 門檻")] = 0.45,
) -> dict:
    """對單張圖片執行 YOLO 推論，回傳偵測到的物件清單與標記後圖片的存檔路徑。"""
    inferencer = YOLOInferencer(model_path=model_path) if model_path else YOLOInferencer()
    detections, annotated_img = inferencer.predict(image_path, conf=conf, iou=iou)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"tool_infer_{Path(image_path).stem}.jpg"
    cv2.imwrite(str(out_path), annotated_img)

    return {"objects": detections, "annotated_image_path": str(out_path)}


@tool
def download_roboflow_dataset(
    workspace: Annotated[str, Field(description="Roboflow workspace slug，來自 universe.roboflow.com/<workspace>/<project> 網址")],
    project: Annotated[str, Field(description="Roboflow project slug，同樣來自該網址")],
    version: Annotated[int, Field(description="要下載的資料集版本號，通常從 1 開始")] = 1,
    dataset_name: Annotated[Optional[str], Field(description="下載後存到 data/datasets/ 底下的資料夾名稱，留空則用 project slug")] = None,
) -> dict:
    """從 Roboflow Universe 下載指定的公開資料集，解壓成 YOLO 格式存到 data/datasets/<dataset_name>/，
    下載完就能直接被 train_yolo_model 使用。

    呼叫前要先猜出正確的 workspace 與 project slug（來自 universe.roboflow.com 的網址，
    例如 https://universe.roboflow.com/roboflow-100/vehicles-oal2j 對應
    workspace="roboflow-100", project="vehicles-oal2j"）。不確定的話就用最符合關鍵字的
    猜測去試；回傳 status="not_found" 代表這個 workspace/project 不存在，換一個猜測再試一次。
    """
    api_key = _load_roboflow_key()
    if not api_key:
        return {"status": "error", "message": "未設定 ROBOFLOW_API_KEY，無法下載"}

    try:
        target_name = _sanitize_name(dataset_name or project)
    except ValueError as e:
        return {"status": "error", "message": str(e)}
    export_url = (
        f"https://api.roboflow.com/{workspace}/{project}/{version}/{ROBOFLOW_EXPORT_FORMAT}"
        f"?api_key={api_key}"
    )

    try:
        resp = requests.get(export_url, timeout=30)
    except requests.RequestException as e:
        return {"status": "error", "message": f"連線失敗：{e}"}

    if resp.status_code == 404:
        return {
            "status": "not_found",
            "message": f"找不到 {workspace}/{project} 第 {version} 版，換一個 workspace/project 猜測再試",
        }
    if not resp.ok:
        return {"status": "error", "message": f"Roboflow 回應錯誤（HTTP {resp.status_code}）：{resp.text[:200]}"}

    link = resp.json().get("export", {}).get("link")
    if not link:
        return {"status": "error", "message": "回應中沒有下載連結，可能該版本不支援匯出"}

    zip_resp = requests.get(link, timeout=120)
    zip_resp.raise_for_status()

    ds_path = DATASET_DIR / target_name
    ds_path.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
            _safe_extract_zip(zf, ds_path)
    except (zipfile.BadZipFile, ValueError) as e:
        return {"status": "error", "message": f"解壓失敗：{e}"}

    return {
        "status": "success",
        "dataset_name": target_name,
        "source": f"roboflow:{workspace}/{project}/v{version}",
        "image_count": _count_images(ds_path),
    }


ALL_TOOLS = [check_dataset_health, import_images_to_dataset, train_yolo_model, evaluate_yolo_model, run_yolo_inference, download_roboflow_dataset]
