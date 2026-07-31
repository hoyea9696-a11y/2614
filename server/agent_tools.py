"""把既有的 YOLOTrainer / YOLOInferencer / AIAnalyzer 包成 LangChain Tool。

Step 2（本檔案的範圍）：只做「整形」——讓這些既有能力具備
Annotated + docstring 的 Tool 格式，未來 Manager Agent 才能用
function-calling 呼叫它們。不修改 trainer.py / inferencer.py /
ai_analyzer.py 的既有邏輯，也不涉及任何 LLM 呼叫。

Manager Agent 本身（何時呼叫、要不要呼叫）留給 Step 3 的 PR。
"""
import hashlib
import io
import json
import os
import re
import shutil
import sys
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Optional

import cv2
import requests
import yaml
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
ROBOFLOW_PROPOSAL_DIR = PROJECT_ROOT / "temp_training" / "roboflow_proposals"
ROBOFLOW_HISTORY_PATH = PROJECT_ROOT / "data" / "roboflow_history.json"
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

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


def _normalize_class_name(name: str) -> str:
    """轉成可安全比對的類別名稱；只做明確比對，不用模糊猜測污染標籤。"""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def _read_dataset_class_names(root: Path) -> list[str]:
    """從 YOLO data.yaml 或常見 class name 檔案讀取類別名稱。"""
    yaml_candidates = [root / "data.yaml", root / "dataset.yaml"]
    yaml_candidates.extend(root.glob("*.yaml"))
    yaml_candidates.extend(root.glob("*.yml"))
    for path in yaml_candidates:
        if not path.exists():
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            names = data.get("names")
            if isinstance(names, list):
                return [str(name) for name in names]
            if isinstance(names, dict):
                return [str(names[key]) for key in sorted(names, key=lambda value: int(value))]
        except (OSError, ValueError, TypeError, yaml.YAMLError):
            continue
    for filename in ("classes.txt", "_classes.txt", "obj.names"):
        for path in root.rglob(filename):
            try:
                names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if names:
                    return names
            except OSError:
                continue
    return []


def _label_path_for_image(image_path: Path) -> Path:
    if image_path.parent.name == "images":
        return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
    return image_path.with_suffix(".txt")


def _image_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _existing_image_hashes(root: Path) -> set[str]:
    hashes = set()
    if not root.exists():
        return hashes
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            try:
                hashes.add(_image_hash(path))
            except OSError:
                continue
    return hashes


def _parse_roboflow_candidate(raw: dict) -> Optional[dict]:
    workspace = raw.get("workspace") or {}
    workspace_slug = workspace.get("url") if isinstance(workspace, dict) else workspace
    universe_url = str(raw.get("url") or "").rstrip("/")
    project_slug = universe_url.rsplit("/", 1)[-1] if "/" in universe_url else ""
    version = raw.get("latestVersion")
    if not workspace_slug or not project_slug:
        return None
    try:
        version = int(version) if version is not None else None
    except (TypeError, ValueError):
        version = None
    return {
        "name": raw.get("name") or project_slug,
        "workspace": str(workspace_slug),
        "project": project_slug,
        "version": version,
        "url": universe_url,
        "type": raw.get("type"),
        "classes": [str(name) for name in (raw.get("classes") or [])],
        "images": int(raw.get("images") or 0),
        "license": raw.get("license") or "未標示",
        "stars": int(raw.get("stars") or 0),
        "downloads": int(raw.get("downloads") or 0),
    }


def _search_roboflow_universe(query: str, max_results: int = 8) -> dict:
    api_key = _load_roboflow_key()
    if not api_key:
        return {"status": "error", "message": "未設定 ROBOFLOW_API_KEY，無法搜尋 Roboflow Universe"}
    try:
        response = requests.get(
            "https://api.roboflow.com/universe/search",
            params={"q": query, "api_key": api_key, "page": 1},
            timeout=30,
        )
    except requests.RequestException as exc:
        return {"status": "error", "message": f"Roboflow Universe 搜尋失敗：{exc}"}
    if not response.ok:
        return {
            "status": "error",
            "message": f"Roboflow Universe 搜尋失敗（HTTP {response.status_code}）：{response.text[:200]}",
        }
    candidates = []
    for raw in (response.json().get("results") or []):
        candidate = _parse_roboflow_candidate(raw)
        if candidate:
            candidates.append(candidate)
        if len(candidates) >= max(1, min(max_results, 20)):
            break
    return {"status": "success", "query": query, "candidates": candidates}


def _download_roboflow_export(workspace: str, project: str, version: int, destination: Path) -> dict:
    api_key = _load_roboflow_key()
    if not api_key:
        return {"status": "error", "message": "未設定 ROBOFLOW_API_KEY，無法下載"}
    export_url = (
        f"https://api.roboflow.com/{workspace}/{project}/{version}/{ROBOFLOW_EXPORT_FORMAT}"
        f"?api_key={api_key}"
    )
    try:
        response = requests.get(export_url, timeout=30)
    except requests.RequestException as exc:
        return {"status": "error", "message": f"取得 Roboflow 匯出連結失敗：{exc}"}
    if response.status_code == 404:
        return {"status": "not_found", "message": f"找不到 {workspace}/{project} 第 {version} 版"}
    if not response.ok:
        return {
            "status": "error",
            "message": f"Roboflow 回應錯誤（HTTP {response.status_code}）：{response.text[:200]}",
        }
    link = response.json().get("export", {}).get("link")
    if not link:
        return {"status": "error", "message": "Roboflow 回應中沒有資料集下載連結"}
    try:
        archive = requests.get(link, timeout=120)
        archive.raise_for_status()
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(archive.content)) as zf:
            _safe_extract_zip(zf, destination)
    except (requests.RequestException, zipfile.BadZipFile, ValueError) as exc:
        return {"status": "error", "message": f"Roboflow 資料下載或解壓失敗：{exc}"}
    return {"status": "success"}


def _proposal_path(proposal_id: str) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(proposal_id))
    if not safe_id or safe_id != proposal_id:
        raise ValueError("Roboflow proposal_id 格式不正確")
    return ROBOFLOW_PROPOSAL_DIR / safe_id


def _load_roboflow_proposal(proposal_id: str) -> dict:
    proposal_file = _proposal_path(proposal_id) / "proposal.json"
    if not proposal_file.exists():
        return {"status": "error", "message": "找不到這份 Roboflow 評估結果，可能已過期"}
    try:
        return json.loads(proposal_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"status": "error", "message": "Roboflow 評估結果無法讀取"}


def _save_roboflow_proposal(proposal: dict) -> None:
    proposal_root = _proposal_path(proposal["proposal_id"])
    proposal_root.mkdir(parents=True, exist_ok=True)
    (proposal_root / "proposal.json").write_text(
        json.dumps(proposal, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _append_roboflow_history(record: dict) -> None:
    history = []
    try:
        history = json.loads(ROBOFLOW_HISTORY_PATH.read_text(encoding="utf-8"))
        if not isinstance(history, list):
            history = []
    except (OSError, json.JSONDecodeError):
        pass
    history.append(record)
    ROBOFLOW_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    ROBOFLOW_HISTORY_PATH.write_text(
        json.dumps(history[-100:], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_latest_roboflow_operation(dataset_name: Optional[str] = None) -> Optional[dict]:
    try:
        history = json.loads(ROBOFLOW_HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(history, list):
        return None
    for record in reversed(history):
        if not dataset_name or record.get("dataset_name") == dataset_name:
            return record
    return None


def discard_roboflow_proposal(proposal_id: str) -> dict:
    try:
        proposal_root = _proposal_path(proposal_id)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    if proposal_root.exists():
        shutil.rmtree(proposal_root, ignore_errors=True)
    return {"status": "discarded", "proposal_id": proposal_id}


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
def search_roboflow_universe(
    query: Annotated[str, Field(description="要在 Roboflow Universe 搜尋的主題，例如 bear object detection")],
    max_results: Annotated[int, Field(description="最多回傳幾個候選資料集，範圍 1 到 20")] = 8,
) -> dict:
    """真正呼叫 Roboflow Universe Search API 搜尋公開資料集，不再猜 workspace/project slug。

    只讀取候選資料集的公開摘要，不下載、不解壓，也不修改目前資料集。
    """
    return _search_roboflow_universe(query, max_results)


@tool
def prepare_roboflow_augmentation(
    dataset_name: Annotated[str, Field(description="要補強的本機資料集名稱")],
    query: Annotated[Optional[str], Field(description="搜尋主題；留空時使用資料集名稱加 object detection")] = None,
    max_candidates: Annotated[int, Field(description="要比較的候選數量，範圍 1 到 10")] = 5,
) -> dict:
    """搜尋 Roboflow 候選、挑選相容者並下載到暫存區完成評估。

    這個工具不會修改正式資料集。它會檢查實際匯出內容、類別對應、重複圖片與可用
    標註，回傳一份 proposal；只有使用者之後確認 proposal，才會正式合併。
    """
    try:
        dataset_name = _sanitize_name(dataset_name)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    dataset_root = DATASET_DIR / dataset_name
    if _count_images(dataset_root) == 0:
        return {"status": "error", "message": f"資料集 {dataset_name} 不存在或沒有圖片"}
    local_classes = _read_dataset_class_names(dataset_root)
    if not local_classes:
        return {"status": "error", "message": f"資料集 {dataset_name} 沒有可辨識的類別名稱，無法安全對應"}

    search_query = (query or f"{dataset_name} object detection").strip()
    search = _search_roboflow_universe(search_query, max(1, min(max_candidates, 10)))
    if search.get("status") != "success":
        return search
    candidates = [
        item for item in search.get("candidates", [])
        if item.get("type") == "object-detection" and item.get("version")
    ]
    local_normalized = {_normalize_class_name(name): index for index, name in enumerate(local_classes)}

    def candidate_score(item: dict) -> tuple:
        overlaps = sum(1 for name in item.get("classes", []) if _normalize_class_name(name) in local_normalized)
        has_license = item.get("license") not in (None, "", "未標示")
        return (
            overlaps > 0,
            overlaps,
            has_license,
            min(item.get("images", 0), 2000),
            item.get("stars", 0),
        )

    candidates.sort(key=candidate_score, reverse=True)
    candidate_summaries = candidates[:]
    existing_hashes = _existing_image_hashes(dataset_root)

    for candidate in candidates:
        proposal_id = uuid.uuid4().hex
        proposal_root = _proposal_path(proposal_id)
        staged_root = proposal_root / "dataset"
        download = _download_roboflow_export(
            candidate["workspace"],
            candidate["project"],
            candidate["version"],
            staged_root,
        )
        if download.get("status") != "success":
            shutil.rmtree(proposal_root, ignore_errors=True)
            continue

        source_classes = _read_dataset_class_names(staged_root)
        class_mapping = []
        mapping_by_id = {}
        for source_id, source_name in enumerate(source_classes):
            target_id = local_normalized.get(_normalize_class_name(source_name))
            if target_id is None:
                continue
            mapping_by_id[source_id] = target_id
            class_mapping.append({
                "source_id": source_id,
                "source_name": source_name,
                "target_id": target_id,
                "target_name": local_classes[target_id],
            })
        if not class_mapping:
            shutil.rmtree(proposal_root, ignore_errors=True)
            continue

        usable_images = 0
        unlabeled_images = 0
        unmapped_images = 0
        duplicate_images = 0
        seen_hashes = set(existing_hashes)
        for image in staged_root.rglob("*"):
            if not (image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS):
                continue
            label_path = _label_path_for_image(image)
            if not label_path.exists() or label_path.stat().st_size == 0:
                unlabeled_images += 1
                continue
            mapped_lines = 0
            for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                try:
                    source_id = int(parts[0])
                except (IndexError, ValueError):
                    continue
                if len(parts) >= 5 and source_id in mapping_by_id:
                    mapped_lines += 1
            if mapped_lines == 0:
                unmapped_images += 1
                continue
            try:
                digest = _image_hash(image)
            except OSError:
                continue
            if digest in seen_hashes:
                duplicate_images += 1
                continue
            seen_hashes.add(digest)
            usable_images += 1

        if usable_images == 0:
            shutil.rmtree(proposal_root, ignore_errors=True)
            continue

        risks = []
        unmapped_classes = [name for index, name in enumerate(source_classes) if index not in mapping_by_id]
        if unmapped_classes:
            risks.append(f"未對應類別將跳過：{', '.join(unmapped_classes)}")
        if duplicate_images:
            risks.append(f"{duplicate_images} 張重複圖片將跳過")
        if candidate.get("license") == "未標示":
            risks.append("來源未標示授權，合併前請自行確認使用權")

        proposal = {
            "status": "proposal_ready",
            "proposal_id": proposal_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dataset_name": dataset_name,
            "query": search_query,
            "candidate": candidate,
            "candidate_count": len(candidates),
            "candidate_summaries": candidate_summaries,
            "local_classes": local_classes,
            "source_classes": source_classes,
            "class_mapping": class_mapping,
            "baseline_images": _count_images(dataset_root),
            "usable_images": usable_images,
            "unlabeled_images": unlabeled_images,
            "unmapped_images": unmapped_images,
            "duplicate_images": duplicate_images,
            "estimated_total_images": _count_images(dataset_root) + usable_images,
            "risks": risks,
        }
        _save_roboflow_proposal(proposal)
        return proposal

    return {
        "status": "no_compatible_candidates",
        "message": "找到候選資料集，但沒有任何候選同時具備可下載版本、可明確對應的類別與可用標註。",
        "query": search_query,
        "candidates": candidate_summaries,
        "local_classes": local_classes,
    }


@tool
def apply_roboflow_proposal(
    proposal_id: Annotated[str, Field(description="prepare_roboflow_augmentation 產生的 proposal_id")],
) -> dict:
    """使用者確認後，才把已評估的 Roboflow 暫存資料正式合併進目標資料集。"""
    try:
        proposal = _load_roboflow_proposal(proposal_id)
    except ValueError as exc:
        return {"status": "error", "message": str(exc)}
    if proposal.get("status") != "proposal_ready":
        return proposal if proposal.get("status") == "error" else {
            "status": "error",
            "message": "這份 Roboflow 評估已經處理過，不能重複套用",
        }
    dataset_name = proposal["dataset_name"]
    dataset_root = DATASET_DIR / dataset_name
    staged_root = _proposal_path(proposal_id) / "dataset"
    if _count_images(dataset_root) == 0 or not staged_root.exists():
        return {"status": "error", "message": "正式資料集或 Roboflow 暫存資料已不存在"}

    if (dataset_root / "train" / "images").exists():
        image_destination = dataset_root / "train" / "images"
        label_destination = dataset_root / "train" / "labels"
    else:
        image_destination = dataset_root / "images"
        label_destination = dataset_root / "labels"
    image_destination.mkdir(parents=True, exist_ok=True)
    label_destination.mkdir(parents=True, exist_ok=True)

    mapping_by_id = {
        int(item["source_id"]): int(item["target_id"])
        for item in proposal.get("class_mapping", [])
    }
    existing_hashes = _existing_image_hashes(dataset_root)
    prefix = _sanitize_name(proposal["candidate"]["project"])
    added = 0
    skipped = 0
    for image in staged_root.rglob("*"):
        if not (image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS):
            continue
        label_path = _label_path_for_image(image)
        if not label_path.exists():
            skipped += 1
            continue
        remapped = []
        for line in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            parts = line.split()
            try:
                source_id = int(parts[0])
            except (IndexError, ValueError):
                continue
            if len(parts) >= 5 and source_id in mapping_by_id:
                parts[0] = str(mapping_by_id[source_id])
                remapped.append(" ".join(parts))
        if not remapped:
            skipped += 1
            continue
        try:
            digest = _image_hash(image)
        except OSError:
            skipped += 1
            continue
        if digest in existing_hashes:
            skipped += 1
            continue
        existing_hashes.add(digest)

        destination = image_destination / f"{prefix}_{image.name}"
        suffix_index = 1
        while destination.exists():
            destination = image_destination / f"{prefix}_{suffix_index}_{image.name}"
            suffix_index += 1
        shutil.copy2(image, destination)
        (label_destination / f"{destination.stem}.txt").write_text(
            "\n".join(remapped) + "\n",
            encoding="utf-8",
        )
        added += 1

    applied_at = datetime.now(timezone.utc).isoformat()
    result = {
        "status": "success" if added else "no_new_images",
        "proposal_id": proposal_id,
        "dataset_name": dataset_name,
        "source": (
            f"roboflow:{proposal['candidate']['workspace']}/"
            f"{proposal['candidate']['project']}/v{proposal['candidate']['version']}"
        ),
        "source_url": proposal["candidate"]["url"],
        "license": proposal["candidate"]["license"],
        "class_mapping": proposal["class_mapping"],
        "added_images": added,
        "skipped_images": skipped,
        "before_images": proposal["baseline_images"],
        "total_images_now": _count_images(dataset_root),
        "applied_at": applied_at,
    }
    proposal.update(result)
    _save_roboflow_proposal(proposal)
    _append_roboflow_history(result)
    shutil.rmtree(staged_root, ignore_errors=True)
    return result


@tool
def download_roboflow_dataset(
    workspace: Annotated[str, Field(description="Roboflow workspace slug，來自 universe.roboflow.com/<workspace>/<project> 網址")],
    project: Annotated[str, Field(description="Roboflow project slug，同樣來自該網址")],
    version: Annotated[int, Field(description="要下載的資料集版本號，通常從 1 開始")] = 1,
    dataset_name: Annotated[Optional[str], Field(description="下載後存到 data/datasets/ 底下的資料夾名稱，留空則用 project slug")] = None,
) -> dict:
    """從 Roboflow Universe 下載指定的公開資料集，解壓成 YOLO 格式存到 data/datasets/<dataset_name>/，
    下載完就能直接被 train_yolo_model 使用。

    呼叫前應先用 search_roboflow_universe 取得真實的 workspace、project 與最新版本，
    不要自行猜測 slug。這個工具用於建立全新資料集；補強既有資料集應改走
    prepare_roboflow_augmentation，並等待使用者確認 proposal。
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


@tool
def augment_dataset_from_roboflow(
    dataset_name: Annotated[str, Field(description="要擴充（合併進去）的現有資料集名稱，對應 data/datasets/ 底下的資料夾")],
    workspace: Annotated[str, Field(description="Roboflow workspace slug，來自 universe.roboflow.com/<workspace>/<project> 網址")],
    project: Annotated[str, Field(description="Roboflow project slug，同樣來自該網址；挑一個跟現有資料集『同類主題』的公開專案")],
    version: Annotated[int, Field(description="要下載的資料集版本號，通常從 1 開始")] = 1,
) -> dict:
    """資料量不足／分數偏低時，從 Roboflow Universe 抓一個『同類主題』的公開資料集，
    把它的已標記圖片『合併進』現有的 dataset_name，用來擴充樣本後再重新訓練。

    重要限制：合併進來的標籤會被一律重新對應成單一類別（class 0）。這對「單一類別
    偵測」是安全的；多類別資料集不要用這個工具自動合併，否則類別會對不上、標籤變無效。

    跟 download_roboflow_dataset 的差別：那個是「建立一個全新的資料集」，這個是「把
    新圖片疊加進已存在的資料集」。呼叫前一樣要自己猜 workspace/project slug，回傳
    status="not_found" 就換一個猜測再試。"""
    api_key = _load_roboflow_key()
    if not api_key:
        return {"status": "error", "message": "未設定 ROBOFLOW_API_KEY，無法下載"}

    try:
        dataset_name = _sanitize_name(dataset_name)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    ds_root = DATASET_DIR / dataset_name
    if _count_images(ds_root) == 0:
        return {
            "status": "error",
            "message": f"資料集 {dataset_name} 不存在或沒有圖片，無法擴充；"
                       f"要從零建立請改用 download_roboflow_dataset",
        }

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

    # 解壓到暫存區（不放進 datasets/，避免半成品被 _count_images 或訓練誤讀），合併完就刪。
    tmp_root = PROJECT_ROOT / "temp_training" / f"_augment_{_sanitize_name(project)}"
    if tmp_root.exists():
        shutil.rmtree(tmp_root, ignore_errors=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        zip_resp = requests.get(link, timeout=120)
        zip_resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
            _safe_extract_zip(zf, tmp_root)

        # 現有資料集有 train/images 就併進去，否則併進根目錄的 images/（兩種結構
        # prepare_dataset 都吃得到）。
        if (ds_root / "train" / "images").exists():
            img_dst, lbl_dst = ds_root / "train" / "images", ds_root / "train" / "labels"
        else:
            img_dst, lbl_dst = ds_root / "images", ds_root / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        added = 0
        for img in tmp_root.rglob("*"):
            if not (img.is_file() and img.suffix.lower() in exts):
                continue
            # YOLO 匯出格式：圖片在 .../images/，標籤在同層的 .../labels/<stem>.txt
            label_src = (img.parent.parent / "labels" / (img.stem + ".txt")) if "images" in img.parts else (img.parent / (img.stem + ".txt"))
            if not (label_src.exists() and label_src.stat().st_size > 0):
                continue  # 只併已標記的圖片，沒標籤的略過
            base = f"{_sanitize_name(project)}_{img.name}"
            dst_img = img_dst / base
            if dst_img.exists():
                base = f"{_sanitize_name(project)}_{added}_{img.name}"
                dst_img = img_dst / base
            shutil.copy2(img, dst_img)
            # 標籤逐行把 class id（第一個 token）改寫成 0，其餘 bbox 座標原樣保留
            remapped = []
            for line in label_src.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    parts[0] = "0"
                    remapped.append(" ".join(parts))
            (lbl_dst / (dst_img.stem + ".txt")).write_text("\n".join(remapped) + "\n", encoding="utf-8")
            added += 1
    except (zipfile.BadZipFile, ValueError) as e:
        return {"status": "error", "message": f"解壓/合併失敗：{e}"}
    except requests.RequestException as e:
        return {"status": "error", "message": f"下載失敗：{e}"}
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    if added == 0:
        return {
            "status": "no_new_images",
            "message": f"{workspace}/{project} 沒有可用的已標記圖片可以合併（可能格式不符或沒有標籤）",
        }
    return {
        "status": "success",
        "dataset_name": dataset_name,
        "source": f"roboflow:{workspace}/{project}/v{version}",
        "added_images": added,
        "total_images_now": _count_images(ds_root),
    }


ALL_TOOLS = [
    check_dataset_health,
    import_images_to_dataset,
    train_yolo_model,
    evaluate_yolo_model,
    run_yolo_inference,
    search_roboflow_universe,
    prepare_roboflow_augmentation,
    download_roboflow_dataset,
]
