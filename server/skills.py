"""Skill 層：把 .agents/SKILLS.md 原本設計、但一直沒有真正實作的 Gate 邏輯做出來。

跟 agent_tools.py 的差別：
- Tool（agent_tools.py）只管「怎麼執行」，不做任何判斷。
- Skill（這個檔案）管「固定的執行順序」與「哪一步該跳過/擋下」，這些判斷寫死在
  Python 裡，不透過 LLM 決定，理由是 AI Detector Multi-Agent System Architecture.docx
  第 8.4 節：檔案/資料是否存在這類判斷應該用固定 Rule，結果才會穩定、不耗 LLM Token。

Skill 內部直接呼叫 agent_tools.py 裡的 Tool（用 .invoke()），呼叫順序跟要不要往下走
都是 Python if/else 決定，不是 LLM。每個 Skill 對外仍然包成一個 @tool，讓 Agent
可以把它當成「一個能力」呼叫，而不是自己去串多個原始 Tool。
"""
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Annotated, Optional

from pydantic import Field
from langchain_core.tools import tool

SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent_tools import (
    DATASET_DIR,
    _count_images,
    _sanitize_name,
    check_dataset_health,
    train_yolo_model,
    run_yolo_inference,
    evaluate_yolo_model,
)

WORKSPACE_STATE_PATH = PROJECT_ROOT / "workspace_state.json"
# 標記覆蓋率門檻（%）：低於此代表還有圖片沒標記，硬訓會學到錯的東西，這才是真正該
# 擋下來的問題。類別平不平衡刻意不在這裡擋——物件偵測本來就容許不平衡（YOLO 照樣
# 訓，只是稀有類表現差），把「不平衡」當成硬性 Gate 會誤殺正常、可訓練的資料集
# （例如 car/truck/person 這種天生就一類特別多的資料）。
COVERAGE_GATE = 70
# 類別平衡度低於此只提出警告、不擋訓練，讓使用者自己決定要不要為稀有類別補資料。
BALANCE_WARN_THRESHOLD = 40
EVAL_MAP50_GATE = 0.5
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "yolov9m.pt"

TAIPEI_TZ = timezone(timedelta(hours=8))


def _now_iso() -> str:
    return datetime.now(TAIPEI_TZ).isoformat()


def _read_state() -> dict:
    if not WORKSPACE_STATE_PATH.exists():
        return {}
    with open(WORKSPACE_STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_state(state: dict) -> None:
    """先寫到暫存檔再 replace，避免 json.dump 中途因型別無法序列化而拋例外時，
    留下寫到一半的殘缺 JSON，導致之後所有 _read_state() 都壞掉。"""
    tmp_path = WORKSPACE_STATE_PATH.with_suffix(".json.tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp_path.replace(WORKSPACE_STATE_PATH)


@tool
def skill_train(
    dataset_name: Annotated[str, Field(description="要用來訓練的資料集名稱")],
    variant: Annotated[str, Field(description="YOLO 權重檔名，例如 yolov9m.pt")] = "yolov9m.pt",
    epochs: Annotated[int, Field(description="訓練輪數")] = 50,
    imgsz: Annotated[int, Field(description="輸入影像尺寸")] = 640,
    batch: Annotated[int, Field(description="batch size")] = 16,
    model_name: Annotated[str, Field(description="輸出的實驗名稱，會存在 runs/train/<model_name> 底下")] = "exp",
) -> dict:
    """完整的模型訓練流程（Skill，不是單純的 Tool）：依序跑過 Dataset Gate（資料集
    是否存在）、Health Gate（標記覆蓋率是否 >= 70%；類別不平衡只附警告、不擋訓練），
    兩個 Gate 都過了才真的呼叫 train_yolo_model 開始訓練。這些 Gate 判斷是寫死的
    規則，不是 LLM 決定，確保同一個資料集每次呼叫的行為都一致。

    若 Dataset Gate 沒過，回傳 status="dataset_not_found"；
    若 Health Gate 沒過（標記覆蓋率不足），回傳 status="health_gate_failed" 並附上
    健康度報告。類別分佈不平衡不會擋訓練，只會在回傳結果裡多帶一個 balance_warning。"""
    try:
        dataset_name = _sanitize_name(dataset_name)
        model_name = _sanitize_name(model_name)
    except ValueError as e:
        return {"status": "error", "message": str(e)}

    ds_path = DATASET_DIR / dataset_name
    if _count_images(ds_path) == 0:
        return {
            "status": "dataset_not_found",
            "message": f"Dataset Gate 沒過：本機找不到資料集 {dataset_name}（沒有圖片），"
                       f"請呼叫 download_roboflow_dataset 補齊資料後再試",
        }

    health = check_dataset_health.invoke({"dataset_name": dataset_name})
    if health.get("status") != "success":
        return {"status": "error", "message": f"Health Gate 檢查失敗：{health.get('message')}"}

    coverage = health.get("coverage") or 0
    balance = health.get("balance") or 0
    # 類別不平衡只警告、不擋訓練：樣本數過少的類別訓練後可能幾乎偵測不到，但這是
    # 「要不要補資料」的判斷題，交給使用者決定，不該讓自動 Gate 直接擋死整個訓練。
    balance_warning = None
    if balance < BALANCE_WARN_THRESHOLD:
        balance_warning = (
            f"類別分佈不平衡（平衡度 {balance} 分）：樣本數過少的類別訓練後可能幾乎偵測"
            f"不到，建議補齊這些類別的圖片、或減少類別數。這次仍照常訓練。"
        )

    state = _read_state()
    state.setdefault("dataset", {})
    state["dataset"].update({
        "imported": True,
        "name": dataset_name,
        "path": str(ds_path),
        "image_count": health.get("total_images"),
        "label_count": health.get("labeled_images"),
        "health_analysis": {
            "passed": coverage >= COVERAGE_GATE,
            "score": health["score"],
            "coverage": coverage,
            "balance": balance,
            "balance_warning": balance_warning,
            "last_checked": _now_iso(),
        },
    })
    _write_state(state)

    # 只在「標記覆蓋率不足」時擋下來——有圖片還沒標記，硬訓會學到錯的東西。
    if coverage < COVERAGE_GATE:
        return {
            "status": "health_gate_failed",
            "score": health["score"],
            "coverage": coverage,
            "message": f"Health Gate 沒過：標記覆蓋率只有 {coverage}%（< {COVERAGE_GATE}% 門檻），"
                       f"代表還有圖片沒標記，請補齊標記後再重新訓練",
        }

    train_result = train_yolo_model.invoke({
        "dataset_name": dataset_name,
        "variant": variant,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "model_name": model_name,
    })

    if train_result.get("status") == "completed":
        state = _read_state()
        state.setdefault("model", {})
        state["model"].update({
            "trained": True,
            "exp_name": model_name,
            "best_model_path": str(PROJECT_ROOT / "runs" / "train" / model_name / "weights" / "best.pt"),
        })
        _write_state(state)

    result = {"gates_passed": ["dataset_exists", "label_coverage"], **train_result}
    if balance_warning:
        result["balance_warning"] = balance_warning
    return result


@tool
def skill_infer(
    image_path: Annotated[str, Field(description="要偵測的圖片絕對路徑")],
    model_path: Annotated[Optional[str], Field(description="模型權重路徑，留空則自動選擇")] = None,
    conf: Annotated[float, Field(description="信心值門檻")] = 0.25,
    iou: Annotated[float, Field(description="IoU 門檻")] = 0.45,
) -> dict:
    """圖片推論流程（Skill）：先跑 Model Selection Gate ——有指定 model_path 就用；
    沒有就讀 workspace_state.json 裡最新訓練出來的 best_model_path；連那個都沒有，
    fallback 用專案內建的預設權重 models/yolov9m.pt。這個選擇邏輯是寫死的規則，
    不透過 LLM 猜。接著檢查圖片是否存在，最後才呼叫 run_yolo_inference。"""
    if not model_path:
        state = _read_state()
        best_model_path = state.get("model", {}).get("best_model_path")
        if best_model_path and Path(best_model_path).exists():
            model_path = best_model_path
        else:
            model_path = str(DEFAULT_MODEL_PATH)

    if not Path(image_path).exists():
        return {"status": "error", "message": f"圖片不存在：{image_path}"}

    result = run_yolo_inference.invoke({
        "image_path": image_path,
        "model_path": model_path,
        "conf": conf,
        "iou": iou,
    })
    return {"status": "success", "model_used": model_path, **result}


TRAIN_RUNS_DIR = PROJECT_ROOT / "runs" / "train"
# 模型效能診斷頁用的三張圖。檔名跟 ultralytics 版本有關（新版的曲線圖多了 Box 前綴），
# 所以每一張都列出可能的檔名，取實際存在的那個。
EVAL_CHART_CANDIDATES = [
    ("混淆矩陣 (Confusion Matrix)", ["confusion_matrix.png", "confusion_matrix_normalized.png"]),
    ("F1-Score 曲線", ["BoxF1_curve.png", "F1_curve.png"]),
    ("P-R 曲線 (Precision-Recall)", ["BoxPR_curve.png", "PR_curve.png"]),
]


def _latest_trained_model() -> Optional[str]:
    """回傳最近一次「真的訓練過」的實驗名稱（以 results.csv 的寫入時間判斷）。

    訓練途中每個 epoch 都會寫 results.csv，所以它的 mtime 比資料夾 mtime 更貼近
    最後一次訓練的時間；沒有 results.csv 的資料夾（訓練一開始就被終止）不算。
    """
    if not TRAIN_RUNS_DIR.exists():
        return None
    runs = [d for d in TRAIN_RUNS_DIR.iterdir() if d.is_dir() and (d / "results.csv").exists()]
    if not runs:
        return None
    return max(runs, key=lambda d: (d / "results.csv").stat().st_mtime).name


def _eval_charts(model_name: str) -> list:
    """回傳這個實驗實際存在的診斷圖表，附上前端可直接載入的 URL。"""
    from urllib.parse import quote

    run_dir = TRAIN_RUNS_DIR / model_name
    charts = []
    for label, filenames in EVAL_CHART_CANDIDATES:
        for fname in filenames:
            if (run_dir / fname).exists():
                charts.append({
                    "name": label,
                    "url": f"/api/model/evaluation/charts/{fname}?model={quote(model_name)}",
                })
                break
    return charts


@tool
def skill_eval(
    model_name: Annotated[str, Field(description="要評估的實驗名稱，對應 runs/train/<model_name>；留空或填 latest 就自動用最近一次訓練完成的模型")] = "",
) -> dict:
    """模型評估流程（Skill）：呼叫 evaluate_yolo_model 拿到 mAP/Loss 摘要後，
    再跑一個確定性的 Evaluation Gate —— mAP50 >= 0.5 才算通過，並把結果寫回
    workspace_state.json，讓下一次 skill_infer/skill_train 可以讀到最新狀態。

    使用者說「剛剛訓練完的模型」「最新的模型」時不用自己猜名稱：model_name 留空
    （或填 latest）就會自動解析成最近一次有訓練紀錄的實驗。回傳值一律包含
    resolved_model_name 與 charts（混淆矩陣／F1／P-R 曲線），評估這個動作本身
    不需要使用者提供任何圖片。"""
    requested = (model_name or "").strip()
    if requested.lower() in ("", "latest", "last", "newest"):
        resolved = _latest_trained_model()
        if not resolved:
            return {
                "status": "no_data",
                "message": "runs/train 底下沒有任何含 results.csv 的訓練紀錄，目前沒有模型可以評估。",
            }
    else:
        resolved = requested
        if not (TRAIN_RUNS_DIR / resolved / "results.csv").exists():
            fallback = _latest_trained_model()
            return {
                "status": "no_data",
                "message": (
                    f"找不到模型「{resolved}」的訓練紀錄（runs/train/{resolved}/results.csv 不存在）。"
                    + (f"最近一次有紀錄的訓練是「{fallback}」，要評估它請改用這個名稱。" if fallback else "")
                ),
            }

    model_name = resolved
    result = evaluate_yolo_model.invoke({"model_name": model_name})
    if result.get("status") != "success":
        return {**result, "resolved_model_name": model_name}

    passed = bool(result.get("final_map50", 0) >= EVAL_MAP50_GATE)
    state = _read_state()
    state.setdefault("model", {})
    state["model"].setdefault("evaluation", {})
    state["model"]["evaluation"].update({
        "passed": passed,
        "last_evaluated": _now_iso(),
    })
    state["model"]["last_training_metrics"] = {
        "mAP50": result.get("final_map50"),
        "final_loss": result.get("final_loss"),
        "total_epochs": result.get("total_epochs"),
    }
    _write_state(state)

    return {
        "gate_passed": passed,
        "gate_threshold": EVAL_MAP50_GATE,
        "resolved_model_name": model_name,
        "charts": _eval_charts(model_name),
        **result,
    }


SKILL_TOOLS = [skill_train, skill_infer, skill_eval]
