"""LangGraph 版本的 Training Workflow 骨架。

Step 1（本檔案的範圍）：只把流程用 LangGraph 的 Graph/Gate 串起來，
不涉及任何 LLM 呼叫，也不修改 trainer.py / ai_analyzer.py 的既有邏輯，
只呼叫它們。Manager Agent / Tool 包裝留給後續的 PR。

流程對應 spec 文件的 Training Workflow：
    Dataset Gate -> (Skip / Import) -> Health Check Gate -> (Skip / Analysis)
    -> Training -> Evaluation
"""
import json
import sys
from pathlib import Path
from typing import Any, Optional, TypedDict

SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from trainer import YOLOTrainer
from ai_analyzer import ai_analyzer

from langgraph.graph import StateGraph, START, END

DATASET_DIR = PROJECT_ROOT / "data" / "datasets"
WORKSPACE_STATE_PATH = PROJECT_ROOT / "workspace_state.json"
IMAGE_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]


class TrainingState(TypedDict, total=False):
    dataset_name: str
    dataset_path: str
    dataset_exists: bool
    force_health_check: bool
    health_report: Optional[dict]
    train_params: dict
    training_status: Optional[str]
    evaluation_summary: Optional[Any]
    error: Optional[str]


def _dataset_has_images(ds_path: Path) -> bool:
    if not ds_path.exists():
        return False
    return any(ds_path.rglob(ext) for ext in IMAGE_EXTS)


def _read_workspace_state() -> dict:
    if WORKSPACE_STATE_PATH.exists():
        try:
            with open(WORKSPACE_STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# ---------- Nodes ----------

def dataset_gate(state: TrainingState) -> dict:
    ds_path = DATASET_DIR / state["dataset_name"]
    return {"dataset_path": str(ds_path), "dataset_exists": _dataset_has_images(ds_path)}


def route_dataset_gate(state: TrainingState) -> str:
    return "skip_import" if state["dataset_exists"] else "need_import"


def skip_import(state: TrainingState) -> dict:
    return {}


def need_import(state: TrainingState) -> dict:
    # 這個 Workflow 不會自動生成或下載資料集，Import 仍然是使用者透過既有
    # /api/datasets/upload_v2 上傳的手動步驟，這裡只負責回報「還沒做」。
    return {
        "error": (
            f"資料集 '{state['dataset_name']}' 不存在或沒有圖片，"
            "請先透過 /api/datasets/upload_v2 上傳資料，此 Workflow 不會自動下載資料集。"
        )
    }


def health_check_gate(state: TrainingState) -> dict:
    return {}


def route_health_check_gate(state: TrainingState) -> str:
    if state.get("error"):
        return "abort"
    if state.get("force_health_check"):
        return "run_health_check"
    ws_state = _read_workspace_state()
    score = ws_state.get("dataset", {}).get("health_analysis", {}).get("score")
    return "skip_health_check" if score is not None else "run_health_check"


def run_health_check(state: TrainingState) -> dict:
    report = ai_analyzer.analyze_dataset_health(state["dataset_name"])
    return {"health_report": report}


def skip_health_check(state: TrainingState) -> dict:
    ws_state = _read_workspace_state()
    return {"health_report": ws_state.get("dataset", {}).get("health_analysis")}


def route_after_health(state: TrainingState) -> str:
    return "abort" if state.get("error") else "train"


def train(state: TrainingState) -> dict:
    params = state.get("train_params") or {}
    trainer = YOLOTrainer(data_root=str(DATASET_DIR))
    data_yaml = trainer.prepare_dataset(state["dataset_path"])
    trainer.train(data_yaml=data_yaml, **params)
    return {"training_status": trainer.progress.get("status")}


def evaluate(state: TrainingState) -> dict:
    exp_name = (state.get("train_params") or {}).get("model_name", "exp")
    df = ai_analyzer.get_latest_results(exp_name)
    return {"evaluation_summary": ai_analyzer.summarize_metrics(df)}


def abort(state: TrainingState) -> dict:
    return {}


def build_training_workflow():
    graph = StateGraph(TrainingState)

    graph.add_node("dataset_gate", dataset_gate)
    graph.add_node("skip_import", skip_import)
    graph.add_node("need_import", need_import)
    graph.add_node("health_check_gate", health_check_gate)
    graph.add_node("run_health_check", run_health_check)
    graph.add_node("skip_health_check", skip_health_check)
    graph.add_node("train", train)
    graph.add_node("evaluate", evaluate)
    graph.add_node("abort", abort)

    graph.add_edge(START, "dataset_gate")
    graph.add_conditional_edges(
        "dataset_gate",
        route_dataset_gate,
        {"skip_import": "skip_import", "need_import": "need_import"},
    )
    graph.add_edge("skip_import", "health_check_gate")
    graph.add_edge("need_import", "health_check_gate")

    graph.add_conditional_edges(
        "health_check_gate",
        route_health_check_gate,
        {
            "run_health_check": "run_health_check",
            "skip_health_check": "skip_health_check",
            "abort": "abort",
        },
    )
    graph.add_conditional_edges(
        "run_health_check", route_after_health, {"train": "train", "abort": "abort"}
    )
    graph.add_conditional_edges(
        "skip_health_check", route_after_health, {"train": "train", "abort": "abort"}
    )

    graph.add_edge("train", "evaluate")
    graph.add_edge("evaluate", END)
    graph.add_edge("abort", END)

    return graph.compile()
