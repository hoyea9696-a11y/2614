"""把既有的 YOLOTrainer / YOLOInferencer / AIAnalyzer 包成 LangChain Tool。

Step 2（本檔案的範圍）：只做「整形」——讓這些既有能力具備
Annotated + docstring 的 Tool 格式，未來 Manager Agent 才能用
function-calling 呼叫它們。不修改 trainer.py / inferencer.py /
ai_analyzer.py 的既有邏輯，也不涉及任何 LLM 呼叫。

Manager Agent 本身（何時呼叫、要不要呼叫）留給 Step 3 的 PR。
"""
import sys
from pathlib import Path
from typing import Annotated, Optional

import cv2
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


@tool
def check_dataset_health(
    dataset_name: Annotated[str, Field(description="資料集名稱，對應 data/datasets/ 底下的資料夾名")],
) -> dict:
    """檢查資料集的健康度（標記覆蓋率、類別平衡、圖片與標籤總數），回傳 0-100 的健康分數。"""
    return ai_analyzer.analyze_dataset_health(dataset_name)


@tool
def train_yolo_model(
    dataset_name: Annotated[str, Field(description="要用來訓練的資料集名稱")],
    variant: Annotated[str, Field(description="YOLO 權重檔名，例如 yolov9m.pt")] = "yolov9m.pt",
    epochs: Annotated[int, Field(description="訓練輪數")] = 50,
    imgsz: Annotated[int, Field(description="輸入影像尺寸")] = 640,
    batch: Annotated[int, Field(description="batch size")] = 16,
    model_name: Annotated[str, Field(description="輸出的實驗名稱，會存在 runs/train/<model_name> 底下")] = "exp",
) -> dict:
    """使用指定資料集訓練 YOLO 模型，回傳最終狀態（completed/error/stopped）。"""
    trainer = YOLOTrainer(data_root=str(DATASET_DIR))
    ds_path = DATASET_DIR / dataset_name
    data_yaml = trainer.prepare_dataset(str(ds_path))
    trainer.train(
        variant=variant,
        data_yaml=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        model_name=model_name,
    )
    return {
        "status": trainer.progress.get("status"),
        "message": trainer.progress.get("message"),
        "model_name": model_name,
        "epochs_run": trainer.progress.get("current_epoch"),
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


ALL_TOOLS = [check_dataset_health, train_yolo_model, evaluate_yolo_model, run_yolo_inference]
