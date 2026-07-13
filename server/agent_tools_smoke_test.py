"""agent_tools.py 的手動驗證腳本（跟 training_workflow_smoke_test.py 同套風格）。

驗證兩件事：
1. 每個 Tool 都有正確的 Tool 格式（name/description/args schema），
   代表未來可以被 Manager Agent 用 .bind_tools([...]) 呼叫。
2. Tool 呼叫真的會打到底層的 YOLOTrainer / YOLOInferencer / AIAnalyzer，
   但重量級呼叫（真的訓練、真的載入 YOLO 權重）全部用 mock 頂替，
   不會觸發真正的訓練或下載模型檔案。

執行方式：python agent_tools_smoke_test.py
"""
import shutil
from unittest.mock import patch, MagicMock

import numpy as np

from agent_tools import ALL_TOOLS, DATASET_DIR, check_dataset_health, train_yolo_model, evaluate_yolo_model, run_yolo_inference


def _make_fake_dataset(name: str):
    ds = DATASET_DIR / name
    (ds / "images").mkdir(parents=True, exist_ok=True)
    (ds / "labels").mkdir(parents=True, exist_ok=True)
    (ds / "images" / "a.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (ds / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")


def case_tool_shape():
    for t in ALL_TOOLS:
        assert t.name, f"{t} 缺少 name"
        assert t.description, f"{t.name} 缺少 description，Manager Agent 會不知道這個 Tool 是做什麼的"
        schema = t.get_input_schema().model_json_schema()
        assert "properties" in schema, f"{t.name} 沒有產生出參數 schema"
    print("[OK] 4 個 Tool 都有 name/description/args schema，符合 function-calling 需要的格式")


def case_check_dataset_health():
    _make_fake_dataset("__tool_smoke_ds__")
    try:
        result = check_dataset_health.invoke({"dataset_name": "__tool_smoke_ds__"})
        assert result["status"] == "success"
        print("[OK] check_dataset_health 呼叫到真的 AIAnalyzer.analyze_dataset_health：", result["score"])
    finally:
        shutil.rmtree(DATASET_DIR / "__tool_smoke_ds__", ignore_errors=True)


def case_train_yolo_model_mocked():
    _make_fake_dataset("__tool_smoke_train__")
    try:
        with patch("agent_tools.YOLOTrainer.prepare_dataset", return_value="fake.yaml") as mock_prepare, \
             patch("agent_tools.YOLOTrainer.train", return_value=None) as mock_train:
            result = train_yolo_model.invoke({
                "dataset_name": "__tool_smoke_train__",
                "epochs": 1,
                "model_name": "smoke_exp",
            })
        assert mock_prepare.called and mock_train.called, "應該呼叫到 trainer 的 prepare_dataset/train"
        print("[OK] train_yolo_model 有正確呼叫 trainer（train() 已 mock，沒有真的訓練）：", result)
    finally:
        shutil.rmtree(DATASET_DIR / "__tool_smoke_train__", ignore_errors=True)


def case_evaluate_yolo_model_no_data():
    result = evaluate_yolo_model.invoke({"model_name": "__no_such_experiment__"})
    assert result["status"] == "no_data"
    print("[OK] evaluate_yolo_model 在沒有訓練結果時正確回報 no_data：", result["message"])


def case_run_yolo_inference_mocked():
    fake_detections = [{"class_id": 0, "name": "object", "confidence": 0.9, "bbox": [0, 0, 10, 10]}]
    fake_img = np.zeros((10, 10, 3), dtype="uint8")

    mock_inferencer = MagicMock()
    mock_inferencer.predict.return_value = (fake_detections, fake_img)

    with patch("agent_tools.YOLOInferencer", return_value=mock_inferencer), \
         patch("agent_tools.cv2.imwrite", return_value=True) as mock_imwrite:
        result = run_yolo_inference.invoke({"image_path": "fake_input.jpg"})

    assert result["objects"] == fake_detections
    assert mock_imwrite.called, "應該有把標記後的圖片存檔"
    print("[OK] run_yolo_inference 有正確回傳偵測結果與存檔路徑（YOLO 本體已 mock，沒有真的載入權重）：", result)


def run():
    case_tool_shape()
    case_check_dataset_health()
    case_train_yolo_model_mocked()
    case_evaluate_yolo_model_no_data()
    case_run_yolo_inference_mocked()
    print("\n全部驗證通過。")


if __name__ == "__main__":
    run()
