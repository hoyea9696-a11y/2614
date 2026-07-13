"""training_workflow.py 的手動驗證腳本（不是 pytest，跟專案現有慣例一致，
例如 standalone_train.py 也是直接跑的腳本）。

目的：確認 Graph 的節點/邊接線正確（Dataset Gate、Health Check Gate 的
skip / execute 分支都會被正確走到），不會真的觸發 YOLO 訓練——
trainer.train() 在這裡是被 mock 掉的，避免跑真的訓練任務。

執行方式：python training_workflow_smoke_test.py
"""
import shutil
from unittest.mock import patch

from training_workflow import DATASET_DIR, build_training_workflow


def _make_fake_dataset(name: str):
    ds = DATASET_DIR / name
    (ds / "images").mkdir(parents=True, exist_ok=True)
    (ds / "labels").mkdir(parents=True, exist_ok=True)
    (ds / "images" / "a.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (ds / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")


def case_missing_dataset(graph):
    result = graph.invoke({"dataset_name": "__smoke_test_missing__"})
    assert result.get("error"), "缺少資料集時應該回報錯誤，但沒有"
    print("[OK] 缺少資料集會被 Dataset Gate 擋下：", result["error"])


def case_fresh_health_check(graph):
    _make_fake_dataset("__smoke_test_ds__")
    try:
        with patch("training_workflow.YOLOTrainer.train", return_value=None) as mock_train, \
             patch("training_workflow.YOLOTrainer.prepare_dataset", return_value="fake.yaml") as mock_prepare, \
             patch("training_workflow.ai_analyzer.get_latest_results", return_value=None):
            result = graph.invoke({
                "dataset_name": "__smoke_test_ds__",
                "force_health_check": True,
            })
        assert not result.get("error"), f"不該有錯誤: {result.get('error')}"
        assert mock_prepare.called and mock_train.called, "應該呼叫到 trainer 的 prepare_dataset/train"
        print(
            "[OK] 完整流程跑通（train 已 mock，未真的訓練）：",
            result.get("training_status"),
            result.get("health_report", {}).get("status"),
        )
    finally:
        shutil.rmtree(DATASET_DIR / "__smoke_test_ds__", ignore_errors=True)


def case_skip_health_check(graph):
    _make_fake_dataset("__smoke_test_ds2__")
    try:
        with patch(
            "training_workflow._read_workspace_state",
            return_value={"dataset": {"health_analysis": {"score": 88.0, "passed": True}}},
        ), patch("training_workflow.YOLOTrainer.train", return_value=None), \
           patch("training_workflow.YOLOTrainer.prepare_dataset", return_value="fake.yaml"), \
           patch("training_workflow.ai_analyzer.analyze_dataset_health") as mock_health, \
           patch("training_workflow.ai_analyzer.get_latest_results", return_value=None):
            result = graph.invoke({"dataset_name": "__smoke_test_ds2__"})
        assert not mock_health.called, "有既有健康報告時，不該重新呼叫 analyze_dataset_health"
        assert result.get("health_report", {}).get("score") == 88.0
        print("[OK] 已有健康報告時會走 skip_health_check，不重新分析")
    finally:
        shutil.rmtree(DATASET_DIR / "__smoke_test_ds2__", ignore_errors=True)


def run():
    graph = build_training_workflow()
    case_missing_dataset(graph)
    case_fresh_health_check(graph)
    case_skip_health_check(graph)
    print("\n全部驗證通過。")


if __name__ == "__main__":
    run()
