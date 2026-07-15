"""skills.py 的手動驗證腳本，跟 agent_tools_smoke_test.py 同套風格。

驗證 3 個 Skill 裡寫死的 Gate 邏輯（不靠 LLM 判斷）：
1. skill_train：Dataset Gate（資料集不存在就擋下）、Health Gate（健康度 < 70 就擋下）、
   兩個 Gate 都過才真的訓練。
2. skill_infer：Model Selection Gate（沒指定就 fallback 到預設權重）、圖片不存在就
   直接擋下，不會呼叫底層推論。
3. skill_eval：Evaluation Gate（mAP50 < 0.5 算沒通過）並回寫 workspace_state.json。

會暫時修改 workspace_state.json，測試前後都會備份還原，不影響專案原本的狀態。
完全不會呼叫 Gemini/OpenAI 等任何真的 LLM API，也不會真的連網路。

執行方式：python skills_smoke_test.py
"""
import shutil
from unittest.mock import patch, MagicMock

from agent_tools import DATASET_DIR
from skills import (
    SKILL_TOOLS,
    WORKSPACE_STATE_PATH,
    skill_train,
    skill_infer,
    skill_eval,
    _read_state,
)


def _make_fake_dataset(name: str):
    ds = DATASET_DIR / name
    (ds / "images").mkdir(parents=True, exist_ok=True)
    (ds / "labels").mkdir(parents=True, exist_ok=True)
    (ds / "images" / "a.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (ds / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")


def _backup_state():
    if WORKSPACE_STATE_PATH.exists():
        return WORKSPACE_STATE_PATH.read_bytes()
    return None


def _restore_state(backup):
    if backup is not None:
        WORKSPACE_STATE_PATH.write_bytes(backup)
    elif WORKSPACE_STATE_PATH.exists():
        WORKSPACE_STATE_PATH.unlink()


def case_skill_shape():
    for t in SKILL_TOOLS:
        assert t.name, f"{t} 缺少 name"
        assert t.description, f"{t.name} 缺少 description"
    print(f"[OK] {len(SKILL_TOOLS)} 個 Skill 都有 name/description，符合 function-calling 需要的格式")


def case_skill_train_dataset_gate():
    result = skill_train.invoke({"dataset_name": "__skill_smoke_missing__"})
    assert result["status"] == "dataset_not_found"
    print("[OK] skill_train 的 Dataset Gate 在資料集不存在時正確擋下：", result["message"])


def case_skill_train_health_gate():
    backup = _backup_state()
    _make_fake_dataset("__skill_smoke_lowhealth__")
    try:
        fake_health = MagicMock()
        fake_health.invoke.return_value = {
            "status": "success", "score": 40.0, "total_images": 10, "labeled_images": 2,
            "coverage": 20.0, "balance": 100.0,
        }
        with patch("skills.check_dataset_health", fake_health):
            result = skill_train.invoke({"dataset_name": "__skill_smoke_lowhealth__"})
        assert result["status"] == "health_gate_failed"
        assert result["score"] == 40.0
        state = _read_state()
        assert state["dataset"]["health_analysis"]["passed"] is False
        print("[OK] skill_train 的 Health Gate 在健康度 40 分（< 70）時正確擋下，並寫回 workspace_state.json")
    finally:
        shutil.rmtree(DATASET_DIR / "__skill_smoke_lowhealth__", ignore_errors=True)
        _restore_state(backup)


def case_skill_train_success():
    backup = _backup_state()
    _make_fake_dataset("__skill_smoke_train__")
    try:
        fake_health = MagicMock()
        fake_health.invoke.return_value = {
            "status": "success", "score": 90.0, "total_images": 10, "labeled_images": 10,
            "coverage": 100.0, "balance": 100.0,
        }
        fake_train = MagicMock()
        fake_train.invoke.return_value = {
            "status": "completed", "message": "done", "model_name": "smoke_exp", "epochs_run": 1,
        }
        with patch("skills.check_dataset_health", fake_health), patch("skills.train_yolo_model", fake_train):
            result = skill_train.invoke({"dataset_name": "__skill_smoke_train__", "model_name": "smoke_exp"})
        assert fake_train.invoke.called, "兩個 Gate 都過了應該真的呼叫 train_yolo_model"
        assert result["status"] == "completed"
        state = _read_state()
        assert state["model"]["trained"] is True
        assert state["model"]["exp_name"] == "smoke_exp"
        print("[OK] skill_train 兩個 Gate 都過關後才呼叫訓練，並更新 workspace_state.json 的 model 狀態")
    finally:
        shutil.rmtree(DATASET_DIR / "__skill_smoke_train__", ignore_errors=True)
        _restore_state(backup)


def case_skill_infer_missing_image():
    fake_infer = MagicMock()
    with patch("skills.run_yolo_inference", fake_infer):
        result = skill_infer.invoke({"image_path": "__no_such_image__.jpg"})
    assert result["status"] == "error"
    assert not fake_infer.invoke.called, "圖片不存在時，Gate 應該直接擋下，不該呼叫底層推論"
    print("[OK] skill_infer 在圖片不存在時正確擋下，沒有多花一次推論呼叫")


def case_skill_infer_model_selection_fallback():
    backup = _backup_state()
    _restore_state(None)  # 清空狀態，確保 model.best_model_path 不存在，逼 Gate 走 fallback
    fake_image = DATASET_DIR / "__skill_smoke_infer_img__.jpg"
    fake_image.parent.mkdir(parents=True, exist_ok=True)
    fake_image.write_bytes(b"\xff\xd8\xff\xd9")
    try:
        fake_infer = MagicMock()
        fake_infer.invoke.return_value = {"objects": [], "annotated_image_path": "out.jpg"}
        with patch("skills.run_yolo_inference", fake_infer):
            result = skill_infer.invoke({"image_path": str(fake_image)})
        assert result["status"] == "success"
        called_model_path = fake_infer.invoke.call_args[0][0]["model_path"]
        assert called_model_path.endswith("yolov9m.pt"), f"沒有 best_model_path 時應該 fallback 到預設權重，實際用了 {called_model_path}"
        print("[OK] skill_infer 的 Model Selection Gate 在沒有已訓練模型時，正確 fallback 到預設權重")
    finally:
        fake_image.unlink(missing_ok=True)
        _restore_state(backup)


def case_skill_eval_gate():
    backup = _backup_state()
    try:
        fake_eval = MagicMock()
        fake_eval.invoke.return_value = {
            "status": "success", "final_map50": 0.3, "final_loss": 1.2, "total_epochs": 20,
        }
        with patch("skills.evaluate_yolo_model", fake_eval):
            result = skill_eval.invoke({"model_name": "__skill_smoke_eval__"})
        assert result["gate_passed"] is False, "mAP50=0.3 < 0.5 門檻，應該判定沒通過"
        state = _read_state()
        assert state["model"]["evaluation"]["passed"] is False
        print("[OK] skill_eval 的 Evaluation Gate 在 mAP50 0.3（< 0.5）時正確判定沒通過，並寫回 workspace_state.json")
    finally:
        _restore_state(backup)


def run():
    case_skill_shape()
    case_skill_train_dataset_gate()
    case_skill_train_health_gate()
    case_skill_train_success()
    case_skill_infer_missing_image()
    case_skill_infer_model_selection_fallback()
    case_skill_eval_gate()
    print("\n全部驗證通過。")


if __name__ == "__main__":
    run()
