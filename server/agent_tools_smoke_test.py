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

from agent_tools import (
    ALL_TOOLS,
    DATASET_DIR,
    check_dataset_health,
    train_yolo_model,
    evaluate_yolo_model,
    run_yolo_inference,
    download_roboflow_dataset,
    search_roboflow_universe,
    prepare_roboflow_augmentation,
    apply_roboflow_proposal,
    discard_roboflow_proposal,
)


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
    print(f"[OK] {len(ALL_TOOLS)} 個 Tool 都有 name/description/args schema，符合 function-calling 需要的格式")


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


def case_train_yolo_model_dataset_not_found():
    result = train_yolo_model.invoke({"dataset_name": "__no_such_dataset__"})
    assert result["status"] == "dataset_not_found"
    assert "download_roboflow_dataset" in result["message"]
    print("[OK] train_yolo_model 在資料集不存在時正確回報 dataset_not_found，並提示呼叫 download_roboflow_dataset")


def case_download_roboflow_dataset_not_found():
    mock_resp = MagicMock(status_code=404, ok=False)
    with patch("agent_tools._load_roboflow_key", return_value="fake_key"), \
         patch("agent_tools.requests.get", return_value=mock_resp):
        result = download_roboflow_dataset.invoke({"workspace": "nope", "project": "nope"})
    assert result["status"] == "not_found"
    print("[OK] download_roboflow_dataset 在猜錯 workspace/project 時正確回報 not_found：", result["message"])


def case_download_roboflow_dataset_success():
    import io
    import zipfile

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("train/images/a.jpg", b"\xff\xd8\xff\xd9")

    export_resp = MagicMock(status_code=200, ok=True)
    export_resp.json.return_value = {"export": {"link": "https://example.com/fake.zip"}}
    zip_resp = MagicMock(status_code=200, content=zip_buf.getvalue())
    zip_resp.raise_for_status.return_value = None

    try:
        with patch("agent_tools._load_roboflow_key", return_value="fake_key"), \
             patch("agent_tools.requests.get", side_effect=[export_resp, zip_resp]):
            result = download_roboflow_dataset.invoke({
                "workspace": "someone",
                "project": "some-project",
                "dataset_name": "__tool_smoke_download__",
            })
        assert result["status"] == "success"
        assert result["image_count"] == 1
        print("[OK] download_roboflow_dataset 有正確解壓 zip 並存到 data/datasets/：", result)
    finally:
        shutil.rmtree(DATASET_DIR / "__tool_smoke_download__", ignore_errors=True)


def case_search_roboflow_universe_uses_real_search_shape():
    response = MagicMock(status_code=200, ok=True)
    response.json.return_value = {
        "query": "bear object detection",
        "results": [{
            "name": "Bear",
            "url": "https://universe.roboflow.com/demo-workspace/bear-demo",
            "workspace": {"name": "Demo", "url": "demo-workspace"},
            "type": "object-detection",
            "classes": ["Bear"],
            "images": 120,
            "license": "CC BY 4.0",
            "stars": 2,
            "latestVersion": 3,
        }],
    }
    with patch("agent_tools._load_roboflow_key", return_value="fake_key"), \
         patch("agent_tools.requests.get", return_value=response) as mock_get:
        result = search_roboflow_universe.invoke({"query": "bear object detection"})
    assert result["status"] == "success"
    assert result["candidates"][0]["workspace"] == "demo-workspace"
    assert result["candidates"][0]["project"] == "bear-demo"
    assert result["candidates"][0]["version"] == 3
    assert mock_get.call_args.kwargs["params"]["q"] == "bear object detection"
    print("[OK] Roboflow Universe 搜尋會解析官方 Search API 的 workspace/project/version")


def case_prepare_then_confirmed_apply_keeps_formal_dataset_unchanged_until_apply():
    dataset_name = "__tool_smoke_proposal__"
    dataset_root = DATASET_DIR / dataset_name
    history_path = DATASET_DIR / "__tool_smoke_roboflow_history__.json"
    _make_fake_dataset(dataset_name)
    (dataset_root / "data.yaml").write_text(
        "names: ['Bear', 'Brown-bear']\n",
        encoding="utf-8",
    )
    candidate = {
        "name": "Bear Mix",
        "workspace": "demo-workspace",
        "project": "bear-mix",
        "version": 2,
        "url": "https://universe.roboflow.com/demo-workspace/bear-mix",
        "type": "object-detection",
        "classes": ["Bear", "Brown-bear", "Polar-bear"],
        "images": 4,
        "license": "CC BY 4.0",
        "stars": 1,
        "downloads": 10,
    }

    def fake_download(_workspace, _project, _version, destination):
        (destination / "train" / "images").mkdir(parents=True, exist_ok=True)
        (destination / "train" / "labels").mkdir(parents=True, exist_ok=True)
        (destination / "data.yaml").write_text(
            "names: ['Bear', 'Brown-bear', 'Polar-bear']\n",
            encoding="utf-8",
        )
        samples = [
            ("new_bear", b"new-bear", "0 0.5 0.5 0.2 0.2\n"),
            ("new_brown", b"new-brown", "1 0.5 0.5 0.2 0.2\n"),
            ("unmapped", b"polar", "2 0.5 0.5 0.2 0.2\n"),
            ("duplicate", b"\xff\xd8\xff\xd9", "0 0.5 0.5 0.2 0.2\n"),
        ]
        for stem, image_bytes, label in samples:
            (destination / "train" / "images" / f"{stem}.jpg").write_bytes(image_bytes)
            (destination / "train" / "labels" / f"{stem}.txt").write_text(label, encoding="utf-8")
        return {"status": "success"}

    proposal_id = None
    try:
        with patch(
            "agent_tools._search_roboflow_universe",
            return_value={"status": "success", "query": "bear", "candidates": [candidate]},
        ), patch("agent_tools._download_roboflow_export", side_effect=fake_download), \
             patch("agent_tools.ROBOFLOW_HISTORY_PATH", history_path):
            proposal = prepare_roboflow_augmentation.invoke({
                "dataset_name": dataset_name,
                "query": "bear",
            })
            proposal_id = proposal.get("proposal_id")
            assert proposal["status"] == "proposal_ready"
            assert proposal["usable_images"] == 2
            assert proposal["unmapped_images"] == 1
            assert proposal["duplicate_images"] == 1
            assert len(proposal["class_mapping"]) == 2
            assert len(list((dataset_root / "images").glob("*.jpg"))) == 1, (
                "評估完成但尚未確認時，正式資料集不能增加圖片"
            )

            applied = apply_roboflow_proposal.invoke({"proposal_id": proposal_id})
            assert applied["status"] == "success"
            assert applied["added_images"] == 2
            assert applied["total_images_now"] == 3
            labels = [
                path.read_text(encoding="utf-8").split()[0]
                for path in (dataset_root / "labels").glob("bear-mix_*.txt")
            ]
            assert sorted(labels) == ["0", "1"], "多類別資料必須依明確類別名稱映射，不能全部改成 class 0"
        print("[OK] Agent 先暫存評估；使用者確認後才合併，並安全保留多類別 mapping")
    finally:
        if proposal_id:
            discard_roboflow_proposal(proposal_id)
        shutil.rmtree(dataset_root, ignore_errors=True)
        if history_path.exists():
            history_path.unlink()


def case_train_yolo_model_blocks_path_traversal():
    # _sanitize_name 會把路徑元件全部拆掉，只留下最後一段檔名，
    # 所以帶 ../ 的 dataset_name 會被當成單純的 "__escaped_dataset__" 處理，
    # 而不是真的跳到 DATASET_DIR 外面 —— 這裡驗證的就是「不會跳出去」這件事。
    result = train_yolo_model.invoke({"dataset_name": "../../__escaped_dataset__"})
    assert result["status"] == "dataset_not_found"
    escaped = DATASET_DIR.parent / "__escaped_dataset__"
    assert not escaped.exists(), "dataset_name 帶 ../ 不應該讓路徑跳出 DATASET_DIR"
    print("[OK] train_yolo_model 的 dataset_name 帶 ../ 時，路徑被限制在 DATASET_DIR 內，沒有跳出去")


def case_download_roboflow_dataset_blocks_zip_slip():
    import io
    import zipfile
    from unittest.mock import patch, MagicMock

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("../../../__zip_slip_escaped__.txt", "pwned")

    export_resp = MagicMock(status_code=200, ok=True)
    export_resp.json.return_value = {"export": {"link": "https://example.com/evil.zip"}}
    zip_resp = MagicMock(status_code=200, content=zip_buf.getvalue())
    zip_resp.raise_for_status.return_value = None

    escaped_file = DATASET_DIR.parent.parent / "__zip_slip_escaped__.txt"
    try:
        with patch("agent_tools._load_roboflow_key", return_value="fake_key"), \
             patch("agent_tools.requests.get", side_effect=[export_resp, zip_resp]):
            result = download_roboflow_dataset.invoke({
                "workspace": "attacker",
                "project": "evil-zip",
                "dataset_name": "__zip_slip_smoke__",
            })
        assert result["status"] == "error"
        assert not escaped_file.exists(), "惡意 zip 內的 ../../ 路徑不應該被解壓到 DATASET_DIR 之外"
        print("[OK] download_roboflow_dataset 偵測到 zip 裡的 ../ 路徑並中止解壓（zip slip 防護）：", result["message"])
    finally:
        shutil.rmtree(DATASET_DIR / "__zip_slip_smoke__", ignore_errors=True)
        if escaped_file.exists():
            escaped_file.unlink()


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
    case_train_yolo_model_dataset_not_found()
    case_train_yolo_model_blocks_path_traversal()
    case_download_roboflow_dataset_not_found()
    case_download_roboflow_dataset_success()
    case_search_roboflow_universe_uses_real_search_shape()
    case_prepare_then_confirmed_apply_keeps_formal_dataset_unchanged_until_apply()
    case_download_roboflow_dataset_blocks_zip_slip()
    case_evaluate_yolo_model_no_data()
    case_run_yolo_inference_mocked()
    print("\n全部驗證通過。")


if __name__ == "__main__":
    run()
