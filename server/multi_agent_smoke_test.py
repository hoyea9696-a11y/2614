"""multi_agent.py 的手動驗證腳本，跟 manager_agent_smoke_test.py 同套風格。

用一個假的 ChatModel（FakeMultiAgentModel）依照預先寫好的腳本回覆，驗證：
1. Planner 在每次新請求時先產出一份 Plan，Manager（decision_engine）的路由決策是
   結構化的 Decision 物件，不是關鍵字文字。
2. Data Agent / Model Agent / Evaluation Agent 各自只會呼叫到自己那組 Tool/Skill，
   結束前都必須呼叫 report_to_manager 回報結構化結果。
3. Manager -> Data Agent -> Observer -> Manager -> Model Agent -> Observer -> Manager
   -> Evaluation Agent -> Observer -> Manager -> FINISH 這個完整的多輪協調流程接線是對的。
4. 已知的 Gate 失敗類型（reason_code 落在 GATE_ROUTING）Manager 直接用固定規則
   決定下一步，完全不呼叫 LLM；超過重試上限（RETRY_LIMITS）就停止重試。
5. Observer 連續偵測到「同一種非已知 Gate 的訊號、沒有進展」達到
   STALL_ROUNDS_BEFORE_REPLAN 次，會判定 plan_invalid，Manager 繞回 Planner
   重新規劃，而不是拿著同一份卡住的 Plan 一直問 LLM。
6. Model Agent 呼叫的是 Skill 層（skill_train），Skill 內部的 Gate 呼叫（check_dataset_health/
   train_yolo_model）不會出現在對話訊息裡——這是刻意的：Agent 只看到「一個 Skill」，
   不會看到 Skill 內部串了幾個 Tool。

完全不會呼叫 Gemini/OpenAI 等任何真的 LLM API。

注意：這個檔案在這次改動（加入 Planner/Observer）之後還沒有實際執行過，
案例 5（case_manager_replans_after_repeated_stall）涉及的輪數最多，
如果要真的驗證，先跑這個檔案並留意例外訊息。

執行方式：python multi_agent_smoke_test.py
"""
import shutil
import uuid
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from agent_tools import DATASET_DIR
from multi_agent import (
    build_multi_agent,
    _extract_text,
    _parse_tool_content,
    Decision,
    Plan,
    Step,
    TrainingHealthReply,
    TrainingParamsReply,
    TrainingProposal,
    _build_approved_training_tool,
    _build_post_augment_training_transition,
    RETRY_LIMITS,
    STALL_ROUNDS_BEFORE_REPLAN,
)
from skills import WORKSPACE_STATE_PATH


class FakeMultiAgentModel:
    """依照預先寫好的回覆腳本作答，不呼叫任何真的 LLM。

    Planner／Manager 的結構化輸出（with_structured_output）跟子 Agent 的一般
    ReAct 呼叫（bind_tools）共用同一個腳本佇列，依實際呼叫順序依序取用——這樣
    才能同時驗證「該問 LLM 的時候有問」跟「不該問的時候完全沒問」（腳本項目
    不夠就會直接因為 pop 到空 list 或型別不對而讓測試失敗）。

    bind_tools／with_structured_output 回傳的包裝物件不同：production code
    對 decision_model/planner_model 用 include_raw=True，取的是
    result["parsed"]/result["raw"]；對子 Agent 的 model_with_tools 則直接用
    invoke() 回傳值當 AIMessage。腳本裡放的都是裸的 Plan/Decision/AIMessage，
    由包裝物件依角色決定要不要包成 {"parsed", "raw"}，不用每個測試案例自己包。"""

    def __init__(self, script):
        self._script = list(script)

    def bind_tools(self, tools):
        return _FakeBoundModel(self._script)

    def with_structured_output(self, schema, method=None, include_raw=False):
        return _FakeStructuredModel(self._script, include_raw)


class _FakeBoundModel:
    def __init__(self, script):
        self._script = script

    def invoke(self, messages):
        return self._script.pop(0)


class _FakeStructuredModel:
    def __init__(self, script, include_raw):
        self._script = script
        self._include_raw = include_raw

    def invoke(self, messages):
        item = self._script.pop(0)
        if self._include_raw:
            return {"parsed": item, "raw": AIMessage(content="")}
        return item


def _make_fake_dataset(name: str):
    ds = DATASET_DIR / name
    (ds / "images").mkdir(parents=True, exist_ok=True)
    (ds / "labels").mkdir(parents=True, exist_ok=True)
    (ds / "images" / "a.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (ds / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")


def _make_unlabeled_dataset(name: str):
    """有圖片但完全沒有標記：coverage=0，會被 Health Gate 擋下的那種資料集。"""
    ds = DATASET_DIR / name
    (ds / "images").mkdir(parents=True, exist_ok=True)
    for i in range(3):
        (ds / "images" / f"a{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")


def _make_imbalanced_dataset(name: str):
    """標記完整但類別數量嚴重偏一邊：coverage=100、balance=20，健康分數不到門檻。

    9 張是類別 0、1 張是類別 1 —— 這是「該建議去 Roboflow 補資料」的情況，跟上面
    完全沒標記的情況要走不一樣的建議。
    """
    ds = DATASET_DIR / name
    (ds / "images").mkdir(parents=True, exist_ok=True)
    (ds / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(10):
        (ds / "images" / f"b{i}.jpg").write_bytes(b"\xff\xd8\xff\xd9")
        cls_id = 1 if i == 9 else 0
        (ds / "labels" / f"b{i}.txt").write_text(f"{cls_id} 0.5 0.5 0.2 0.2\n")


def _report_call(status: str, reason: str, suggestion: str = "", reason_code: str = "", call_id: str = "call"):
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "report_to_manager",
            "args": {"status": status, "reason": reason, "suggestion": suggestion, "reason_code": reason_code},
            "id": call_id,
        }],
    )


def _tool_call(name: str, args: dict, call_id: str = "call"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def _thread(name: str) -> str:
    """build_multi_agent 用真的 SqliteSaver（跟 production 共用同一個 data/agent_checkpoints.sqlite），
    固定字串當 thread_id 會讓這次執行讀到上一次跑測試留下的舊訊息，導致斷言誤判——每次跑都
    加一段隨機字尾，確保每個測試案例、每次執行都是全新的一份記憶。"""
    return f"{name}_{uuid.uuid4().hex[:8]}"


def _resume_param_question(graph, config, result):
    """回答 train_params_gate 問的訓練參數，讓流程往下走到確認 Gate。

    這個 Gate 刻意不用 interrupt()：它走 ASK_USER，也就是「這一輪直接結束、等使用者
    在聊天視窗回一句話」。所以測試不能用 Command(resume=...) 恢復，必須真的再送一則
    HumanMessage 進去，模擬使用者回「就用這些」。腳本裡要對應放一筆
    TrainingParamsReply，那是 Manager 解析這句回覆時會消耗的一格。
    """
    if not result.get("pending_param_question"):
        return result
    return graph.invoke({"messages": [HumanMessage(content="就用這些")]}, config=config)


def _is_training_interrupt(result) -> bool:
    """這次 interrupt 是不是「要不要開始訓練」的確認框。

    結束時還有另一個 interrupt 會問「要不要把模型存進模型庫」，那個不該由測試代答，
    所以要能分辨——只有訓練確認才自動按下同意。
    """
    interrupts = result.get("__interrupt__") or []
    if not interrupts:
        return False
    payload = getattr(interrupts[0], "value", None) or {}
    return payload.get("action_type") == "training" or "training_params" in payload


def _invoke_confirming_training(graph, initial_input, config, max_stops=12):
    """跑到底：訓練前有兩個停頓點，兩個都答掉才走得到 model_agent/skill_train。

    1. train_params_gate 用 ASK_USER 問參數 —— 補送一句「就用這些」（見
       _resume_param_question）。
    2. train_confirm_gate 用 interrupt() 等最後確認 —— 模擬使用者按下確認
       （approved=True），跟 stream_trace() 對 interrupt 的處理邏輯一樣。

    訓練失敗後 Manager 重試會再繞回這兩個 Gate 一次，所以這裡用迴圈處理到不再停下來
    為止，不是只答一次；max_stops 只是避免測試在接線出錯時無限迴圈的保險絲。
    """
    result = graph.invoke(initial_input, config=config)
    for _ in range(max_stops):
        if result.get("pending_param_question"):
            result = _resume_param_question(graph, config, result)
            continue
        if _is_training_interrupt(result):
            result = graph.invoke(Command(resume={"approved": True}), config=config)
            continue
        break
    return result


def case_extract_text_handles_gemini_content_blocks():
    # 真實跑過網站發現：Gemini 有時回傳的 AIMessage.content 不是純字串，而是一包
    # 帶 grounding/signature metadata 的內容區塊 list，直接當字串用會讓 reply
    # 欄位變成一包不一致的 JSON 結構。
    blocks = [{"type": "text", "text": "健康度不足，無法訓練。", "extras": {"signature": "abc123"}}]
    assert _extract_text(blocks) == "健康度不足，無法訓練。"
    assert _extract_text("純字串沒問題") == "純字串沒問題"
    print("[OK] _extract_text 能把 Gemini 回傳的內容區塊 list 正確轉成純文字，不會是原始 list")


def case_parse_tool_content_handles_json_string_and_dict():
    # report_to_manager 的 ToolMessage.content 依 LangChain 版本可能是 dict，
    # 也可能被序列化成 JSON 字串，兩種都要能還原成 dict，不能直接拋例外。
    assert _parse_tool_content({"status": "success"}) == {"status": "success"}
    assert _parse_tool_content('{"status": "success"}') == {"status": "success"}
    assert _parse_tool_content("不是 JSON") == {"raw": "不是 JSON"}
    print("[OK] _parse_tool_content 能同時處理 dict／JSON 字串／無法解析的內容")


def case_manager_routes_to_data_agent_then_finish():
    script = [
        Plan(goal="查看資料集健康度", steps=[Step(id="1", description="檢查資料集健康度", agent="data_agent")]),
        Decision(action="call_agent", target_agent="data_agent", rationale="使用者要看資料集健康度"),
        _tool_call("check_dataset_health", {"dataset_name": "__multi_smoke_ds__"}, "call_1"),
        _report_call("success", "資料集健康度 100 分，沒問題。", call_id="call_2"),
        Decision(action="finish", rationale="健康度已確認沒問題", final_reply="資料集健康度 100 分，沒問題。"),
    ]
    graph = build_multi_agent(FakeMultiAgentModel(script))

    _make_fake_dataset("__multi_smoke_ds__")
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content="幫我看一下資料集健康度")]},
            config={"configurable": {"thread_id": _thread("smoke_data_agent")}},
        )
    finally:
        shutil.rmtree(DATASET_DIR / "__multi_smoke_ds__", ignore_errors=True)

    messages = result["messages"]
    tool_names = [m.name for m in messages if isinstance(m, ToolMessage)]
    assert tool_names == ["check_dataset_health", "report_to_manager"], (
        f"Data Agent 應該真的呼叫到 check_dataset_health，結束前呼叫 report_to_manager：{tool_names}"
    )
    assert result["evaluation"]["verdict"] == "on_track"
    assert result["plan"]["steps"][0]["status"] == "done", "Observer 應該把對應的 Plan Step 標成 done"
    assert messages[-1].content == "資料集健康度 100 分，沒問題。"
    print("[OK] Planner 產出計畫，Manager 把資料集相關需求交給 Data Agent，"
          "Observer 判讀成功後 Manager 才決定 FINISH")


def case_manager_hands_off_data_agent_to_model_agent_to_evaluation_agent():
    script = [
        Plan(goal="訓練並評估模型", steps=[
            Step(id="1", description="確認/補齊資料", agent="data_agent"),
            Step(
                id="2",
                description="訓練模型",
                agent="model_agent",
                is_training=True,
                training_params=TrainingProposal(
                    dataset_name="__multi_smoke_train__",
                    epochs=1,
                    model_name="smoke_exp",
                ),
            ),
            Step(id="3", description="評估模型", agent="evaluation_agent"),
        ]),
        Decision(action="call_agent", target_agent="data_agent", rationale="需要先確認/補齊資料"),
        _tool_call("download_roboflow_dataset", {"workspace": "ws", "project": "proj", "dataset_name": "__multi_smoke_train__"}, "call_1"),
        _report_call("success", "已從 Roboflow 補齊資料，共 1 張圖片。", call_id="call_2"),
        Decision(action="call_agent", target_agent="model_agent", rationale="資料備妥，開始訓練"),
        # train_params_gate 先用對話問一次參數，使用者回「就用這些」的解析結果
        TrainingParamsReply(intent="use_suggested"),
        _tool_call("skill_train", {"dataset_name": "__multi_smoke_train__", "epochs": 1, "model_name": "smoke_exp"}, "call_3"),
        _report_call("success", "模型訓練完成，接下來幫你評估效果。", call_id="call_4"),
        Decision(action="call_agent", target_agent="evaluation_agent", rationale="訓練完成，評估效果"),
        _tool_call("skill_eval", {"model_name": "smoke_exp"}, "call_5"),
        _report_call("success", "評估完成，mAP50 已達標。", call_id="call_6"),
        Decision(action="finish", rationale="評估已通過門檻", final_reply="評估完成，mAP50 已達標，模型可以使用。"),
    ]
    graph = build_multi_agent(FakeMultiAgentModel(script))

    import io
    import zipfile
    from unittest.mock import patch, MagicMock

    # zip 裡同時放圖片跟對應的標籤，確保 skill_train 的 Health Gate（>=70 分）會過關
    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        zf.writestr("train/images/a.jpg", b"\xff\xd8\xff\xd9")
        zf.writestr("train/labels/a.txt", "0 0.5 0.5 0.2 0.2\n")
    export_resp = MagicMock(status_code=200, ok=True)
    export_resp.json.return_value = {"export": {"link": "https://example.com/fake.zip"}}
    zip_resp = MagicMock(status_code=200, content=zip_buf.getvalue())
    zip_resp.raise_for_status.return_value = None

    fake_eval = MagicMock()
    fake_eval.invoke.return_value = {
        "status": "success", "final_map50": 0.6, "final_loss": 1.0, "total_epochs": 1,
    }

    state_backup = WORKSPACE_STATE_PATH.read_bytes() if WORKSPACE_STATE_PATH.exists() else None
    try:
        with patch("agent_tools._load_roboflow_key", return_value="fake_key"), \
             patch("agent_tools.requests.get", side_effect=[export_resp, zip_resp]), \
             patch("agent_tools.YOLOTrainer.prepare_dataset", return_value="fake.yaml"), \
             patch("agent_tools.YOLOTrainer.train", return_value=None), \
             patch("skills.evaluate_yolo_model", fake_eval):
            result = _invoke_confirming_training(
                graph,
                {"messages": [HumanMessage(content="幫我用 __multi_smoke_train__ 訓練並評估一個模型")]},
                config={"configurable": {"thread_id": _thread("smoke_full_pipeline")}},
            )
    finally:
        shutil.rmtree(DATASET_DIR / "__multi_smoke_train__", ignore_errors=True)
        if state_backup is not None:
            WORKSPACE_STATE_PATH.write_bytes(state_backup)
        elif WORKSPACE_STATE_PATH.exists():
            WORKSPACE_STATE_PATH.unlink()

    messages = result["messages"]
    tool_names = [m.name for m in messages if isinstance(m, ToolMessage)]
    domain_calls = [n for n in tool_names if n != "report_to_manager"]
    assert domain_calls == ["download_roboflow_dataset", "skill_train", "skill_eval"], (
        f"預期只看到 3 個 Agent 層級的 Skill/Tool 呼叫（Skill 內部的 check_dataset_health/"
        f"train_yolo_model 不該出現在對話訊息裡），實際看到：{domain_calls}"
    )
    assert tool_names.count("report_to_manager") == 3, "三個子 Agent 都應該各自呼叫一次 report_to_manager 回報"
    assert [s["status"] for s in result["plan"]["steps"]] == ["done", "done", "done"], (
        "三個 Step 都成功，Observer 應該依序把它們標成 done"
    )
    assert messages[-1].content == "評估完成，mAP50 已達標，模型可以使用。"
    print("[OK] Planner 先產出三步驟計畫，Manager 依序協調 Data Agent -> Model Agent -> "
          "Evaluation Agent，Observer 逐步把 Plan Step 標成 done")


def case_training_confirmation_contains_exact_params_and_operation_id():
    proposal = TrainingProposal(
        dataset_name="__multi_smoke_confirm__",
        variant="yolov9s.pt",
        epochs=3,
        imgsz=320,
        batch=2,
        model_name="confirm_exp",
    )
    graph = build_multi_agent(FakeMultiAgentModel([
        Plan(goal="確認後訓練", steps=[Step(
            id="1",
            description="使用指定低資源參數訓練",
            agent="model_agent",
            is_training=True,
            training_params=proposal,
        )]),
        Decision(action="call_agent", target_agent="model_agent", rationale="參數已完整，等待確認"),
        TrainingParamsReply(intent="use_suggested"),
    ]))

    _make_fake_dataset("__multi_smoke_confirm__")
    config = {"configurable": {"thread_id": _thread("smoke_exact_confirmation")}}
    try:
        first = graph.invoke(
            {"messages": [HumanMessage(content="用指定參數訓練")]},
            config=config,
        )
        # 確認框不是第一個停頓點：先問參數（ASK_USER），回答完才進 train_confirm_gate。
        assert first.get("pending_param_question"), "訓練前應該先用對話問一次參數"
        result = _resume_param_question(graph, config, first)
    finally:
        shutil.rmtree(DATASET_DIR / "__multi_smoke_confirm__", ignore_errors=True)

    assert "__interrupt__" in result, "具體訓練方案準備完成後應停在確認 Gate"
    pending = result["__interrupt__"][0]
    payload = pending.value
    assert pending.id and payload["operation_id"].startswith("train-")
    assert payload["training_params"] == proposal.model_dump()
    snapshot = graph.get_state(config)
    assert snapshot.values["pending_operation"]["operation_id"] == payload["operation_id"]
    print("[OK] 訓練確認包含唯一 operation_id 與完整參數，且已先寫入 checkpoint")


def case_training_tool_rejects_unapproved_params():
    params = TrainingProposal(dataset_name="bear", epochs=3, batch=2).model_dump()
    guarded_tool = _build_approved_training_tool(params, "train-guard-test")
    with patch(
        "multi_agent.skill_train.func",
        return_value={"status": "completed", "model_name": "exp"},
    ) as train_mock:
        rejected = guarded_tool.invoke({**params, "epochs": 99})
        assert rejected["status"] == "approval_mismatch"
        train_mock.assert_not_called()

        accepted = guarded_tool.invoke(params)
        assert accepted["status"] == "completed"
        train_mock.assert_called_once()
    print("[OK] Model Agent 改動未核准參數時，skill_train 會在啟動前阻止執行")


def _training_step(dataset_name: str, model_name: str):
    return Step(
        id="1",
        description="訓練模型",
        agent="model_agent",
        is_training=True,
        training_params=TrainingProposal(dataset_name=dataset_name, epochs=1, model_name=model_name),
    )


def case_unhealthy_dataset_asks_before_showing_params():
    # 健康檢查沒過時只講健康狀況，不要同時把參數列出來——那等於一邊說資料集有問題、
    # 一邊把訓練流程推到底。參數要等使用者明確表示「還是要訓練」的下一句話才出現。
    graph = build_multi_agent(FakeMultiAgentModel([
        Plan(goal="訓練模型", steps=[_training_step("__multi_smoke_unlabeled__", "smoke_unhealthy")]),
        Decision(action="call_agent", target_agent="model_agent", rationale="使用者要訓練"),
        TrainingHealthReply(intent="train_anyway"),
    ]))

    _make_unlabeled_dataset("__multi_smoke_unlabeled__")
    config = {"configurable": {"thread_id": _thread("smoke_unhealthy_gate")}}
    try:
        first = graph.invoke(
            {"messages": [HumanMessage(content="用 __multi_smoke_unlabeled__ 訓練")]},
            config=config,
        )
        warning = first["messages"][-1].content
        assert first.get("pending_health_question"), "健康沒過應該停下來等使用者決定"
        assert not first.get("pending_param_question"), "健康沒過時不該同時丟出參數問題"
        assert "沒有通過健康檢查" in warning
        assert "標記覆蓋率：0%（0/3 張）" in warning, f"應該如實顯示覆蓋率，實際：{warning}"
        assert "epochs" not in warning and "yolov9s.pt" not in warning, (
            f"上半段不該出現任何訓練參數，實際：{warning}"
        )
        assert "去 Roboflow 補資料也解決不了" in warning, "圖片沒標記的情況不該建議去補資料"

        second = graph.invoke({"messages": [HumanMessage(content="還是要訓練")]}, config=config)
        params_msg = second["messages"][-1].content
        assert second.get("pending_param_question"), "使用者堅持後才進到問參數"
        assert "epochs" in params_msg and "yolov9s.pt" in params_msg, (
            f"下半段才顯示參數，實際：{params_msg}"
        )
        assert second["pending_operation"]["health_ack"] is True, (
            "已經知情的 operation 要標記起來，重試時不該再問一次健康度"
        )
    finally:
        shutil.rmtree(DATASET_DIR / "__multi_smoke_unlabeled__", ignore_errors=True)
    print("[OK] 資料集健康沒過時只先講健康狀況，使用者堅持要訓練才顯示參數")


def case_labeled_but_imbalanced_suggests_roboflow():
    # 標記是完整的、問題只在類別不平衡：這種缺圖片的情況去 Roboflow 補資料才有解，
    # 建議內容要跟「圖片根本沒標記」那種情況分開。
    graph = build_multi_agent(FakeMultiAgentModel([
        Plan(goal="訓練模型", steps=[_training_step("__multi_smoke_imbalanced__", "smoke_imbalanced")]),
        Decision(action="call_agent", target_agent="model_agent", rationale="使用者要訓練"),
    ]))

    _make_imbalanced_dataset("__multi_smoke_imbalanced__")
    config = {"configurable": {"thread_id": _thread("smoke_imbalanced_gate")}}
    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content="用 __multi_smoke_imbalanced__ 訓練")]},
            config=config,
        )
    finally:
        shutil.rmtree(DATASET_DIR / "__multi_smoke_imbalanced__", ignore_errors=True)

    warning = result["messages"][-1].content
    health = result["pending_operation"]["health"]
    assert health["coverage"] == 100.0 and health["balance"] < 40, (
        f"這份資料集應該是「標記完整但不平衡」，實際：{health}"
    )
    assert health["blocks_training"] is False, "標記完整就不該被 Health Gate 擋下"
    assert "Roboflow Universe" in warning, f"這種情況要建議去補資料，實際：{warning}"
    assert "去補資料" in warning and "還是要訓練" in warning, "要把兩個選項都給使用者"
    assert "epochs" not in warning, "這一段同樣不該出現訓練參數"
    print("[OK] 標記完整但類別不平衡時，建議去 Roboflow 補資料，且一樣先不顯示參數")


def case_roboflow_merge_builds_refresh_and_retrain_transition():
    state = {
        "pending_operation": {
            "operation_id": "train-before-augment",
            "type": "training",
            "status": "completed",
            "params": {
                "dataset_name": "bear",
                "variant": "yolov9s.pt",
                "epochs": 3,
                "imgsz": 320,
                "batch": 2,
                "model_name": "bear_exp",
            },
        },
        "plan": {"goal": "改善模型", "steps": []},
    }
    merge_result = {
        "status": "success",
        "dataset_name": "bear",
        "added_images": 12,
        "total_images_now": 42,
    }
    transition = _build_post_augment_training_transition(
        state,
        merge_result,
        {"status": "success", "coverage": 95.0},
    )
    # 補強後的重訓也是先問參數（train_params_gate），不是直接跳確認框
    assert transition["next"] == "train_params_gate"
    assert transition["dataset_refresh"]["total_images_now"] == 42
    assert transition["post_augment_eval_required"] is True
    assert transition["pending_operation"]["operation_id"] != "train-before-augment"
    assert transition["pending_operation"]["params"]["model_name"].startswith("bear_exp_aug_")
    assert [s["agent"] for s in transition["plan"]["steps"][-2:]] == ["model_agent", "evaluation_agent"]

    blocked = _build_post_augment_training_transition(
        state,
        merge_result,
        {"status": "success", "coverage": 60.0},
    )
    assert blocked["next"] == "FINISH"
    assert blocked["dataset_refresh"]["dataset_name"] == "bear"
    assert blocked["post_augment_eval_required"] is False
    print("[OK] Roboflow 合併後會立即刷新資料集；健康檢查通過才建立新的訓練確認與評估步驟")


def case_roboflow_merge_continues_through_confirm_train_and_eval():
    graph = build_multi_agent(FakeMultiAgentModel([
        Plan(goal="補強後重訓並評估", steps=[]),
        # 補強後的重訓一樣要先經過 train_params_gate 問一次參數
        TrainingParamsReply(intent="use_suggested"),
        _tool_call("skill_train", {}, "call_aug_train"),
        _report_call("success", "補強後模型訓練完成。", call_id="call_aug_train_report"),
        _tool_call("skill_eval", {"model_name": "bear_aug_test"}, "call_aug_eval"),
        _report_call("success", "mAP50=0.72，已通過 0.5 門檻。", call_id="call_aug_eval_report"),
    ]))
    config = {"configurable": {"thread_id": _thread("smoke_post_augment_pipeline")}}
    proposal = {
        "status": "proposal_ready",
        "proposal_id": "smoke-post-augment",
        "dataset_name": "bear",
        "candidate_count": 1,
        "candidate": {"name": "Bear Mix"},
    }
    initial = {
        "messages": [HumanMessage(content="接受補強後重新訓練並評估")],
        "roboflow_proposal": proposal,
        "roboflow_proposal_handled": False,
        "pending_operation": {
            "operation_id": "train-before-augment",
            "type": "training",
            "status": "completed",
            "params": TrainingProposal(dataset_name="bear", epochs=1, model_name="bear_exp").model_dump(),
        },
    }

    with patch("multi_agent.apply_roboflow_proposal.func", return_value={
        "status": "success",
        "dataset_name": "bear",
        "source": "Bear Mix",
        "added_images": 12,
        "before_images": 30,
        "total_images_now": 42,
    }), patch("multi_agent.check_dataset_health.func", return_value={
        "status": "success", "coverage": 95.0,
    }), patch("multi_agent.skill_train.func", return_value={
        "status": "completed", "model_name": "bear_aug_test",
    }), patch("multi_agent.skill_eval.func", return_value={
        "status": "success",
        "gate_passed": True,
        "resolved_model_name": "bear_aug_test",
        "final_map50": 0.72,
        "final_loss": 0.8,
        "charts": [],
    }):
        first = graph.invoke(initial, config=config)
        assert first["__interrupt__"][0].value["action_type"] == "roboflow_proposal"

        after_merge = graph.invoke(Command(resume={"approved": True}), config=config)
        assert after_merge.get("pending_param_question"), "補強後的重新訓練也要先問一次參數"
        assert after_merge["dataset_refresh"]["total_images_now"] == 42

        second = _resume_param_question(graph, config, after_merge)
        assert second["__interrupt__"][0].value["action_type"] == "training"

        third = graph.invoke(Command(resume={"approved": True}), config=config)
        assert third["__interrupt__"][0].value["action_type"] == "model_save"

        result = graph.invoke(Command(resume={"approved": False}), config=config)

    tool_names = [m.name for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "skill_train" in tool_names and "skill_eval" in tool_names
    assert result["post_augment_eval_required"] is False
    assert any("重新訓練與評估已完成" in getattr(m, "content", "") for m in result["messages"])
    print("[OK] Roboflow 接受合併後會依序刷新、重新確認訓練、執行訓練並自動評估")


def case_manager_auto_retries_known_gate_without_llm():
    # Data Agent 已經完成暫存評估時，Manager 必須直接進入 proposal 確認 Gate，
    # 不再自動重訓，也不讓 LLM 有機會提早 FINISH。拒絕後正式資料集不會變。
    graph = build_multi_agent(FakeMultiAgentModel([Plan(goal="評估 Roboflow 補強資料", steps=[])]))
    config = {"configurable": {"thread_id": _thread("smoke_proposal_gate")}}
    proposal = {
        "status": "proposal_ready",
        "proposal_id": "smoke-proposal",
        "dataset_name": "__multi_smoke_retry__",
        "candidate_count": 3,
        "candidate": {
            "name": "Bear Mix",
            "workspace": "demo",
            "project": "bear-mix",
            "version": 2,
            "url": "https://universe.roboflow.com/demo/bear-mix",
            "license": "CC BY 4.0",
        },
        "class_mapping": [{"source_name": "Bear", "target_name": "Bear"}],
        "usable_images": 20,
        "baseline_images": 10,
        "estimated_total_images": 30,
        "risks": [],
    }
    first = graph.invoke(
        {
            "messages": [HumanMessage(content="評估完再讓我確認")],
            "roboflow_proposal": proposal,
            "roboflow_proposal_handled": False,
        },
        config=config,
    )
    assert "__interrupt__" in first, "有 proposal 時必須停在最後確認 Gate"
    payload = first["__interrupt__"][0].value
    assert payload["action_type"] == "roboflow_proposal"
    assert payload["proposal"]["proposal_id"] == "smoke-proposal"

    result = graph.invoke(Command(resume={"approved": False}), config=config)
    assert result["roboflow_proposal_handled"] is True
    assert "正式資料集沒有變更" in result["messages"][-1].content
    print("[OK] Data Agent 完成暫存評估後，Manager 一定顯示最終 proposal；拒絕時不修改正式資料集")


def case_manager_gives_up_after_retry_limit():
    # 已經搜尋過但沒有產生相容 proposal 時，固定規則應直接停止，不會無限搜尋、
    # 自動重訓或再問一次 LLM。
    graph = build_multi_agent(FakeMultiAgentModel([Plan(goal="搜尋 Roboflow 候選", steps=[])]))

    result = graph.invoke(
        {
            "messages": [HumanMessage(content="模型還是沒過")],
            "last_report": {
                "agent": "evaluation_agent",
                "status": "blocked",
                "reason": "mAP50 仍未達標",
                "suggestion": "",
                "reason_code": "eval_gate_failed",
            },
            "evaluation": {"verdict": "step_failed", "reason_code": "eval_gate_failed", "progress_since_last_round": False},
            "augment_offered": True,
        },
        config={"configurable": {"thread_id": _thread("smoke_retry_limit")}},
    )

    assert not any(isinstance(m, ToolMessage) for m in result["messages"]), "不該再呼叫任何子 Agent"
    assert "沒有找到" in result["messages"][-1].content
    print("[OK] Roboflow 已搜尋但沒有相容 proposal 時，Manager 直接停止，不會無限搜尋或重訓")


def case_manager_replans_after_repeated_stall():
    # Model Agent 連續回報同一種「不是已知 Gate」的失敗（status=blocked，reason_code
    # 留空——沒有落在 GATE_ROUTING，Decision Engine 沒有固定規則可用，每次都要問 LLM），
    # Observer 偵測到連續 STALL_ROUNDS_BEFORE_REPLAN 輪都沒有變化，判定 plan_invalid，
    # Manager 應該繞回 Planner 重新規劃，而不是拿著同一份 Plan 一直重試。
    #
    # STALL_ROUNDS_BEFORE_REPLAN=2 代表「連續兩次沒有變化」，也就是同樣的失敗訊號要
    # 出現 3 次（第 1 次沒有比較基準，視為有進展；第 2 次跟第 1 次相同才算第 1 次沒
    # 進展；第 3 次跟第 2 次相同才算第 2 次沒進展，觸發 plan_invalid）。
    assert STALL_ROUNDS_BEFORE_REPLAN == 2, "這個測試的輪數是照 STALL_ROUNDS_BEFORE_REPLAN=2 設計的，改了門檻要一起調整這裡"

    script = [
        Plan(goal="訓練模型", steps=[Step(
            id="1",
            description="訓練模型",
            agent="model_agent",
            is_training=True,
            training_params=TrainingProposal(
                dataset_name="__multi_smoke_stall__",
                epochs=1,
                model_name="smoke_exp3",
            ),
        )]),
        # 每一輪重試都會重新繞過 train_params_gate 問一次參數，所以三輪各要一筆
        # TrainingParamsReply；少一筆的話 Manager 會把下一個腳本項目誤當成參數回覆。
        Decision(action="call_agent", target_agent="model_agent", rationale="開始訓練"),
        TrainingParamsReply(intent="use_suggested"),
        _report_call("blocked", "訓練時遇到未知錯誤", suggestion="重試一次", call_id="call_y1"),
        Decision(action="call_agent", target_agent="model_agent", rationale="再試一次"),
        TrainingParamsReply(intent="use_suggested"),
        _report_call("blocked", "訓練時遇到未知錯誤", suggestion="重試一次", call_id="call_y2"),
        Decision(action="call_agent", target_agent="model_agent", rationale="再試最後一次"),
        TrainingParamsReply(intent="use_suggested"),
        _report_call("blocked", "訓練時遇到未知錯誤", suggestion="重試一次", call_id="call_y3"),
        Plan(goal="訓練模型（改用其他方式排查）", steps=[Step(id="1", description="請使用者確認環境設定", agent="model_agent")]),
        Decision(action="ask_user", rationale="重新規劃後判斷需要使用者提供更多資訊", final_reply="訓練一直卡在同樣的錯誤，能否提供更多環境或資料細節？"),
    ]
    graph = build_multi_agent(FakeMultiAgentModel(script))

    _make_fake_dataset("__multi_smoke_stall__")
    try:
        result = _invoke_confirming_training(
            graph,
            {"messages": [HumanMessage(content="幫我訓練模型")]},
            config={"configurable": {"thread_id": _thread("smoke_replan_stall")}},
        )
    finally:
        shutil.rmtree(DATASET_DIR / "__multi_smoke_stall__", ignore_errors=True)

    assert not result.get("evaluation") or result["evaluation"]["verdict"] != "plan_invalid", (
        "重新規劃後 evaluation 應該被清掉，不會停在 plan_invalid"
    )
    assert result["plan"]["goal"] == "訓練模型（改用其他方式排查）", "應該是重新規劃後的第二份 Plan，不是原本那份"
    assert result["messages"][-1].content == "訓練一直卡在同樣的錯誤，能否提供更多環境或資料細節？"
    print("[OK] Observer 連續偵測到同一種未知失敗沒有進展，Manager 繞回 Planner 重新規劃，"
          "不是拿著卡住的 Plan 一直問 LLM")


def case_manager_answers_directly_without_sub_agent():
    # 使用者只是打招呼，這時還沒有任何子 Agent 回報過，Manager 應該直接判斷
    # action="finish" 並在 final_reply 放自然語言回覆，不能留空白內容給使用者。
    script = [
        Plan(goal="打招呼", steps=[]),
        Decision(action="finish", rationale="使用者只是打招呼", final_reply="你好，我可以幫你訓練、推論或評估模型。"),
    ]
    graph = build_multi_agent(FakeMultiAgentModel(script))
    result = graph.invoke(
        {"messages": [HumanMessage(content="你好")]},
        config={"configurable": {"thread_id": _thread("smoke_greeting")}},
    )
    assert len(result["messages"]) == 2, "不需要任何子 Agent 時，不該多繞一圈，但也不能沒有回覆內容"
    assert result["messages"][-1].content == "你好，我可以幫你訓練、推論或評估模型。"
    print("[OK] 不需要子 Agent 時，Planner 給空計畫、Manager 直接回答，不會回傳空白內容給使用者")


def run():
    case_extract_text_handles_gemini_content_blocks()
    case_parse_tool_content_handles_json_string_and_dict()
    case_manager_routes_to_data_agent_then_finish()
    case_manager_hands_off_data_agent_to_model_agent_to_evaluation_agent()
    case_training_confirmation_contains_exact_params_and_operation_id()
    case_training_tool_rejects_unapproved_params()
    case_unhealthy_dataset_asks_before_showing_params()
    case_labeled_but_imbalanced_suggests_roboflow()
    case_roboflow_merge_builds_refresh_and_retrain_transition()
    case_roboflow_merge_continues_through_confirm_train_and_eval()
    case_manager_auto_retries_known_gate_without_llm()
    case_manager_gives_up_after_retry_limit()
    case_manager_replans_after_repeated_stall()
    case_manager_answers_directly_without_sub_agent()
    print("\n全部驗證通過。")


if __name__ == "__main__":
    run()
