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


def _invoke_confirming_training(graph, initial_input, config):
    """跟 stream_trace() 對 interrupt 的處理邏輯一樣：Plan 裡有訓練字樣的步驟第一次
    派給 model_agent 時，train_confirm_gate 會 interrupt() 暫停——這裡直接模擬使用者
    按下確認（approved=True）讓流程恢復，測試才能繼續往下走到 model_agent/skill_train。"""
    result = graph.invoke(initial_input, config=config)
    if "__interrupt__" in result:
        result = graph.invoke(Command(resume={"approved": True}), config=config)
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
            Step(id="2", description="訓練模型", agent="model_agent"),
            Step(id="3", description="評估模型", agent="evaluation_agent"),
        ]),
        Decision(action="call_agent", target_agent="data_agent", rationale="需要先確認/補齊資料"),
        _tool_call("download_roboflow_dataset", {"workspace": "ws", "project": "proj", "dataset_name": "__multi_smoke_train__"}, "call_1"),
        _report_call("success", "已從 Roboflow 補齊資料，共 1 張圖片。", call_id="call_2"),
        Decision(action="call_agent", target_agent="model_agent", rationale="資料備妥，開始訓練"),
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

    stalled_leg = [
        _report_call("blocked", "訓練時遇到未知錯誤", suggestion="重試一次", call_id="call_y"),
    ]
    script = [
        Plan(goal="訓練模型", steps=[Step(id="1", description="訓練模型", agent="model_agent")]),
        Decision(action="call_agent", target_agent="model_agent", rationale="開始訓練"),
        *stalled_leg,  # 第 1 次卡住
        Decision(action="call_agent", target_agent="model_agent", rationale="再試一次"),
        *stalled_leg,  # 第 2 次卡住（跟第 1 次同樣訊號，stall_count=1）
        Decision(action="call_agent", target_agent="model_agent", rationale="再試最後一次"),
        *stalled_leg,  # 第 3 次卡住（跟第 2 次同樣訊號，stall_count=2 -> plan_invalid）
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
    case_manager_auto_retries_known_gate_without_llm()
    case_manager_gives_up_after_retry_limit()
    case_manager_replans_after_repeated_stall()
    case_manager_answers_directly_without_sub_agent()
    print("\n全部驗證通過。")


if __name__ == "__main__":
    run()
