"""manager_agent.py 的手動驗證腳本。

用一個假的 ChatModel（FakeManagerModel）取代真的 LLM，依照預先寫好的
腳本回覆，藉此驗證 Manager Agent 的 ReAct 迴圈（LLM 判斷 -> 呼叫 Tool
-> 把結果餵回去 -> LLM 給最終答案）接線是對的。

這裡完全不會呼叫 Gemini/OpenAI 等任何真的 LLM API，不需要 .env 裡有
有效的金鑰，也不會產生任何額度費用。

執行方式：python manager_agent_smoke_test.py
"""
import shutil

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent_tools import DATASET_DIR
from manager_agent import build_manager_agent


class FakeManagerModel:
    """依照預先寫好的回覆腳本作答，不呼叫任何真的 LLM。"""

    def __init__(self, script):
        self._script = list(script)

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return self._script.pop(0)


def _make_fake_dataset(name: str):
    ds = DATASET_DIR / name
    (ds / "images").mkdir(parents=True, exist_ok=True)
    (ds / "labels").mkdir(parents=True, exist_ok=True)
    (ds / "images" / "a.jpg").write_bytes(b"\xff\xd8\xff\xd9")
    (ds / "labels" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")


def case_manager_calls_tool_then_answers():
    tool_call_response = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "check_dataset_health",
                "args": {"dataset_name": "__manager_smoke_ds__"},
                "id": "call_1",
            }
        ],
    )
    final_response = AIMessage(content="資料集健康度是 100 分，可以開始訓練。")
    graph = build_manager_agent(FakeManagerModel([tool_call_response, final_response]))

    _make_fake_dataset("__manager_smoke_ds__")
    try:
        result = graph.invoke({"messages": [HumanMessage(content="幫我看一下資料集健康度")]})
    finally:
        shutil.rmtree(DATASET_DIR / "__manager_smoke_ds__", ignore_errors=True)

    messages = result["messages"]
    assert any(isinstance(m, ToolMessage) for m in messages), "應該真的呼叫到 check_dataset_health 這個 Tool"
    assert messages[-1].content == "資料集健康度是 100 分，可以開始訓練。"
    print("[OK] Manager Agent 會判斷要呼叫 Tool、拿到真的執行結果、再給出最終回覆")


def case_manager_answers_directly():
    direct_response = AIMessage(content="你好，我可以幫你訓練、推論或評估模型。")
    graph = build_manager_agent(FakeManagerModel([direct_response]))

    result = graph.invoke({"messages": [HumanMessage(content="你是誰？")]})

    assert len(result["messages"]) == 2, "不需要工具時，不該多繞一圈去呼叫 Tool"
    assert result["messages"][-1].content == "你好，我可以幫你訓練、推論或評估模型。"
    print("[OK] 不需要工具時，Manager Agent 會直接回覆，不會硬套工具")


def run():
    case_manager_calls_tool_then_answers()
    case_manager_answers_directly()
    print("\n全部驗證通過。")


if __name__ == "__main__":
    run()
