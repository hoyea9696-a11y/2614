"""Manager Agent（Step 3）。

跟 Step 1/2 不一樣，這裡才第一次真正需要 LLM——Manager Agent 要自己判斷
使用者說的話對應哪個 Tool（或不需要用 Tool，直接回答）。對應 spec 文件
的原則：「Agent 決定要做什麼」，實際「怎麼做」還是交給 Step 2 包好的
agent_tools.py，不重新實作那些邏輯。

這個檔案只建立 Manager Agent 的 Graph（LLM + Tool 的 ReAct 迴圈），
不會新增任何網站路由——要不要接到 main.py 的 API 留給下一個 PR，
先確認這一層的推理迴圈本身是對的。
"""
import os
import sys
from pathlib import Path

from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode

SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent_tools import ALL_TOOLS

MANAGER_SYSTEM_PROMPT = """你是 AI Detector 系統的 Manager Agent，系統協調者。

你的職責：
- 理解使用者需求，判斷任務類型
- 選擇適合的 Tool 呼叫（check_dataset_health / train_yolo_model /
  evaluate_yolo_model / run_yolo_inference）
- 不需要用 Tool 就能回答的問題，直接回答，不要硬套工具

你不負責：
- 實際的模型訓練、推論、資料處理邏輯（那些都在 Tool 裡面，你只負責決定呼叫哪個）
"""


def _load_env():
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def get_default_chat_model():
    """預設沿用現有 .env 裡的 Gemini 設定（跟 ai_service.py 一致），
    之後要換 Groq/Ollama，只要在這裡換一個 LangChain ChatModel 實作即可，
    Manager Agent 的 Graph 本身不用改。"""
    _load_env()
    from langchain_google_genai import ChatGoogleGenerativeAI

    model_name = os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(model=model_name, google_api_key=os.getenv("GEMINI_API_KEY"))


def build_manager_agent(chat_model):
    """chat_model 用依賴注入的方式傳進來，方便測試時換成假模型，
    不需要真的呼叫任何 LLM API。"""
    model_with_tools = chat_model.bind_tools(ALL_TOOLS)
    tool_node = ToolNode(ALL_TOOLS)

    def call_model(state: MessagesState) -> dict:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=MANAGER_SYSTEM_PROMPT)] + list(messages)
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}

    def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "end"

    graph = StateGraph(MessagesState)
    graph.add_node("manager", call_model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "manager")
    graph.add_conditional_edges("manager", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "manager")

    return graph.compile()
