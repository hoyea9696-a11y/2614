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
  evaluate_yolo_model / run_yolo_inference / download_roboflow_dataset）
- 不需要用 Tool 就能回答的問題，直接回答，不要硬套工具

補資料規則：
- 如果 train_yolo_model 回傳 status="dataset_not_found"，代表本機沒有這個資料集，
  不要直接放棄或回問使用者要不要自己找資料。你要自己根據資料集名稱／使用者描述的關鍵字，
  猜一個最可能符合的 Roboflow Universe workspace 與 project slug，呼叫
  download_roboflow_dataset 下載；如果回傳 status="not_found"，換一個猜測再試，
  最多嘗試 3 次不同猜測，全部失敗才回報使用者需要提供正確的資料集網址。
- 下載成功後，用同一個 dataset_name 重新呼叫 train_yolo_model 繼續原本的任務。

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


def get_default_chat_model(model_type: str = "gemini"):
    """依 model_type 選擇底層 LLM。設定（API Key、模型名稱、Ollama 位址）直接沿用
    ai_service.py 的 _get_config()，不重複定義一次同樣的 .env 讀取邏輯——跟一般
    聊天視窗（/api/ai/chat）共用同一份設定來源，改 .env 兩邊都會同時生效。

    Gemini 走原生 langchain_google_genai；Groq/Gemma/Local 都是 OpenAI 相容端點，
    統一用 langchain_openai.ChatOpenAI 呼叫（只是換 base_url/api_key）。

    注意：Manager Agent 的 ReAct 迴圈跟 Decision Engine 都依賴 tool-calling／
    structured output，Groq 上的一般對話模型（llama-3.3 等）支援沒問題，但本機
    Ollama 模型是否支援 function calling 要看實際模型，不保證都能正常跑完整個
    多輪協調流程。"""
    _load_env()
    from ai_service import AIService

    conf = AIService()._get_config()
    platform = (model_type or "gemini").lower()
    if platform == "gemma":
        # 跟 ai_service.py 同一套判斷：Gemma 系列在 Groq 上更穩定，沒有 Groq 金鑰才退回本機 Ollama
        platform = "groq" if conf.get("groq_api_key") else "local"

    if platform in ("gemini", "gemini-pro"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = conf.get("gemini_pro_model") if platform == "gemini-pro" else conf.get("gemini_flash_model", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=conf.get("gemini_api_key"))

    from langchain_openai import ChatOpenAI

    if platform == "groq":
        return ChatOpenAI(
            model=conf.get("groq_model", "llama-3.3-70b-versatile"),
            api_key=conf.get("groq_api_key"),
            base_url="https://api.groq.com/openai/v1",
        )
    return ChatOpenAI(
        model=conf.get("ollama_model", "llama3.2"),
        api_key="ollama",
        base_url=conf.get("ollama_base_url", "http://localhost:11434/v1"),
    )


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
