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
from typing import Optional

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
  evaluate_yolo_model / run_yolo_inference / search_roboflow_universe /
  prepare_roboflow_augmentation / download_roboflow_dataset）
- 不需要用 Tool 就能回答的問題，直接回答，不要硬套工具

補資料規則：
- 如果 train_yolo_model 回傳 status="dataset_not_found"，代表本機沒有這個資料集，
  先用 search_roboflow_universe 依資料集名稱／使用者描述搜尋真實候選，再把搜尋結果的
  workspace、project 與 version 傳給 download_roboflow_dataset，不要自行猜 slug。
- 下載成功後，用同一個 dataset_name 重新呼叫 train_yolo_model 繼續原本的任務。
- 補強既有資料集時只可呼叫 prepare_roboflow_augmentation 產生暫存 proposal；
  正式合併必須由 Multi-Agent LangGraph 的人工確認 Gate 處理，你不能自行套用。

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


# LLM 呼叫（Planner/Decision Engine/子 Agent 的 ReAct 迴圈）沒有這層上限的話，
# provider 端網路卡頓或無回應時 .invoke() 會不斷等下去，_invoke_with_retry 的
# 重試也只是再等一次同樣沒上限的呼叫，使用者只能整個關掉伺服器才能脫困。
LLM_REQUEST_TIMEOUT_SECONDS = 60


def get_default_chat_model(model_type: str = "gemini", requests_per_second: Optional[float] = None):
    """依 model_type 選擇 Manager Agent 底層的 LLM。設定（API Key、模型名稱）直接
    沿用 ai_service.py 的 _get_config()，不重複定義一次同樣的 .env 讀取邏輯。

    Manager Agent 引擎選單目前只開放三個選項：
    - gemini：langchain_google_genai，原生支援 structured output。
    - gpt：OpenAI 官方 API（langchain_openai.ChatOpenAI 預設走 OpenAI 端點），
      預設用 gpt-4.1-mini，可用 .env 的 OPENAI_MODEL 覆蓋。
    - gpt-oss：Groq 上的 openai/gpt-oss-120b（OpenAI 相容端點），可用 .env 的
      GPT_OSS_MODEL 覆蓋 model id。原本這裡是 Llama 4 Scout
      （meta-llama/llama-4-scout-17b-16e-instruct），但 Groq 已在 2026-06-17
      棄用該模型，實測會直接回 404 model_not_found，改用 Groq 官方建議的
      替代模型。

    Manager Agent 的 ReAct 迴圈跟 Decision Engine 都依賴 tool-calling／structured
    output，這三個選項都經過 with_structured_output(method="function_calling")
    驗證過可以跑，不是隨便換一個模型都保證能跑完整個多輪協調流程。

    requests_per_second 不傳時完全不節流，行為跟以前一樣（Manager 模式就是這樣用的，
    它一輪只有幾次呼叫，撞不到 provider 的 RPM）。傳了才掛 rate limiter，目前只有
    Deep Agent 模式會傳——理由見 main.py 的 DEEP_AGENT_REQUESTS_PER_SECOND。"""
    _load_env()
    from ai_service import AIService

    conf = AIService()._get_config()
    platform = (model_type or "gemini").lower()

    # BaseChatModel 的 rate_limiter 欄位（所有 provider 都吃這個），在真正發出 HTTP
    # 請求前先擋一道。max_bucket_size=1 是刻意的：允許 burst 就等於允許並行 fan-out
    # 一口氣把整分鐘的額度打完，那正是要防的情況。
    rate_limiter = None
    if requests_per_second:
        from langchain_core.rate_limiters import InMemoryRateLimiter

        rate_limiter = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            max_bucket_size=1,
        )

    if platform in ("gemini", "gemini-pro"):
        from langchain_google_genai import ChatGoogleGenerativeAI

        model_name = conf.get("gemini_pro_model") if platform == "gemini-pro" else conf.get("gemini_flash_model", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=conf.get("gemini_api_key"),
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
            rate_limiter=rate_limiter,
        )

    from langchain_openai import ChatOpenAI

    if platform in ("gpt", "openai"):
        return ChatOpenAI(
            model=conf.get("openai_model", "gpt-4.1-mini"),
            api_key=conf.get("openai_api_key"),
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
            rate_limiter=rate_limiter,
        )

    if platform in ("gpt-oss", "llama4-scout", "llama-4-scout"):
        return ChatOpenAI(
            model=conf.get("gpt_oss_model", "openai/gpt-oss-120b"),
            api_key=conf.get("groq_api_key"),
            base_url="https://api.groq.com/openai/v1",
            timeout=LLM_REQUEST_TIMEOUT_SECONDS,
            rate_limiter=rate_limiter,
        )

    raise ValueError(f"Manager Agent 不支援的 model_type：{model_type}")


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
