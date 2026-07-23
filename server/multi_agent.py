"""Multi-Agent 架構：Manager Agent（Planner + Decision Engine + Observer）+
Data Agent + Model Agent + Evaluation Agent。

跟上一版最大的差別：Manager 不再是「收到回報就立刻自己猜下一步」的單一節點，
拆成三個角色，對應 Plan -> Act -> Observe -> Decide 的循環：
- Planner：新請求進來時，把使用者目標拆成一份高層次的 Plan（一串 Step，每個
  Step 標明要交給哪個子 Agent）。Decision Engine 判斷目前的 Plan 已經不合理時
  （plan_invalid），也會繞回 Planner 重新規劃，不是卡死或直接放棄。
- Observer：子 Agent 回報後，先解讀這份回報對 Plan 來說代表什麼（成功/失敗/
  沒有進展/矛盾），產出結構化的 Evaluation，再交給 Decision Engine——Decision
  Engine 不用自己一邊判讀回報一邊決定下一步，兩件事分開才不會又混成一次
  LLM 呼叫瞎猜。Observer 全部是確定性規則，不呼叫 LLM，才不會拖慢每一輪。
- Decision Engine（原 supervisor）：已知的 Gate 失敗類型（GATE_ROUTING）用固定
  規則直接決定，不問 LLM；規則判斷不出來，或 Observer 判定 Plan 已經失效，
  才呼叫 LLM 做結構化決策（Decision），不再用文字關鍵字比對。

子 Agent 結束時必須呼叫 report_to_manager，回報結構化的 status/reason/
suggestion/reason_code，不是留一段自由文字讓 Manager 自己猜。suggestion 只是
子 Agent 的建議，不代表 Manager 一定要照做——路由權永遠在 Manager 身上。

每個已知 Gate 有自己的重試上限（RETRY_LIMITS），跟全域 MAX_ROUNDS 是兩層不同
的保險絲：MAX_ROUNDS 防整個對話跑不完，RETRY_LIMITS 防同一個 Gate 一直重試卻
沒有進展；Observer 額外用 stall_count 偵測「同一種非已知 Gate 的訊號連續兩輪
沒有變化」，這時判定 plan_invalid，繞回 Planner 重新規劃，而不是讓 Decision
Engine 的 LLM fallback 一直用同一份 Plan 瞎猜。

依照 .agents/AGENTS.md 的分工設計，並對齊
「AI Detector Multi-Agent System Architecture.docx」的 Agent / Skill / Tool 分層：
- Data Agent：資料集相關，綁 DATA_TOOLS（check_dataset_health / import_images_to_dataset /
  download_roboflow_dataset）。
- Model Agent：只綁 Skill 層的 skill_train / skill_infer，不直接綁原始 Tool——
  訓練/推論裡「資料集存不存在」「該用哪個模型權重」這些判斷已經寫成確定性 Gate
  收在 skills.py，Model Agent 不用自己猜。
- Evaluation Agent：只綁 skill_eval，跟文件一樣獨立出來，不跟 Model Agent 混在一起。
- Manager：不執行任何 Tool，只判斷這一輪該交給哪個子 Agent，或是否已經可以結束。

跟 manager_agent.py（單一 Agent 版本）並存，不刪除舊版。
"""
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from typing import Annotated, Literal, Optional, TypedDict

from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command
from langchain_core.tools import tool

SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent
CHECKPOINT_DB_PATH = PROJECT_ROOT / "data" / "agent_checkpoints.sqlite"
MODELS_DIR = PROJECT_ROOT / "models"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent_tools import check_dataset_health, import_images_to_dataset, download_roboflow_dataset, DATASET_DIR, _count_images
from skills import skill_train, skill_infer, skill_eval

DATA_TOOLS = [check_dataset_health, import_images_to_dataset, download_roboflow_dataset]
MODEL_TOOLS = [skill_train, skill_infer]
EVAL_TOOLS = [skill_eval]

AgentName = Literal["data_agent", "model_agent", "evaluation_agent"]


@tool
def report_to_manager(
    status: Annotated[str, Field(description='這次任務的結果："success"／"blocked"（需要 Manager 換別的方式）／"failed"（重試多次仍失敗）')],
    reason: Annotated[str, Field(description="人類可讀的診斷說明，具體發生了什麼事")],
    suggestion: Annotated[str, Field(description="你認為接下來可以怎麼做，僅供 Manager 參考，不代表最終決定")] = "",
    reason_code: Annotated[str, Field(description="機器可判斷的簡短代碼，只有落在已知類型才填：dataset_not_found / health_gate_failed / eval_gate_failed，其他情況留空")] = "",
) -> dict:
    """回報這次任務的結果給 Manager Agent，這是每個子 Agent結束前必須呼叫的最後一步。

    不要直接用文字結束對話——Manager 只看這個工具的回傳值來決定下一步該找誰，
    你的 suggestion 只是建議，路由權在 Manager，不是你。"""
    return {"status": status, "reason": reason, "suggestion": suggestion, "reason_code": reason_code}


DATA_AGENT_PROMPT = """你是 Data Agent，只負責資料集相關的任務，依下面順序判斷該用哪個工具：
1. 先呼叫 check_dataset_health 確認本機這個資料集是否已經有圖片——已經有就不用再匯入
   或下載，直接回報現況即可。
2. 使用者提到本機/已上傳有現成圖片路徑（例如推論頁面暫存上傳的 data/uploads/ 或
   temp_verify/），且本機資料集還沒有圖片時，呼叫 import_images_to_dataset 把這些
   圖片匯入。這個工具只搬圖片、不會產生標記檔，匯入後健康度是 0% 標記率是正常現象，
   不用因此判定失敗。
3. 本機、使用者提供的路徑都沒有圖片可用時，才呼叫 download_roboflow_dataset 從
   Roboflow Universe 下載補齊：自己猜最符合的 workspace/project slug，猜錯
   （status=not_found）就換一個再試，最多嘗試 3 次，全部失敗才回報無法補齊。

完成後一定要呼叫 report_to_manager 回報結果：本機、上傳、Roboflow 都找不到資料就
status="blocked"、reason_code="dataset_not_found"；其他情況成功就 status="success"，
reason_code 留空。

你不負責訓練、推論、評估模型，這些請交給 Model Agent / Evaluation Agent。"""

MODEL_AGENT_PROMPT = """你是 Model Agent，只負責模型的訓練與推論，用兩個 Skill 完成：
- skill_train：完整訓練流程，內部已經自動檢查資料集是否存在、健康度是否 >= 70 分，
  你不用自己再檢查一次。
- skill_infer：完整推論流程，內部已經自動選好要用哪個模型權重。

完成後一定要呼叫 report_to_manager 回報結果：
- skill_train 回傳 status="dataset_not_found" → status="blocked", reason_code="dataset_not_found"
- skill_train 回傳 status="health_gate_failed" → status="blocked", reason_code="health_gate_failed"，
  reason 附上健康度報告
- skill_train 回傳 status="stopped" → 這是使用者主動要求終止訓練，不是失敗，
  status="blocked", reason_code 留空，reason 寫「使用者已取消這次訓練」，
  不要自己再重試或建議繼續訓練
- skill_train 回傳 status="already_training" → 已經有另一個訓練在跑，
  status="blocked", reason_code 留空，reason 說明目前已有訓練在進行中
- 訓練/推論成功 → status="success", reason_code 留空
不要自己猜要交給誰處理，誠實回報狀態跟原因就好，路由是 Manager 的事。

你不負責資料集健康檢查、下載資料、評估模型效果，這些請交給 Data Agent / Evaluation Agent。"""

EVALUATION_AGENT_PROMPT = """你是 Evaluation Agent，只負責用 skill_eval 評估已經訓練完成
的模型：讀取 mAP/Loss 等指標，判斷有沒有過擬合或資料傾斜，並把結果解釋給使用者。

完成後一定要呼叫 report_to_manager 回報結果：
- skill_eval 回傳 gate_passed=False → status="blocked", reason_code="eval_gate_failed"，
  reason 附上關鍵指標（mAP50/loss）
- gate_passed=True → status="success", reason_code 留空
suggestion 可以給診斷建議（例如資料不足、過擬合），但要不要照做是 Manager 決定。

你不負責訓練模型或處理資料集，這些請交給 Model Agent / Data Agent。"""

PLANNER_SYSTEM_PROMPT = """你是 Manager Agent 的規劃模組。看使用者的請求（跟目前的對話紀錄），
把它拆成高層次的步驟清單，每個步驟只能交給以下三個子 Agent 之一負責：
- data_agent：確認/補齊資料集
- model_agent：訓練、推論
- evaluation_agent：評估模型效果

不用寫技術細節，只要列出「做什麼、交給誰」，依實際執行順序排列。如果使用者的請求
不需要任何子 Agent（例如打招呼、問一般問題），steps 留空清單即可。

指派給 model_agent 的步驟要標 is_training：這個步驟本身要不要啟動一次新的訓練
流程。使用者要求「用剛剛訓練好的模型做推論/驗證」是 is_training=False——句子裡
出現「訓練」兩個字只是在描述要用哪個模型，動作本身是推論/驗證，不是訓練；只有
真的要開始訓練一個新模型時才是 is_training=True。這個欄位會直接決定要不要跳出
「確認是否訓練」的畫面，標錯會讓使用者在做推論時被誤問一次訓練確認。

如果對話裡已經有失敗紀錄，代表原本的計畫卡住了，這次要參考失敗原因重新規劃，
不要原封不動照抄同一份計畫。"""

MANAGER_SYSTEM_PROMPT = """你是 Manager Agent，系統唯一的決策者。你會看到使用者原始訊息、
目前的計畫進度，以及（如果有的話）Observer 對上一個子 Agent 回報的結構化判讀
（verdict/progress_since_last_round/contradiction）。

你要決定下一步：
- action="call_agent" + target_agent：還需要某個子 Agent 繼續處理
  - data_agent：資料集健康度、下載/補齊資料
  - model_agent：訓練、推論
  - evaluation_agent：評估模型效果
- action="finish"：任務已完成或已經有足夠答案，final_reply 填入要給使用者看的完整回覆
- action="ask_user"：需要使用者提供更多資訊才能繼續，final_reply 填入要問使用者的問題

Roboflow workspace/project slug、確切下載網址這類技術細節，是 data_agent 自己的職責
（它會自己猜、猜錯再換一個試），不算「缺資訊」——Plan 已經把某個步驟指派給某個子
Agent 時，直接呼叫那個子 Agent 讓它自己想辦法，不要因為使用者沒有提供這些技術細節
就選 ask_user。action="ask_user" 只留給子 Agent 已經回報卡住（例如 blocked/
dataset_not_found 重試上限用完），且需要使用者提供只有人類才知道的資訊
（例如使用者自己要驗證的圖片檔案本身）時才用。

訓練資料集的選擇例外：如果要指派給 model_agent 的步驟是訓練，使用者訊息裡沒有
明確指名要用哪一個本機資料集，且下面列出的「目前本機已有的資料集」不只一個，
不要自己隨便挑一個或叫 model_agent 自己猜——用 action="ask_user"，final_reply
列出這些資料集名稱請使用者選一個。如果目前本機剛好只有一個資料集，直接指派給
model_agent 使用它即可，不用多問。

訓練參數的選擇例外：這台機器是 CPU-only、記憶體有限，訓練很吃資源，直接套用
預設參數（yolov9m.pt/epochs=50/imgsz=640/batch=16）容易讓整台機器當機。如果
要指派給 model_agent 的步驟是訓練，且使用者訊息裡（包含之前的對話）沒有明確
指定 epochs/imgsz/batch/模型權重這些訓練參數，不要直接放行——用 action=
"ask_user"，final_reply 列出建議的低資源預設值（例如 yolov9s.pt、epochs=20、
imgsz=320、batch=4），請使用者確認要不要用這組，或自己指定別的數值。使用者已經
明確講清楚要用的參數（不管是這輪還是對話中前面已經指定過），就直接照做，不用
重複問。

子 Agent 的 suggestion 只是參考意見，不代表你一定要照做，你可以選擇別的 Agent 或直接結束。"""

# 已知的 Gate 失敗類型，直接用固定規則決定下一步找誰，不問 LLM——
# 結果才會穩定，也不會像純 LLM 分類那樣被自然語言雜訊誤導成提早結束。
GATE_ROUTING = {
    "dataset_not_found": "data_agent",
    "health_gate_failed": "data_agent",
    "eval_gate_failed": "model_agent",
}
# 每個 Gate 各自的重試上限，跟全域 MAX_ROUNDS 是兩層不同的保險絲：
# 這裡管「同一個 Gate 最多重試幾次」，避免無限迴圈卻沒有實際進展。
RETRY_LIMITS = {
    "dataset_not_found": 3,
    "health_gate_failed": 2,
    "eval_gate_failed": 3,
}
# Observer 判定「連續幾輪同一種非已知 Gate 的訊號都沒有變化」就算 Plan 失效，
# 該重新規劃了，不要讓 Decision Engine 拿著同一份 Plan 一直瞎猜。
STALL_ROUNDS_BEFORE_REPLAN = 2

# 同一個使用者請求最多繞回 Planner 重新規劃幾次——沒有這層上限，replan 會跟真正
# 執行 Agent 步驟共用同一個 MAX_ROUNDS 預算，卡住的請求可能把整個預算都耗在
# 「重新規劃」上，一次真正的步驟都沒跑到就 FINISH。
MAX_REPLANS = 2

MAX_ROUNDS = 8

# 訓練是高成本、耗時的操作，啟動前要讓使用者確認——但 Manager 只知道「這一步要
# 交給 model_agent」，不知道 model_agent 內部要跑 skill_train 還是 skill_infer
# （那是 model_agent 自己的 ReAct 迴圈決定的）。原本用 Step 描述裡有沒有「訓練」
# 字樣做關鍵字判斷，但使用者要求「用剛剛訓練好的模型做推論」這種描述一樣會包含
# 「訓練」兩個字，關鍵字比對會誤判成要開新訓練，跳出不該出現的確認畫面。改成用
# Step.is_training 這個由 Planner 直接判斷、結構化的欄位，不做字串比對。


def _existing_dataset_names() -> list:
    """列出本機真正「有圖片」的資料集名稱（跟 skill_train 的 Dataset Gate 用同一個
    判斷條件），給 Manager 在決定要不要問使用者「要用哪個資料集」時參考——這裡只是
    列清單，要不要因此問使用者是 Manager 的 LLM 判斷，不在這裡用規則寫死。"""
    if not DATASET_DIR.exists():
        return []
    return sorted(d.name for d in DATASET_DIR.iterdir() if d.is_dir() and _count_images(d) > 0)


class Decision(BaseModel):
    """Manager 每輪的結構化決策，取代舊版「回一個關鍵字」的自由文字分類。"""

    action: Literal["call_agent", "finish", "ask_user"] = Field(
        description="call_agent：交給子 Agent 繼續處理；finish：已有足夠答案；ask_user：需要使用者補充資訊"
    )
    target_agent: Optional[AgentName] = Field(
        default=None, description="action 是 call_agent 時必填，指定要交給哪個子 Agent"
    )
    rationale: str = Field(description="做出這個決定的理由，用於稽核紀錄")
    final_reply: Optional[str] = Field(
        default=None, description="action 是 finish 或 ask_user 時，這段文字會直接顯示給使用者"
    )


class Step(BaseModel):
    """Plan 裡的一個步驟，只描述「打算做什麼、交給誰」，不包含事實細節或失敗原因——
    那些是 AgentReport／Evaluation 的責任，Step 只記錄意圖跟目前做到哪了。"""

    id: str = Field(description="步驟編號，字串，例如 '1'")
    description: str = Field(description="這個步驟要做什麼，簡短描述")
    agent: AgentName = Field(description="這個步驟由哪個子 Agent 負責")
    is_training: bool = Field(
        default=False,
        description="這個步驟指派給 model_agent 時，是不是要啟動一次新的訓練流程"
                     "（不是推論、不是評估，也不是單純提到「剛剛訓練好的模型」）。"
                     "指派給 data_agent／evaluation_agent 的步驟一律 False。",
    )


class Plan(BaseModel):
    """Planner 的結構化輸出。"""

    goal: str = Field(description="使用者這次請求的目標，一句話")
    steps: list[Step] = Field(
        default_factory=list,
        description="達成目標需要哪些步驟，依序排列；不需要任何子 Agent 就回空清單",
    )


class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next: str
    rounds: int
    retry_counters: dict
    last_report: Optional[dict]
    plan: Optional[dict]
    evaluation: Optional[dict]
    stall_count: int
    usage: Optional[dict]
    replan_count: int
    train_confirmed: bool
    last_trained_model: Optional[str]
    model_save_offered: bool


def _build_sub_agent(chat_model, tools, system_prompt):
    """跟舊版同一套 ReAct 迴圈骨架，多綁一個 report_to_manager 作為結尾用的結構化回報工具。

    子 Agent 呼叫 report_to_manager 後就直接結束這個子圖，不會再繞回 LLM 多問一次
    「還有沒有事要做」——省一次不必要的 LLM 呼叫，也讓「什麼時候算結束」是確定性規則
    （呼叫了 report_to_manager 就是結束），不是靠 LLM 自己決定要不要繼續講話。
    """
    all_tools = list(tools) + [report_to_manager]
    model_with_tools = chat_model.bind_tools(all_tools)
    tool_node = ToolNode(all_tools)

    def call_model(state) -> dict:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt)] + list(messages)
        try:
            response = _invoke_with_retry(lambda: model_with_tools.invoke(messages))
        except Exception as e:
            # 重試後仍失敗：回一則沒有 tool_calls 的訊息讓這個子 Agent 自然結束
            # （should_continue 判斷沒有 tool_calls 就是 "end"），_extract_agent_report
            # 找不到 report_to_manager 時本來就會退化成 status="partial"，讓 Observer/
            # Decision Engine 走既有的「未知失敗」流程決定下一步，而不是讓例外往上
            # 一路炸穿 graph.stream()，中斷整段對話。
            response = AIMessage(content=f"（LLM 呼叫失敗，這一輪任務中止：{e}）")
        return {"messages": [response]}

    def should_continue(state) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "end"

    def after_tools(state) -> str:
        last = state["messages"][-1]
        if isinstance(last, ToolMessage) and last.name == "report_to_manager":
            return "end"
        return "agent"

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_conditional_edges("tools", after_tools, {"agent": "agent", "end": END})
    return graph.compile()


def _extract_text(content) -> str:
    """LangChain 的 AIMessage.content 不保證是純字串——Gemini 有時會回傳一包內容
    區塊的 list（例如帶 grounding/citation metadata 的區塊），這裡統一轉成純文字。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)
    return str(content) if content else ""


def _usage_from_message(m) -> Optional[dict]:
    """從一則 AIMessage 撈 usage_metadata（input/output/total tokens）——不是每個
    provider、每個模型都會填這個欄位（例如本機 Ollama 常常沒有），拿不到就回
    None，不要假裝有數字。"""
    usage = getattr(m, "usage_metadata", None) if m is not None else None
    if not usage:
        return None
    return {
        "input_tokens": usage.get("input_tokens", 0) or 0,
        "output_tokens": usage.get("output_tokens", 0) or 0,
        "total_tokens": usage.get("total_tokens", 0) or 0,
    }


def _invoke_with_retry(invoke_fn, retries: int = 1):
    """呼叫底層 LLM 時，Reasoning 模型（例如 Groq 的 gpt-oss-120b）偶爾會把推理過程
    誤當成最終輸出，導致 tool-calling／structured output 解析失敗，provider 直接
    回 400 拋例外——這種通常換一次生成就會恢復正常，不是每次都會發生。這裡固定重試
    一次，兩次都失敗才真的往上拋，讓呼叫端自己決定怎麼優雅降級，而不是讓整個
    graph.stream() 直接崩潰、中斷整段對話。"""
    last_err = None
    for _ in range(retries + 1):
        try:
            return invoke_fn()
        except Exception as e:
            last_err = e
    raise last_err


def _parse_tool_content(content) -> dict:
    """ToolMessage.content 依 LangChain 版本可能已經是 dict，也可能被序列化成 JSON
    字串——這裡統一還原成 dict，解析失敗就包成 {"raw": ...} 而不是直接拋例外。"""
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            return json.loads(content)
        except (ValueError, TypeError):
            return {"raw": content}
    return {"raw": str(content)}


def _extract_trained_model_name(new_messages: list) -> Optional[str]:
    """model_agent 這一輪如果成功呼叫 skill_train 訓練出一個新模型，回傳它的
    model_name，讓任務結束時可以問使用者要不要存進模型庫。Gate 沒過、使用者
    取消、或這輪其實只是推論而沒有真的呼叫 skill_train，都回 None，不會誤問。"""
    for m in new_messages:
        if isinstance(m, ToolMessage) and m.name == "skill_train":
            data = _parse_tool_content(m.content)
            if data.get("status") == "completed":
                return data.get("model_name")
    return None


def _add_model_to_library(model_name: str) -> dict:
    """把 runs/train/<model_name>/weights/best.pt 複製一份到 models/ 資料夾，
    命名規則跟現有手動搬過去的檔案一致（<model_name>_best.pt）——「AI 影像驗證
    中心」的模型下拉選單（/api/models，掃 models/*.pt）只看這個資料夾，訓練
    輸出本身不會自動出現在那裡，要手動搬過去才能被選到。"""
    src = PROJECT_ROOT / "runs" / "train" / model_name / "weights" / "best.pt"
    if not src.exists():
        return {"status": "error", "message": f"找不到訓練產出的權重檔：{src}"}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / f"{model_name}_best.pt"
    shutil.copy2(src, dest)
    return {"status": "success", "path": str(dest)}


def _extract_agent_report(agent_name: str, new_messages: list) -> dict:
    """從子 Agent 這一輪新增的訊息裡找出 report_to_manager 的結構化回報。

    正常情況一定找得到（Prompt 要求子 Agent 一定要呼叫它）；找不到代表子 Agent
    沒有照規則結束，這時退化成用它最後一句文字包一份 status="partial" 的報告，
    讓 Manager 至少還有東西可以看，不會完全拿不到任何結構化資訊。"""
    for m in reversed(new_messages):
        if isinstance(m, ToolMessage) and m.name == "report_to_manager":
            data = _parse_tool_content(m.content)
            return {
                "agent": agent_name,
                "status": data.get("status", "partial"),
                "reason": data.get("reason", ""),
                "suggestion": data.get("suggestion", ""),
                "reason_code": data.get("reason_code", ""),
            }

    text = ""
    for m in reversed(new_messages):
        if not isinstance(m, ToolMessage) and getattr(m, "content", None):
            text = _extract_text(m.content)
            break
    return {"agent": agent_name, "status": "partial", "reason": text, "suggestion": "", "reason_code": ""}


def _plan_to_state(plan: Plan) -> dict:
    steps = [
        {"id": s.id, "description": s.description, "agent": s.agent, "is_training": s.is_training, "status": "pending"}
        for s in plan.steps
    ]
    return {"goal": plan.goal, "steps": steps}


def _summarize_plan(plan: Optional[dict]) -> str:
    if not plan or not plan.get("steps"):
        return ""
    lines = [f"目標：{plan.get('goal', '')}"]
    for s in plan["steps"]:
        lines.append(f"- [{s.get('status')}] {s.get('description')}（負責：{s.get('agent')}）")
    return "\n".join(lines)


def _mark_step_in_progress(plan: Optional[dict], target_agent: str) -> Optional[dict]:
    """派工給某個子 Agent 時，把 Plan 裡屬於它、還沒開始的第一個 Step 標成 in_progress。

    Plan 只是輔助脈絡，不是硬性合約——找不到對應 Step 就原樣回傳，不強制要求
    每次派工都對得上計畫（例如 Decision Engine 臨時決定做計畫外的事）。"""
    if not plan or not plan.get("steps"):
        return plan
    steps = [dict(s) for s in plan["steps"]]
    for s in steps:
        if s.get("agent") == target_agent and s.get("status") == "pending":
            s["status"] = "in_progress"
            break
    return {**plan, "steps": steps}


def build_multi_agent(chat_model):
    """組裝 Planner + Manager（Decision Engine）+ Observer + 三個子 Agent 的
    Multi-Agent Graph。對外的呼叫方式跟舊版一樣：
    graph.invoke({"messages": [HumanMessage(...)]})，方便 main.py 直接替換。
    """
    data_agent = _build_sub_agent(chat_model, DATA_TOOLS, DATA_AGENT_PROMPT)
    model_agent = _build_sub_agent(chat_model, MODEL_TOOLS, MODEL_AGENT_PROMPT)
    evaluation_agent = _build_sub_agent(chat_model, EVAL_TOOLS, EVALUATION_AGENT_PROMPT)
    # method="function_calling" 明確指定用 Tool-Calling 做結構化輸出——Groq 的
    # llama-3.3 等模型不支援 with_structured_output 預設用的 json_schema
    # response_format（實測會直接回 400），但都支援 function calling。
    # include_raw=True 讓 invoke() 多回傳原始 AIMessage（.invoke 預設只給解析後的
    # Pydantic 物件），這樣才能撈到 usage_metadata 算 token 用量——結構化決策本身
    # 邏輯不變，只是多留一份原始訊息。
    decision_model = chat_model.with_structured_output(Decision, method="function_calling", include_raw=True)
    planner_model = chat_model.with_structured_output(Plan, method="function_calling", include_raw=True)

    def planner_node(state: SupervisorState) -> dict:
        context = [SystemMessage(content=PLANNER_SYSTEM_PROMPT)] + list(state["messages"])
        try:
            result = _invoke_with_retry(lambda: planner_model.invoke(context))
        except Exception:
            # 規劃失敗就不擋流程——Plan 本來就只是輔助脈絡，不是硬性合約（見
            # _mark_step_in_progress 的說明），沒有 Plan，Manager 一樣能只靠對話
            # 紀錄做決策，好過整段對話直接因為例外中斷。
            return {"plan": None, "stall_count": 0, "evaluation": None}
        plan = result["parsed"]
        # 重新規劃後一定要把舊的 evaluation 清掉，不然 decision_engine 下一輪還是會讀到
        # 舊的 verdict="plan_invalid"，繞回 planner -> manager -> planner 一直重複，
        # 直到 MAX_ROUNDS 才停下來。
        return {
            "plan": _plan_to_state(plan),
            "stall_count": 0,
            "evaluation": None,
            "usage": _usage_from_message(result["raw"]),
        }

    def observer_node(state: SupervisorState) -> dict:
        """讀最新一份 AgentReport，判讀它對 Plan 來說代表什麼，寫成結構化 Evaluation。
        全部用確定性規則，不呼叫 LLM——這一步只負責「解讀發生了什麼」，不決定下一步。"""
        report = state.get("last_report") or {}
        prev_evaluation = state.get("evaluation") or {}
        plan = state.get("plan") or {"goal": "", "steps": []}
        steps = [dict(s) for s in plan.get("steps", [])]

        agent_name = report.get("agent", "")
        status = report.get("status", "")
        reason_code = report.get("reason_code") or ""

        if status == "success":
            verdict = "on_track"
        elif reason_code in GATE_ROUTING:
            verdict = "step_failed"
        else:
            # blocked/failed/partial 但不是已知 Gate 類型，代表 Decision Engine
            # 的固定規則幫不上忙，需要 LLM 判斷——先標成 ambiguous，讓 Manager 知道
            # 這不是能靠規則處理的已知模式。
            verdict = "step_ambiguous"

        same_signal_as_last_round = (
            verdict != "on_track"
            and prev_evaluation.get("verdict") == verdict
            and prev_evaluation.get("reason_code", "") == reason_code
        )
        progress = not same_signal_as_last_round

        stall_count = state.get("stall_count", 0)
        stall_count = stall_count + 1 if not progress else 0

        # 矛盾檢查：子 Agent 自報成功，卻同時帶著已知的失敗代碼——這種情況不該發生，
        # 代表子 Agent 沒有照 Prompt 規則回報，先標記起來讓 Manager 有機會留意。
        contradiction = status == "success" and bool(reason_code)

        if verdict != "on_track" and stall_count >= STALL_ROUNDS_BEFORE_REPLAN:
            verdict = "plan_invalid"
            stall_count = 0

        current_step = next(
            (s for s in steps if s.get("agent") == agent_name and s.get("status") == "in_progress"),
            None,
        )
        if current_step:
            if verdict == "on_track":
                current_step["status"] = "done"
            elif verdict in ("step_failed", "plan_invalid"):
                current_step["status"] = "failed"

        evaluation = {
            "step_id": current_step.get("id") if current_step else None,
            "verdict": verdict,
            "progress_since_last_round": progress,
            "contradiction": contradiction,
            "reason_code": reason_code,
            "evidence": [
                f"{agent_name or '子 Agent'} 回報 status={status}" + (f" reason_code={reason_code}" if reason_code else "")
            ],
            "notes": report.get("reason", ""),
        }

        return {"evaluation": evaluation, "plan": {**plan, "steps": steps}, "stall_count": stall_count}

    def train_confirm_gate(state: SupervisorState) -> dict:
        """manager 決定路由到 model_agent 時，先經過這一關：不是訓練步驟就直接
        放行；是訓練步驟就用 interrupt() 暫停，等待前端使用者確認後才真的進
        model_agent（訓練耗時、耗運算資源，不該讓 Agent 自己說做就做）。

        train_confirmed 記錄「這個任務裡使用者已經確認過一次」——同一個任務裡
        Gate 失敗（dataset_not_found/health_gate_failed）重試、或 Observer 判定
        plan_invalid 繞回 Planner 後再度指派 model_agent，都不該讓使用者對同一次
        訓練意圖重複按確認；只有使用者發送新的一則訊息（stream_trace 會重置這個
        欄位）才要重新問一次。"""
        plan = state.get("plan") or {"steps": []}
        step = next(
            (s for s in plan.get("steps", []) if s.get("agent") == "model_agent" and s.get("status") == "in_progress"),
            None,
        )
        if not (step or {}).get("is_training"):
            return {"next": "model_agent"}

        if state.get("train_confirmed"):
            return {"next": "model_agent"}

        resume = interrupt({
            "reason": "即將啟動模型訓練，這是高成本、耗時的操作，請確認是否繼續。",
            "step_description": step.get("description", ""),
        })
        if isinstance(resume, dict) and resume.get("approved"):
            return {"next": "model_agent", "train_confirmed": True}
        return {
            "next": "FINISH",
            "messages": [AIMessage(content="使用者取消了這次訓練，已停止。")],
        }

    def decision_engine(state: SupervisorState) -> dict:
        rounds = state.get("rounds", 0) + 1
        retry_counters = dict(state.get("retry_counters", {}))

        if rounds > MAX_ROUNDS:
            return {
                "next": "FINISH",
                "rounds": rounds,
                "messages": [AIMessage(content="已經嘗試多輪仍未完成，請確認需求或稍後再協助確認資料/模型狀態。")],
            }

        evaluation = state.get("evaluation") or {}
        if evaluation.get("verdict") == "plan_invalid":
            # Observer 判定目前的 Plan 已經卡住、沒有進展，繞回 Planner 重新規劃，
            # 不要讓 LLM fallback 拿著同一份失效的 Plan 一直瞎猜。連續重新規劃太多次
            # 還是卡在同樣的問題，代表不是「換個計畫就能解決」，直接停下來讓使用者
            # 知道，不要把 MAX_ROUNDS 的預算都耗在一直重新規劃上。
            replan_count = state.get("replan_count", 0)
            if replan_count >= MAX_REPLANS:
                return {
                    "next": "FINISH",
                    "rounds": rounds,
                    "retry_counters": retry_counters,
                    "messages": [AIMessage(
                        content=f"已經重新規劃 {replan_count} 次仍卡在同樣的問題，"
                                f"需要你確認需求或提供更多資訊後再試一次。"
                    )],
                }
            return {
                "next": "planner",
                "rounds": rounds,
                "retry_counters": retry_counters,
                "replan_count": replan_count + 1,
            }

        last_report = state.get("last_report") or {}
        reason_code = last_report.get("reason_code") or ""

        # 已知 Gate 失敗類型：固定規則直接決定，不問 LLM。
        if reason_code in GATE_ROUTING:
            attempts = retry_counters.get(reason_code, 0)
            limit = RETRY_LIMITS[reason_code]
            if attempts < limit:
                retry_counters[reason_code] = attempts + 1
                target = GATE_ROUTING[reason_code]
                plan = _mark_step_in_progress(state.get("plan"), target)
                return {"next": target, "rounds": rounds, "retry_counters": retry_counters, "plan": plan}
            return {
                "next": "FINISH",
                "rounds": rounds,
                "retry_counters": retry_counters,
                "messages": [AIMessage(
                    content=f"「{reason_code}」已經重試 {limit} 次仍未解決：{last_report.get('reason', '')}"
                            f"，需要你確認資料或需求後再試一次。"
                )],
            }

        # 規則判斷不出來（新請求、或未知的失敗類型），交給 LLM 做結構化決策，
        # 附上目前的 Plan 進度摘要，讓它記得原本的目標做到哪了，不是只看對話紀錄猜。
        context = [SystemMessage(content=MANAGER_SYSTEM_PROMPT)]
        plan_summary = _summarize_plan(state.get("plan"))
        if plan_summary:
            context.append(SystemMessage(content=f"目前的計畫進度：\n{plan_summary}"))
        dataset_names = _existing_dataset_names()
        context.append(SystemMessage(
            content=f"目前本機已有的資料集：{', '.join(dataset_names) if dataset_names else '（無）'}"
        ))
        context += list(state["messages"])
        try:
            result = _invoke_with_retry(lambda: decision_model.invoke(context))
        except Exception as e:
            # 重試後仍失敗：直接結束這輪並誠實告知使用者，好過讓例外炸穿
            # graph.stream()、讓整段對話（包含前面已經跑完的步驟）都消失。
            return {
                "next": "FINISH",
                "rounds": rounds,
                "retry_counters": retry_counters,
                "messages": [AIMessage(content=f"決策時發生錯誤，請稍後再試一次：{e}")],
            }
        decision = result["parsed"]
        usage = _usage_from_message(result["raw"])

        if decision.action == "call_agent" and decision.target_agent:
            plan = _mark_step_in_progress(state.get("plan"), decision.target_agent)
            return {"next": decision.target_agent, "rounds": rounds, "retry_counters": retry_counters, "plan": plan, "usage": usage}

        reply_text = decision.final_reply or decision.rationale
        return {
            "next": "FINISH",
            "rounds": rounds,
            "retry_counters": retry_counters,
            "messages": [AIMessage(content=reply_text)],
            "usage": usage,
        }

    def _run_sub_agent(agent_name, sub_agent, state: SupervisorState) -> dict:
        input_messages = state["messages"]
        result = sub_agent.invoke({"messages": input_messages})
        new_messages = result["messages"][len(input_messages):]
        report = _extract_agent_report(agent_name, new_messages)
        return {"messages": new_messages, "last_report": report}

    def run_data_agent(state: SupervisorState) -> dict:
        return _run_sub_agent("data_agent", data_agent, state)

    def run_model_agent(state: SupervisorState) -> dict:
        result = _run_sub_agent("model_agent", model_agent, state)
        trained_model = _extract_trained_model_name(result["messages"])
        if trained_model:
            result["last_trained_model"] = trained_model
        return result

    def save_offer_gate(state: SupervisorState) -> dict:
        """任務結束前的最後一關：這一次任務裡如果有成功訓練出新模型，不要自動
        存進模型庫——先讓 Manager 剛剛的分析（mAP/loss/有無過擬合）講給使用者
        聽，再用 interrupt() 問使用者要不要真的存，不是每次隨手測試訓練都塞進
        驗證中心的下拉選單。同一個任務裡已經問過就不再問第二次（例如任務結束
        前 Manager 又多繞了一輪）。"""
        model_name = state.get("last_trained_model")
        if not model_name or state.get("model_save_offered"):
            return {}

        resume = interrupt({
            "reason": f"模型「{model_name}」已訓練完成，要把它加入模型庫（驗證中心下拉選單可選）嗎？",
            "model_name": model_name,
        })
        if isinstance(resume, dict) and resume.get("approved"):
            result = _add_model_to_library(model_name)
            if result.get("status") == "success":
                note = f"已將模型「{model_name}」加入模型庫，可以在 AI 影像驗證中心的下拉選單選到它了。"
            else:
                note = f"存入模型庫失敗：{result.get('message')}"
        else:
            note = f"好的，模型「{model_name}」先不存入模型庫，訓練結果仍留在 runs/train/{model_name} 底下。"
        return {"model_save_offered": True, "messages": [AIMessage(content=note)]}

    def run_evaluation_agent(state: SupervisorState) -> dict:
        return _run_sub_agent("evaluation_agent", evaluation_agent, state)

    graph = StateGraph(SupervisorState)
    graph.add_node("planner", planner_node)
    graph.add_node("manager", decision_engine)
    graph.add_node("observer", observer_node)
    graph.add_node("train_confirm_gate", train_confirm_gate)
    graph.add_node("save_offer_gate", save_offer_gate)
    graph.add_node("data_agent", run_data_agent)
    graph.add_node("model_agent", run_model_agent)
    graph.add_node("evaluation_agent", run_evaluation_agent)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "manager")
    graph.add_conditional_edges(
        "manager",
        lambda s: s["next"],
        {
            "data_agent": "data_agent",
            "model_agent": "train_confirm_gate",
            "evaluation_agent": "evaluation_agent",
            "planner": "planner",
            "FINISH": "save_offer_gate",
        },
    )
    graph.add_conditional_edges(
        "train_confirm_gate",
        lambda s: s["next"],
        {"model_agent": "model_agent", "FINISH": "save_offer_gate"},
    )
    graph.add_edge("save_offer_gate", END)
    graph.add_edge("data_agent", "observer")
    graph.add_edge("model_agent", "observer")
    graph.add_edge("evaluation_agent", "observer")
    graph.add_edge("observer", "manager")
    # checkpointer 讓同一個 thread_id 的多輪對話記得住 messages（歷史對話內容），
    # 是 /api/agent/chat 做 session 記憶的前提——stream_trace 呼叫時另外會重置
    # rounds/retry_counters/plan 等「單次任務用的計數器」，不讓它們跨任務累積。
    #
    # 用 SqliteSaver 而不是 MemorySaver：訓練前的 interrupt() 可能暫停好幾分鐘
    # 才等到使用者確認，伺服器重啟（重新部署、當機）不該讓這份對話記憶直接消失。
    # SqliteSaver 建構子要求傳入 sqlite3.Connection（SqliteSaver.from_conn_string
    # 回的是 context manager，不適合這種要活整個 process 生命週期的用法）；
    # check_same_thread=False 是因為 FastAPI 的 StreamingResponse 可能在不同
    # worker thread 呼叫，同一個連線要能跨 thread 使用。
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(CHECKPOINT_DB_PATH), check_same_thread=False)
    return graph.compile(checkpointer=SqliteSaver(conn))


def stream_trace(graph, thread_id: str, user_message: Optional[str] = None, resume_value: Optional[dict] = None):
    """觀測層：用 graph.stream() 逐步 yield 每個節點做了什麼（Planner 的計畫、
    Manager 的路由決策、子 Agent 呼叫了哪些 Tool、回報了什麼結果、Observer
    的判讀），給 /api/agent/chat 邊跑邊往前端推，這樣使用者才「看得到」Agent
    跑到哪一步，而不是等全部跑完才一次性看到決策過程。

    user_message／resume_value 兩者互斥：user_message 是使用者發的新訊息，走
    一般流程；resume_value 是 train_confirm_gate 這類 interrupt() 暫停後，
    使用者確認/取消的回應（例如 {"approved": True}），改用 Command(resume=...)
    從暫停處接著跑，不會重新建構 initial_input。圖跑到 interrupt() 時會 yield
    一個 type="interrupt" 的 entry 然後直接結束（不會再多送一個 final entry），
    等呼叫端帶著 resume_value 再呼叫一次這個函式才會繼續。

    中途任何 node 拋例外都會被攔下來、yield 一個 error entry 再結束——這樣
    已經跑出來的步驟不會因為後面失敗就整包消失。

    thread_id 對應同一個 checkpointer 裡的一份記憶——同一個 thread_id 的
    messages 會跟之前的對話合併，但 rounds/retry_counters/stall_count/
    plan/evaluation/last_report/train_confirmed/last_trained_model/
    model_save_offered 每次新訊息都明確重置，這些欄位
    是「這一次任務」用的保險絲，不該延續到下一次使用者發的新任務（resume 不算
    新任務，不會重置這些欄位，接著用暫停當下的狀態繼續——train_confirmed 也是
    因為這樣才能在同一個任務裡重試訓練時不用重複跳出確認畫面）。"""
    all_messages = [HumanMessage(content=user_message)] if user_message is not None else []
    seed_len = len(all_messages)
    config = {"configurable": {"thread_id": thread_id}}
    interrupted = False

    try:
        if resume_value is not None:
            stream_input = Command(resume=resume_value)
        else:
            stream_input = {
                "messages": list(all_messages),
                "rounds": 0,
                "retry_counters": {},
                "stall_count": 0,
                "plan": None,
                "evaluation": None,
                "last_report": None,
                "replan_count": 0,
                "train_confirmed": False,
                "last_trained_model": None,
                "model_save_offered": False,
            }
        total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        llm_calls = 0

        def _add_usage(usage):
            nonlocal llm_calls
            if not usage:
                return
            for k in total_usage:
                total_usage[k] += usage[k]
            llm_calls += 1

        for step in graph.stream(stream_input, config=config, stream_mode="updates"):
            if "__interrupt__" in step:
                # interrupt() 暫停時這一個 chunk 的形狀跟平常的 {node_name: node_output}
                # 不一樣，是 {"__interrupt__": (Interrupt(value=...),)}——這裡是暫停點，
                # 之後不會再有新 chunk，呼叫端要帶著使用者的回應再呼叫一次才會繼續。
                interrupt_obj = step["__interrupt__"][0]
                yield {"type": "interrupt", "payload": interrupt_obj.value}
                interrupted = True
                continue
            for node_name, node_output in step.items():
                if node_output is None:
                    # 節點回傳 {}（沒有實際要更新的欄位，例如 save_offer_gate
                    # 判定這次沒有新訓練出的模型可問）時，LangGraph 在
                    # stream_mode="updates" 這一格會給 None 而不是 {}，直接對
                    # None 做 "in" 判斷會整段對話都炸掉，所以先跳過這個節點，
                    # 不產生任何 trace entry（反正它本來就沒有東西要回報）。
                    continue
                entry = {"type": "step", "node": node_name}
                if "next" in node_output:
                    entry["decision"] = node_output["next"]
                if "last_report" in node_output:
                    entry["report"] = node_output["last_report"]
                if "evaluation" in node_output:
                    entry["evaluation"] = node_output["evaluation"]
                if "plan" in node_output and node_output["plan"]:
                    entry["plan"] = node_output["plan"]

                # Planner/Decision Engine 呼叫 LLM 的 usage 直接附在 node_output 上
                # （見 planner_node/decision_engine），子 Agent 的 usage 則要從它
                # 這輪新增的每一則 AIMessage 各自撈（一次 ReAct 迴圈可能呼叫多次）。
                node_usage = node_output.get("usage")
                _add_usage(node_usage)

                new_messages = node_output.get("messages") or []
                all_messages.extend(new_messages)

                tool_calls = []
                last_text = None
                sub_agent_usage = None
                for m in new_messages:
                    calls = getattr(m, "tool_calls", None)
                    if calls:
                        tool_calls.extend([{"tool": c["name"], "args": c["args"]} for c in calls])
                    elif isinstance(m, ToolMessage):
                        tool_calls.append({"tool_result": m.name, "output": m.content})
                    elif getattr(m, "content", None):
                        last_text = _extract_text(m.content)

                    msg_usage = _usage_from_message(m)
                    if msg_usage:
                        _add_usage(msg_usage)
                        sub_agent_usage = sub_agent_usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                        for k in sub_agent_usage:
                            sub_agent_usage[k] += msg_usage[k]

                if tool_calls:
                    entry["tool_calls"] = tool_calls
                if last_text:
                    entry["text"] = last_text
                if node_usage or sub_agent_usage:
                    entry["usage"] = node_usage or sub_agent_usage
                yield entry
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {"type": "error", "message": str(e)}
        return

    if interrupted:
        return

    reply = _extract_text(all_messages[-1].content) if len(all_messages) > seed_len else ""
    yield {"type": "final", "reply": reply, "usage": total_usage, "llm_calls": llm_calls}
