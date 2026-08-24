"""Deep Agent 模式（聊天視窗的第三個模式）。

跟 multi_agent.py（Manager 模式）並存、完全獨立，不是取代它。目的是拿
LangChain 生態系的 deepagents preset 重做一套「能力對等」的協調層，然後跟手寫的
Manager 模式 A/B 比較穩定性與 token 消耗——所以這一版刻意不加 Manager 模式沒有的
功能（檔案系統 scratchpad、跨 session 長期記憶那些是下一階段的事）。

跟 Manager 模式的對應關係：
- Planner（規劃）        -> deepagents 內建的 write_todos
- Decision Engine（路由）-> 主 agent 自己用 task 工具決定派給哪個 subagent
- 三個子 Agent            -> 宣告式 SubAgent（data/model/evaluation）
- report_to_manager      -> 不需要：task 工具直接把 subagent 的最後一則訊息當回傳值
- train_confirm_gate     -> interrupt_on={"skill_train": True}（tool 層級的 HITL）
- SupervisorState 的 20+ 欄位 -> DeepAgentState（deepagents 自己管）

刻意不做的兩件事，理由都是「不要動到 Manager 模式的程式碼路徑」：
1. 不共用 /api/agent/chat，另開 /api/agent/deep/* 端點。
2. 不共用 checkpoint DB——Manager 模式的 state 是 SupervisorState，這裡是
   DeepAgentState，兩種 schema 用同一個 thread_id 寫進同一個 sqlite 會在 resume
   時讀到不匹配的欄位，而且症狀通常不是乾脆報錯，是靜默的錯誤行為。

Tool／Skill 兩層完全重用（agent_tools.py / skills.py），只有協調層是新的——
這也是為什麼這裡直接從 agent_tools/skills import，不從 multi_agent import 工具清單：
兩個模式各自宣告自己要用哪些工具，改一邊不會波及另一邊。
"""
import hashlib
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from deepagents import create_deep_agent

SERVER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_DIR.parent
DEEP_CHECKPOINT_DB_PATH = PROJECT_ROOT / "data" / "deep_agent_checkpoints.sqlite"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent_tools import (
    check_dataset_health,
    download_roboflow_dataset,
    import_images_to_dataset,
    prepare_roboflow_augmentation,
    search_roboflow_universe,
)
from skills import skill_train, skill_infer, skill_eval

# _extract_text／_usage_from_message 是「怎麼從 LangChain 訊息裡撈文字跟 token 數」
# 這種跟 Manager 邏輯無關的通用工具，重用而不是複製一份（專案裡 multi_agent.py 也
# 是這樣從 agent_tools.py import _count_images／_sanitize_name 的）。
from multi_agent import _extract_text, _usage_from_message

DATA_TOOLS = [
    check_dataset_health,
    import_images_to_dataset,
    download_roboflow_dataset,
    search_roboflow_universe,
    prepare_roboflow_augmentation,
]
MODEL_TOOLS = [skill_train, skill_infer]
EVAL_TOOLS = [skill_eval]

# 訓練是唯一會長時間佔用機器、而且會寫出檔案的動作，跟 Manager 模式的
# train_confirm_gate 對齊：跑之前一定要人工確認。工具名稱就是 @tool 函式名。
TRAIN_TOOL_NAME = "skill_train"

# create_deep_agent 內建 FilesystemMiddleware 的 7 個工具，這個模式一個都用不到：
# 它的職責是「協調三個 subagent 去操作資料集與模型」，不需要讀寫檔案，execute 在
# 預設的 StateBackend 上本來就只會回傳錯誤訊息。
#
# 為什麼要主動拿掉：實測第一次呼叫的 input 就有 6466 tokens，而這 7 個工具的 schema
# 佔了約 2547（read_file 611、execute 831、grep 344…），每一次 LLM 呼叫都要重送一遍。
# 而且模型真的會被它們帶偏——第一個問題「檢查一下 車 這個資料集」時，模型的第一個
# 動作是 ls({"path": "/"})，回傳空陣列，整整浪費一次 LLM 往返。
#
# 之後階段 B 要做「讀 runs/train/*/results.csv 自己比較多次實驗」時會把這些工具加回來
# （搭配 FilesystemBackend 與 FilesystemPermission 限制根目錄），現在拿掉只是因為
# 階段 A 的目標是跟 Manager 模式能力對等，用不到的東西不該讓模型每次都看到。
EXCLUDED_BUILTIN_TOOLS = frozenset({
    "ls", "read_file", "write_file", "edit_file", "glob", "grep", "execute",
})


class DropToolsMiddleware(AgentMiddleware):
    """把指定工具在「送給模型之前」從請求裡移除，省掉它們的 schema token。

    deepagents 官方的做法是 register_harness_profile() 註冊帶 excluded_tools 的
    HarnessProfile，但那是寫進全域 registry 的 beta API，還要自己推算 provider key
    （pre-built model 走的是 provider:identifier 的查表邏輯）。這裡用公開的
    middleware 介面做同一件事——wrap_model_call 裡 request.override(tools=...)，
    跟 deepagents 內部的 _ToolExclusionMiddleware 完全一樣的機制——作用範圍只限
    我們自己建的這個 agent，不會影響同一個 process 裡其他人建的 deep agent。

    注意：頂層 middleware 只作用在主 agent 上，subagent 會自己組一份 middleware
    清單，所以三個 SubAgent 的 middleware 欄位也要各自掛一份（見 _build_subagents）。
    """

    def __init__(self, excluded=EXCLUDED_BUILTIN_TOOLS):
        super().__init__()
        self._excluded = frozenset(excluded)

    def wrap_model_call(self, request, handler):
        if self._excluded:
            kept = [t for t in request.tools if getattr(t, "name", None) not in self._excluded]
            request = request.override(tools=kept)
        return handler(request)


DEEP_AGENT_SYSTEM_PROMPT = """你是 AI Detector 系統的協調者，負責把使用者的需求拆解成步驟，
再交給專責的 subagent 執行。你自己不直接呼叫任何訓練／推論／資料集工具。

你有三個 subagent，用 task 工具派工：
- data_agent：資料集健康檢查、匯入本機圖片、去 Roboflow 搜尋與準備補強提案
- model_agent：訓練模型（skill_train）、用模型做推論（skill_infer）
- evaluation_agent：評估已訓練模型的效果（skill_eval，讀 mAP/loss）

你「沒有」檔案系統工具（ls／read_file／write_file／glob／grep／execute 都已停用），
所有實際操作都必須透過 task 交給 subagent。不要試圖自己去看目錄或讀檔案。

工作方式：
1. 需求需要兩個步驟以上時，先用 write_todos 列出計畫，每完成一步就更新狀態。
   單一步驟的請求（例如「檢查一下資料集」）不用列 todo，直接派工，也不要先探查環境。
2. 一次只派一個 subagent，看它回報什麼再決定下一步，不要一次把三個都叫起來。
3. subagent 回報卡關時，判斷是「換個做法還能繼續」還是「需要使用者補資訊」。
   同一個 subagent 用同樣的方式連續失敗兩次就不要再試第三次，改成把狀況講給使用者。

【「評估」有兩種，不要混在一起講】
- 資料集健康評估：data_agent 的 check_dataset_health，訓練「之前」就能做，看的是
  標記覆蓋率、類別平衡、圖片與標籤數，跟有沒有訓練過無關。
- 模型效能評估：evaluation_agent 的 skill_eval，只能在訓練「之後」做，看 mAP/loss。
使用者問「這個資料集適不適合訓練」「資料集狀況如何」時指的一律是前者，要派
data_agent 去拿實際數字回答，不可以用「還沒訓練所以沒有評估指標」帶過——那答的
根本不是他問的問題。

【訓練的注意事項】
- 這台機器是 CPU-only、記憶體有限。使用者沒指定參數時採低資源組合：
  variant=yolov9s.pt、epochs=20、imgsz=320、batch=4、model_name=exp。
- 要訓練但使用者沒指名資料集、而本機有多個資料集時，先問他要用哪一個，不要自己挑。
  本機剛好只有一個資料集時直接用，不用多問。
- skill_train 執行前系統會自動跳出確認視窗給使用者，你不需要自己再問一次
  「要開始訓練嗎」，直接派 model_agent 去做就好。

【回覆使用者】
講結果要帶實際數字（模型名稱、mAP50、mAP50-95、最終 loss、健康度分數），
只寫「訓練完成」「評估完成」使用者什麼都看不到。
"""

DATA_SUBAGENT_PROMPT = """你是 Data Agent，只負責資料集相關的任務，依下面順序判斷該用哪個工具：
1. 先呼叫 check_dataset_health 確認本機這個資料集是否已經有圖片——已經有就不用再匯入
   或下載，直接回報現況即可。注意：健康分數低如果只是「類別不平衡（balance 低）」而
   標記覆蓋率（coverage）夠，不算卡關；只有「覆蓋率低（很多圖沒標記）」或「根本沒有
   圖片」才算真正的問題。
   ——但這條「已經有圖片就停手」不適用於使用者主動要求去外部找資料的情況：只要使用者
   的意思是要補強／擴充／去外面抓圖片，不管本機健康分數多高都要繼續往第 4 點走，
   不可以用「資料集已經很健康、無需補強」當理由停下來。要不要補強是使用者的決定。
2. 使用者提到本機／已上傳有現成圖片路徑（例如 data/uploads/ 或 temp_verify/），且本機
   資料集還沒有圖片時，呼叫 import_images_to_dataset。這個工具只搬圖片、不會產生標記
   檔，匯入後標記率是 0% 是正常現象，不用因此判定失敗。
3. 本機、使用者提供的路徑都沒有圖片可用，而且確實要建立全新資料集時，才呼叫
   download_roboflow_dataset。先用 search_roboflow_universe 取得真實候選，
   不要自己猜 workspace/project slug。
4. 只要使用者的意思是「去外面找圖片／抓資料／補強／擴充這個資料集」，就直接呼叫
   prepare_roboflow_augmentation——不需要出現「Roboflow」這個字，也不需要分數低才做。
   絕對不要只在回覆裡提到工具名稱、叫使用者自己去用：你就是負責呼叫這個工具的人。
   它會自行搜尋候選、下載到暫存區，檢查類別對應、重複圖片與可用標註，完全不會修改
   正式資料集。回傳 status="proposal_ready" 時，把 proposal_id、來源、可用圖片數、
   類別對應與風險完整寫進你的結論。不要呼叫任何正式合併工具。
5. 多類別資料集只能接受名稱能明確對應的類別；沒有對應的外部類別必須跳過，不能把所有
   標籤硬改成 class 0。

任務結束時，用一段文字回報結果，必須包含：這次做了什麼、實際數字（圖片數、標記數、
覆蓋率、平衡度）、以及結論是「完成」還是「卡住」。卡住的話寫清楚卡在哪裡——本機、
上傳路徑、Roboflow 都找不到資料就明講「找不到這個資料集的任何可用資料」。

你不負責訓練、推論、評估模型。"""

MODEL_SUBAGENT_PROMPT = """你是 Model Agent，只負責模型的訓練與推論，用兩個 Skill 完成：
- skill_train：完整訓練流程，內部已經自動檢查資料集是否存在、標記覆蓋率是否足夠
  （類別不平衡只會附一個 balance_warning、不會擋訓練），你不用自己再檢查一次。
- skill_infer：完整推論流程，內部已經自動選好要用哪個模型權重。

呼叫 skill_train 前系統會自動暫停、跳出確認視窗給使用者，這是正常流程，你不用自己
先問一次「要開始訓練嗎」。使用者在確認視窗按取消時，skill_train 會回一則說明被拒絕
的訊息——這時不要重試，直接回報「使用者取消了這次訓練」。

任務結束時用一段文字回報，必須包含實際數字與明確結論：
- skill_train 回傳 status="dataset_not_found" → 回報「找不到資料集」，要 Data Agent 處理
- status="health_gate_failed" → 回報「資料集健康檢查沒通過」並附上健康度報告數字
- status="stopped" → 這是使用者主動終止，不是失敗，回報「使用者已取消這次訓練」，
  不要自己重試或建議繼續訓練
- status="already_training" → 回報「已經有另一個訓練在進行中」
- 訓練成功 → 回報模型名稱、mAP50、最終 loss、實際跑了幾個 epoch
- 推論成功 → 回報偵測到什麼、幾個目標

你不負責資料集健康檢查、下載資料、評估模型效果。"""

EVAL_SUBAGENT_PROMPT = """你是 Evaluation Agent，只負責用 skill_eval 評估已經訓練完成的模型：
讀取 mAP/Loss 等指標，判斷有沒有過擬合或資料傾斜，並把結果解釋清楚。

評估只讀訓練紀錄，不需要使用者提供任何圖片——不要向使用者要圖片。使用者說「剛剛
訓練完的模型」「最新的模型」時，直接呼叫 skill_eval 並把 model_name 留空，它會自己
解析成最近一次有訓練紀錄的實驗；不要自己猜一個名稱（例如 exp），猜錯就會評估到
別的模型。skill_eval 回傳的 resolved_model_name 就是實際評估的模型，解釋結果時要
明確講出這個名稱。

任務結束時回報的內容必須帶上實際數字：resolved_model_name、mAP50、mAP50-95、
最終 loss、gate_passed。只寫「評估完成」等於沒有回報。gate_passed=False 時明講沒通過
門檻，並說明可能原因（資料不足、過擬合等）。

你不負責訓練模型或處理資料集。"""


def _build_subagents(chat_model):
    """三個宣告式 SubAgent。model 明確指定成跟主 agent 同一個 chat_model，
    不讓 deepagents 掉回它預設的 claude-sonnet——這個專案的 .env 沒有
    ANTHROPIC_API_KEY，掉回預設會直接在執行期噴驗證錯誤。"""
    return [
        {
            "name": "data_agent",
            "description": "處理資料集：健康檢查、匯入本機圖片、去 Roboflow 搜尋與準備補強提案。"
                           "需要確認資料集狀況、補資料、擴充資料集時派給它。",
            "system_prompt": DATA_SUBAGENT_PROMPT,
            "tools": DATA_TOOLS,
            "model": chat_model,
            "middleware": [DropToolsMiddleware()],
        },
        {
            "name": "model_agent",
            "description": "訓練模型與用模型做推論。要開始訓練、或要拿圖片測試模型抓不抓得到時派給它。",
            "system_prompt": MODEL_SUBAGENT_PROMPT,
            "tools": MODEL_TOOLS,
            "model": chat_model,
            "middleware": [DropToolsMiddleware()],
        },
        {
            "name": "evaluation_agent",
            "description": "評估已訓練模型的效果（讀 mAP/loss 訓練紀錄，不需要圖片）。"
                           "使用者問模型表現如何、mAP 多少時派給它。",
            "system_prompt": EVAL_SUBAGENT_PROMPT,
            "tools": EVAL_TOOLS,
            "model": chat_model,
            "middleware": [DropToolsMiddleware()],
        },
    ]


def build_deep_agent(chat_model):
    """chat_model 用依賴注入傳進來，跟 build_multi_agent 同一個慣例，方便測試時換假模型。

    checkpointer 用自己的 sqlite 檔（不是 Manager 模式那個 agent_checkpoints.sqlite），
    理由見模組開頭。連線的建立方式跟 multi_agent.py 一致：SqliteSaver 要的是
    sqlite3.Connection，check_same_thread=False 是因為 FastAPI 的 StreamingResponse
    可能在不同 worker thread 上呼叫同一個連線。
    """
    DEEP_CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DEEP_CHECKPOINT_DB_PATH), check_same_thread=False)

    return create_deep_agent(
        model=chat_model,
        system_prompt=DEEP_AGENT_SYSTEM_PROMPT,
        subagents=_build_subagents(chat_model),
        # 頂層設定會被宣告式 SubAgent 繼承，所以 model_agent 裡的 skill_train
        # 也會先暫停等確認，不用在 subagent 上再設一次。
        interrupt_on={TRAIN_TOOL_NAME: True},
        # 使用者 middleware 會插在 base stack（Todo/Filesystem/SubAgent 等注入工具的
        # middleware）之後，所以這裡拿得到、也拿得掉它們注入的工具。
        middleware=[DropToolsMiddleware()],
        checkpointer=SqliteSaver(conn),
    )


def _training_operation_id(interrupt_id: str, args: dict) -> str:
    """給前端確認視窗用的操作編號。做成「同一個中斷 + 同一份參數必定算出同一個值」，
    /api/agent/deep/resume 才能在恢復前重算一次比對，確認使用者按下確認的就是當下
    這一次訓練，不是上一個已經失效的中斷。"""
    payload = json.dumps(args or {}, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha1(f"{interrupt_id}|{payload}".encode("utf-8")).hexdigest()
    return f"deep-{digest[:12]}"


def normalize_interrupt_payload(interrupt_id: str, value) -> dict:
    """把 deepagents/HumanInTheLoopMiddleware 的 HITLRequest 轉成前端確認視窗
    既有的 payload 形狀，這樣 showAgentConfirmation() 一行都不用改就能重用。

    HITLRequest 長這樣：
        {"action_requests": [{"name": ..., "args": {...}, "description": ...}],
         "review_configs":  [{"action_name": ..., "allowed_decisions": [...]}]}
    前端要的是：
        {"action_type": "training", "reason": ..., "operation_id": ...,
         "training_params": {...}}
    """
    requests = (value or {}).get("action_requests") or []
    if not requests:
        # 不是預期的 HITLRequest（例如未來 deepagents 改格式）：原樣帶回去，
        # 前端會走「其他操作」那條分支顯示一個通用確認框，不會整段崩掉。
        return {"interrupt_id": interrupt_id, "raw": value}

    action = requests[0] or {}
    name = action.get("name") or ""
    args = action.get("args") or {}
    payload = {
        "interrupt_id": interrupt_id,
        "tool_name": name,
        "reason": action.get("description") or "此操作需要你的確認。",
    }
    if name == TRAIN_TOOL_NAME:
        payload["action_type"] = "training"
        payload["operation_id"] = _training_operation_id(interrupt_id, args)
        payload["training_params"] = dict(args)
        payload["reason"] = action.get("description") or "即將開始訓練，確認後 Model Agent 只能使用以下參數。"
    return payload


def build_resume_value(approved: bool) -> dict:
    """把前端的「確認／取消」翻成 HumanInTheLoopMiddleware 要的 HITLResponse。

    interrupt_on={"skill_train": True} 的預設 allowed_decisions 含 approve/reject，
    這一版只用這兩個——前端確認視窗只有兩顆按鈕、改不了參數值。edit（讓使用者直接
    修改 tool call 參數）是 deepagents 免費送的能力，但那會需要改前端 UI，留到之後。
    """
    if approved:
        return {"decisions": [{"type": "approve"}]}
    return {"decisions": [{"type": "reject", "message": "使用者在確認視窗選擇取消，不要執行這個操作，也不要重試。"}]}


def stream_deep_trace(graph, thread_id: str, user_message: Optional[str] = None, resume_value: Optional[dict] = None):
    """觀測層，刻意產生跟 multi_agent.stream_trace 完全相同的 SSE 事件形狀
    （status／step／interrupt／error／final），前端的 trace 渲染器與確認視窗才能
    直接重用，不用為第三個模式再寫一套解析邏輯。

    user_message 與 resume_value 互斥，跟 stream_trace 同一個慣例。

    跟 stream_trace 的差別：這裡沒有「單次任務用的計數器」要重置——rounds／
    retry_counters／stall_count／plan 那些是 SupervisorState 特有的保險絲，
    DeepAgentState 的 todo 清單由 deepagents 自己管，每次新訊息進來由主 agent
    自行決定要不要重新規劃。
    """
    config = {"configurable": {"thread_id": thread_id}}
    interrupted = False
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    llm_calls = 0
    last_reply = ""

    def _add_usage(usage):
        nonlocal llm_calls
        if not usage:
            return
        for k in total_usage:
            total_usage[k] += usage[k]
        llm_calls += 1

    try:
        yield {
            "type": "status",
            "stage": "deep_agent",
            "message": "Deep Agent 正在理解需求並規劃步驟…",
        }
        if resume_value is not None:
            stream_input = Command(resume=resume_value)
        else:
            stream_input = {"messages": [HumanMessage(content=user_message)]}

        # subgraphs=True：subagent 是透過 task 工具跑起來的「嵌套子圖」，不開這個參數
        # 的話它內部的 LLM 呼叫完全不會出現在 stream 裡——結果是決策過程只看得到外層的
        # 三個步驟，回報的 token 數嚴重低估（實測外層報 20323，但真正的成本還要加上
        # subagent 自己那一輪 ReAct 迴圈）。使用者看不到真實成本就沒辦法跟 Manager
        # 模式比較，所以這裡把嵌套層一起串出來。
        #
        # 開了之後每個 chunk 從 {node: update} 變成 (namespace, {node: update})，
        # namespace 是空 tuple 代表最外層，非空代表在某個 subagent 裡面。
        for chunk in graph.stream(stream_input, config=config, stream_mode="updates", subgraphs=True):
            if isinstance(chunk, tuple) and len(chunk) == 2:
                namespace, step = chunk
            else:
                namespace, step = (), chunk

            if "__interrupt__" in step:
                interrupt_obj = step["__interrupt__"][0]
                yield {
                    "type": "interrupt",
                    "interrupt_id": interrupt_obj.id,
                    "payload": normalize_interrupt_payload(interrupt_obj.id, interrupt_obj.value),
                }
                interrupted = True
                continue

            # namespace 的元素長得像 "task:<uuid>"，uuid 對使用者沒有意義，只保留
            # 「這一步發生在 subagent 裡」這個資訊，讓決策過程看得出層次。
            depth_prefix = "subagent/" if namespace else ""

            for node_name, node_output in step.items():
                if not node_output:
                    # 跟 stream_trace 同一個坑：節點沒有欄位要更新時，
                    # stream_mode="updates" 這一格可能是 None，直接做 "in" 判斷會炸。
                    continue

                entry = {"type": "step", "node": f"{depth_prefix}{node_name}"}
                new_messages = node_output.get("messages") or []

                tool_calls = []
                last_text = None
                node_usage = None
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
                        node_usage = node_usage or {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                        for k in node_usage:
                            node_usage[k] += msg_usage[k]

                # todo 清單是 deepagents 的規劃產物，形狀跟 Manager 模式的 plan 不同，
                # 但前端的 plan 區塊只需要「有哪些步驟、各自什麼狀態」，所以轉成同一個
                # 鍵名送過去，讓既有的計畫顯示直接沿用。
                todos = node_output.get("todos")
                if todos:
                    entry["plan"] = {
                        "goal": "Deep Agent 的待辦清單",
                        "steps": [
                            {
                                "id": str(i + 1),
                                "description": (t.get("content") if isinstance(t, dict) else str(t)),
                                "status": (t.get("status") if isinstance(t, dict) else None),
                            }
                            for i, t in enumerate(todos)
                        ],
                    }

                if tool_calls:
                    entry["tool_calls"] = tool_calls
                if last_text:
                    entry["text"] = last_text
                    # 給使用者的最終回覆只能取最外層主 agent 的最後一段文字——subagent
                    # 的結論是寫給主 agent 看的中間產物（而且是英文的），拿它當回覆會
                    # 蓋掉主 agent 整理過的版本。
                    if not namespace:
                        last_reply = last_text
                if node_usage:
                    entry["usage"] = node_usage
                yield entry
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {"type": "error", "message": str(e)}
        return

    if interrupted:
        return

    yield {"type": "final", "reply": last_reply, "usage": total_usage, "llm_calls": llm_calls}
