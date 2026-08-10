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
import hashlib
import shutil
import sqlite3
import sys
import uuid
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
WORKSPACE_STATE_PATH = PROJECT_ROOT / "workspace_state.json"
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from agent_tools import (
    DATASET_DIR,
    _count_images,
    apply_roboflow_proposal,
    check_dataset_health,
    discard_roboflow_proposal,
    download_roboflow_dataset,
    import_images_to_dataset,
    prepare_roboflow_augmentation,
    search_roboflow_universe,
)
from skills import BALANCE_WARN_THRESHOLD, COVERAGE_GATE, skill_train, skill_infer, skill_eval

DATA_TOOLS = [
    check_dataset_health,
    import_images_to_dataset,
    download_roboflow_dataset,
    search_roboflow_universe,
    prepare_roboflow_augmentation,
]
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
   或下載，直接回報現況即可。注意：健康分數低如果只是「類別不平衡（balance 低）」而
   標記覆蓋率（coverage）夠，不算卡關，照常回報 status="success"，把不平衡當備註寫進
   reason 即可；只有「覆蓋率低（很多圖沒標記）」或「根本沒有圖片」才算真正的問題。
   ——但這條「已經有圖片就停手」不適用於使用者主動要求去外部找資料的情況：只要使用者
   的意思是要補強／擴充／去外面抓圖片，不管本機健康分數多高、圖片多完整，都要繼續往
   第 4 點走，不可以用「資料集已經很健康、無需補強」當理由停下來。要不要補強是使用者
   的決定，不是你的。
2. 使用者提到本機/已上傳有現成圖片路徑（例如推論頁面暫存上傳的 data/uploads/ 或
   temp_verify/），且本機資料集還沒有圖片時，呼叫 import_images_to_dataset 把這些
   圖片匯入。這個工具只搬圖片、不會產生標記檔，匯入後健康度是 0% 標記率是正常現象，
   不用因此判定失敗。
3. 本機、使用者提供的路徑都沒有圖片可用，而且使用者確實要建立全新資料集時，才呼叫
   download_roboflow_dataset。先用 search_roboflow_universe 取得真實候選，不要猜
   workspace/project slug。
4. 只要使用者的意思是「去外面找圖片／抓資料／補強／擴充這個資料集／多找一些樣本」，
   就直接呼叫 prepare_roboflow_augmentation——不需要出現「Roboflow」這個字，也不需要
   資料集分數低才做。**絕對不要只在回覆裡提到工具名稱、叫使用者自己去用**：你就是負責
   呼叫這個工具的人，只寫出工具名稱等於這一步沒做。它會自行搜尋候選、下載到暫存區，
   檢查實際類別對應、重複圖片和可用標註，完全不會修改正式資料集。回傳
   status="proposal_ready" 時，把 proposal_id、來源、可用圖片數、類別對應與風險完整
   寫進 reason，然後 status="success" 回報 Manager。不要呼叫任何正式合併工具；
   Manager 會把評估結果交給使用者，只有使用者最後確認後才套用。
5. 多類別資料集可以準備 Roboflow 評估，但只能接受名稱能明確對應的類別；沒有對應的
   外部類別必須跳過，不能把所有標籤硬改成 class 0。

完成後一定要呼叫 report_to_manager 回報結果：本機、上傳、Roboflow 都找不到資料就
status="blocked"、reason_code="dataset_not_found"；其他情況成功就 status="success"，
reason_code 留空。

你不負責訓練、推論、評估模型，這些請交給 Model Agent / Evaluation Agent。"""

MODEL_AGENT_PROMPT = """你是 Model Agent，只負責模型的訓練與推論，用兩個 Skill 完成：
- skill_train：完整訓練流程，內部已經自動檢查資料集是否存在、標記覆蓋率是否足夠
  （類別不平衡只會附一個 balance_warning、不會擋訓練），你不用自己再檢查一次。
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

評估只讀訓練紀錄，不需要使用者提供任何圖片——不要向使用者要圖片。使用者說「剛剛
訓練完的模型」「最新的模型」時，直接呼叫 skill_eval 並把 model_name 留空，它會自己
解析成最近一次有訓練紀錄的實驗；不要自己猜一個名稱（例如 exp），猜錯就會評估到
別的模型。skill_eval 回傳的 resolved_model_name 就是實際評估的模型，解釋結果時要
明確講出這個名稱，讓使用者知道看的是哪一個。

回報給 Manager 的 reason 要帶上實際數字（resolved_model_name、mAP50、mAP50-95、
最終 loss、gate_passed），Manager 才有東西可以寫進給使用者看的回覆——只寫「評估
完成」會讓使用者什麼都看不到。

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

使用者只是想知道「模型表現/效能怎麼樣」（例如「剛剛訓練完的模型如何」「評估一下
這個模型」「mAP 多少」）時，只排一個 evaluation_agent 步驟就好——評估是讀訓練紀錄
裡的 mAP/Loss，不需要任何圖片。不要順便加一個「對新圖片做推論」的 model_agent
步驟，那會逼使用者去找圖片給你，而他要的只是評估結果。只有使用者明確說要「用某張
圖/某個資料夾測試、看看抓不抓得到」時才排推論步驟。

評估步驟的描述不用指定模型名稱：使用者說「剛剛訓練完的」「最新的」時，
evaluation_agent 的 skill_eval 會自己解析成最近一次有訓練紀錄的實驗，不要自己猜
一個名稱（猜錯會評估到別的模型）。

指派給 model_agent 的步驟要標 is_training：這個步驟本身要不要啟動一次新的訓練
流程。使用者要求「用剛剛訓練好的模型做推論/驗證」是 is_training=False——句子裡
出現「訓練」兩個字只是在描述要用哪個模型，動作本身是推論/驗證，不是訓練；只有
真的要開始訓練一個新模型時才是 is_training=True。這個欄位會直接決定要不要跳出
「確認是否訓練」的畫面，標錯會讓使用者在做推論時被誤問一次訓練確認。

is_training=True 時也要填 training_params，內容必須反映對話裡已經確認的實際值：
dataset_name、variant、epochs、imgsz、batch、model_name。使用者沒指定的欄位採用低資源
建議值 yolov9s.pt / epochs=20 / imgsz=320 / batch=4 / model_name=exp；不要改回高資源
預設值。這份參數會直接顯示在確認視窗，之後 skill_train 也只能用完全相同的值。

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
（它會透過 Roboflow Universe Search API 搜尋並在暫存區評估），不算「缺資訊」——Plan
已經把某個步驟指派給某個子 Agent 時，直接呼叫那個子 Agent，不要因為使用者沒有提供
這些技術細節就選 ask_user。action="ask_user" 只留給子 Agent 已經回報卡住（例如 blocked/
dataset_not_found 重試上限用完），且需要使用者提供只有人類才知道的資訊
（例如使用者自己要驗證的圖片檔案本身）時才用。

【「評估」有兩種，不要混在一起講】
系統裡「評估」指的是兩件不同的事，使用者問起來的時候要先分清楚是哪一種：
- 資料集健康評估：data_agent 的 check_dataset_health。訓練「之前」就能做，看的是標記
  覆蓋率、類別平衡、圖片與標籤數，跟有沒有訓練過完全無關。
- 模型效能評估：evaluation_agent 的 skill_eval。只能在訓練「之後」做，看的是 mAP/loss。

使用者問「這個資料集有沒有通過評估」「資料集適不適合訓練」「資料集狀況如何」時，指的
一律是資料集健康評估。這時要派 data_agent 去拿 check_dataset_health 的實際數字來回答，
「絕對不可以」用「還沒訓練所以還沒有評估指標」這類話帶過——那是模型效能評估的限制，
答的根本不是使用者問的問題。訓練前系統已經自己跑過一次健康檢查並把數字顯示給使用者，
你要據實引用那些數字，不可以說「尚未評估」。

健康檢查沒通過時，「要不要照樣訓練」的決定權在使用者：把沒通過的項目與你的建議講清楚，
然後等使用者決定，不要自己代替他決定要訓練或不訓練。

評估任務完成時的回覆內容：final_reply 不要只說「評估完成」，要明確寫出實際評估的
模型名稱，以及 Evaluation Agent 報告裡的關鍵指標（mAP50、mAP50-95、最終 loss、有沒有
通過門檻），並簡短解讀好壞。寫完指標後，再主動問一句要不要用實際圖片做推論驗證——
聊天視窗輸入框左邊有 📎 按鈕，也可以直接把圖片拖進對話視窗，使用者不需要自己去找
檔案路徑（不要叫他手動輸入路徑）。這只是邀請，不是 ask_user：評估本身已經完成，
action 仍然是 finish。

訓練資料集的選擇例外：如果要指派給 model_agent 的步驟是訓練，使用者訊息裡沒有
明確指名要用哪一個本機資料集，且下面列出的「目前本機已有的資料集」不只一個，
不要自己隨便挑一個或叫 model_agent 自己猜——用 action="ask_user"，final_reply
列出這些資料集名稱請使用者選一個。如果目前本機剛好只有一個資料集，直接指派給
model_agent 使用它即可，不用多問。

訓練參數不用你問：每一次要開始訓練前，程式都會固定先在對話裡把即將使用的參數
列出來讓使用者用文字修改，之後才跳訓練確認畫面。所以即使使用者沒有指定
epochs/imgsz/batch/模型權重，也不要用 action="ask_user" 去問參數——直接指派
model_agent 就好，參數那一關會自己出現；你多問一次只會讓使用者被問兩遍。
這台機器是 CPU-only、記憶體有限，Planner 填的建議值本來就是低資源組合
（yolov9s.pt、epochs=20、imgsz=320、batch=4）。

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

# 分數不夠時可以嘗試「去 Roboflow 增強」的 Gate 類型：health（資料集健康度不足）
# 與 eval（訓練後 mAP 太低），兩者的共同解法都是「補更多已標記的樣本」。
AUGMENTABLE_GATES = {"health_gate_failed", "eval_gate_failed"}


def _active_dataset_name() -> Optional[str]:
    """讀 workspace_state.json 拿目前正在用的資料集名稱——skill_train 每次訓練
    （含 health_gate_failed 提早返回前）都會把 dataset.name 寫進去，所以分數不夠
    時可以據此知道該對哪個資料集做增強。"""
    try:
        with open(WORKSPACE_STATE_PATH, "r", encoding="utf-8") as f:
            return (json.load(f).get("dataset") or {}).get("name")
    except Exception:
        return None


def _active_dataset_is_single_class() -> bool:
    """目前資料集『實際用到的類別』是不是只有一種。augment_dataset_from_roboflow 會把
    合併進來的標籤全部改成 class 0，只有單一類別偵測用它才安全；多類別資料集用了會把
    別的類別的框全污染成同一類。所以「去 Roboflow 增強」這條路只在單類別時才提議。"""
    ds_name = _active_dataset_name()
    if not ds_name:
        return False
    try:
        health = check_dataset_health.invoke({"dataset_name": ds_name})
        dist = health.get("class_distribution") or {}
        used = [c for c in dist.values() if (c.get("count") or 0) > 0]
        return len(used) == 1
    except Exception:
        return False

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


class TrainingProposal(BaseModel):
    """使用者確認前就固定下來的實際訓練參數。"""

    dataset_name: Optional[str] = Field(
        default=None,
        description="要用來訓練的本機資料集名稱；對話中已指定時必須照填",
    )
    variant: str = Field(
        default="yolov9s.pt",
        description="YOLO 權重檔名；使用者未指定時採低資源建議值 yolov9s.pt",
    )
    epochs: int = Field(default=20, ge=1, description="訓練輪數")
    imgsz: int = Field(default=320, ge=32, description="輸入影像尺寸")
    batch: int = Field(default=4, ge=1, description="batch size")
    model_name: str = Field(default="exp", description="輸出的實驗名稱")


class TrainingParamsReply(BaseModel):
    """使用者用文字回覆「這次訓練要用什麼參數」時的結構化解讀。

    訓練前改成先在對話裡問一次參數，使用者回的是自然語言（「就用這些」、「epochs 改
    50、batch 8」、「先不要訓練」），這裡只把它翻成明確意圖與欄位；值合不合理由
    Manager 端再檢查一次，不信任 LLM 直接填進 skill_train。
    """

    intent: Literal["use_suggested", "override", "cancel", "unrelated"] = Field(
        description="use_suggested：照建議的參數跑；override：要改其中幾個參數；"
                    "cancel：不要訓練了；unrelated：這句話不是在回答參數問題，是別的需求"
    )
    variant: Optional[str] = Field(default=None, description="要改用的 YOLO 權重檔名，沒提到就留空")
    epochs: Optional[int] = Field(default=None, description="要改成的訓練輪數，沒提到就留空")
    imgsz: Optional[int] = Field(default=None, description="要改成的輸入影像尺寸，沒提到就留空")
    batch: Optional[int] = Field(default=None, description="要改成的 batch size，沒提到就留空")
    model_name: Optional[str] = Field(default=None, description="要改成的實驗名稱，沒提到就留空")


class AugmentReply(BaseModel):
    """使用者用文字回覆「要不要去 Roboflow 補資料」時的結構化解讀。"""

    decision: Literal["yes", "no", "unrelated"] = Field(
        description="yes：同意現在去搜尋候選資料；no：不要；"
                    "unrelated：這句話不是在回答這個問題，是別的需求"
    )
    rationale: str = Field(default="", description="判斷依據，用於稽核紀錄")


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
    training_params: Optional[TrainingProposal] = Field(
        default=None,
        description="is_training=True 時要執行的確切訓練參數；使用者確認畫面與 skill_train 都使用這一份",
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
    pending_operation: Optional[dict]
    train_confirmed_operation_id: Optional[str]
    # skill_eval 回報的診斷圖表（混淆矩陣／F1／P-R），由 stream_trace 轉成 SSE 事件
    eval_charts: Optional[dict]
    # 分數不夠時只問一次「要不要去 Roboflow 增強」，避免反覆騷擾；baseline 記錄提議
    # 增強當下的圖片數，用來判斷之後資料到底有沒有真的長大（有長大才值得再重訓）。
    augment_offered: bool
    augment_baseline_count: Optional[int]
    roboflow_proposal: Optional[dict]
    roboflow_proposal_handled: bool
    # Frontend refresh payload emitted after a confirmed Roboflow merge.
    dataset_refresh: Optional[dict]
    # Keeps the post-augmentation train -> evaluate continuation deterministic.
    post_augment_eval_required: bool
    # 訓練前「先用對話問一次參數」的待辦。使用者要回一句話才會繼續，所以這一輪會先
    # 結束；而 stream_trace 在新訊息進來時會把 pending_operation／
    # post_augment_eval_required 這類單次任務欄位重置，因此整份 operation 與「補強後
    # 必須自動接評估」的旗標都要存在這裡，下一輪 Manager 解析完回覆才復原。
    pending_param_question: Optional[dict]
    # 已經在對話裡建議過「去 Roboflow 補資料」，正在等使用者回文字決定要不要做。
    augment_suggested: bool


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


def _extract_roboflow_proposal(new_messages: list) -> Optional[dict]:
    """從 Data Agent 的工具結果取出已完成的暫存評估，交給確認 Gate 顯示。

    不依賴 LLM 是否把 proposal_id 正確複述在 report_to_manager 裡；真正的工具回傳
    才是結構化事實來源。
    """
    for message in new_messages:
        if isinstance(message, ToolMessage) and message.name == "prepare_roboflow_augmentation":
            data = _parse_tool_content(message.content)
            if data.get("status") == "proposal_ready" and data.get("proposal_id"):
                return data
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


def _extract_eval_charts(new_messages: list) -> Optional[dict]:
    """從 skill_eval 的回傳裡取出實際評估的模型名稱與診斷圖表清單。

    圖表路徑一律取自 Tool 的結構化回傳，不讓 LLM 轉述——它抄錯檔名就會變成破圖。
    """
    for m in reversed(new_messages):
        if isinstance(m, ToolMessage) and m.name == "skill_eval":
            data = _parse_tool_content(m.content)
            if isinstance(data, dict) and data.get("charts"):
                return {
                    "model_name": data.get("resolved_model_name", ""),
                    "charts": data["charts"],
                }
    return None


def _plan_to_state(plan: Plan) -> dict:
    steps = [
        {
            "id": s.id,
            "description": s.description,
            "agent": s.agent,
            "is_training": s.is_training,
            "training_params": s.training_params.model_dump() if s.training_params else None,
            "status": "pending",
        }
        for s in plan.steps
    ]
    return {"goal": plan.goal, "steps": steps}


def _normalized_training_params(step: dict) -> dict:
    """把 Planner 的提案補成可直接交給 skill_train 的完整參數。"""
    raw = dict(step.get("training_params") or {})
    datasets = _existing_dataset_names()
    dataset_name = raw.get("dataset_name")
    if not dataset_name and len(datasets) == 1:
        dataset_name = datasets[0]
    return {
        "dataset_name": dataset_name,
        "variant": raw.get("variant") or "yolov9s.pt",
        "epochs": int(raw.get("epochs") or 20),
        "imgsz": int(raw.get("imgsz") or 320),
        "batch": int(raw.get("batch") or 4),
        "model_name": raw.get("model_name") or "exp",
    }


# 訓練前健康摘要的判讀門檻：覆蓋率與平衡度直接沿用 skill_train 的 Health Gate 常數，
# 總分另外定一個，因為 skill_train 沒有拿總分當 Gate（分數低但標記完整仍然可以訓練）。
HEALTH_SCORE_ADVICE_THRESHOLD = 60


def _training_health_summary(dataset_name: str) -> Optional[dict]:
    """問訓練參數之前先跑一次 check_dataset_health，整理成要一起講給使用者聽的摘要。

    這裡只負責「讓使用者在決定訓練之前就看到真實數字與建議」，不擋訓練——真正會擋
    下來的 Health Gate 仍然是 skill_train 裡的標記覆蓋率門檻。回 None 代表檢查本身
    失敗（例如資料集讀不到），這時不阻斷流程，照原本的方式問參數就好。
    """
    try:
        health = check_dataset_health.invoke({"dataset_name": dataset_name})
    except Exception:
        return None
    if health.get("status") != "success":
        return None

    coverage = health.get("coverage") or 0
    balance = health.get("balance") or 0
    score = health.get("score") or 0

    # 樣本數最少的幾個類別：使用者要判斷「值不值得現在訓練」的時候，看具體哪個類別缺圖
    # 比看一個平衡度分數有用得多。
    counts = [
        (v.get("name") or str(k), v.get("count") or 0)
        for k, v in (health.get("class_distribution") or {}).items()
    ]
    thinnest = sorted(counts, key=lambda kv: kv[1])[:3]

    concerns = []
    if coverage < COVERAGE_GATE:
        concerns.append(
            f"標記覆蓋率只有 {coverage}%，低於 {COVERAGE_GATE}% 門檻——這一項會讓訓練在啟動時"
            f"被 Health Gate 擋下來，必須先補標記"
        )
    if balance < BALANCE_WARN_THRESHOLD:
        detail = "、".join(f"{name} {count} 張" for name, count in thinnest)
        concerns.append(
            f"類別分佈不平衡（平衡度 {balance} 分）：樣本最少的是 {detail}，"
            f"這些類別訓練後可能幾乎偵測不到"
        )
    if not concerns and score < HEALTH_SCORE_ADVICE_THRESHOLD:
        concerns.append(f"健康總分偏低（{score} 分）")

    return {
        "dataset_name": dataset_name,
        "score": score,
        "coverage": coverage,
        "balance": balance,
        "total_images": health.get("total_images"),
        "labeled_images": health.get("labeled_images"),
        "passed": not concerns,
        "blocks_training": coverage < COVERAGE_GATE,
        "concerns": concerns,
    }


def _format_training_health_block(summary: Optional[dict]) -> str:
    """把健康摘要寫成問參數時擺在最前面的那段文字：先講結論，再給建議。

    數字一律由 check_dataset_health 算出來後直接帶進字串，不經過 LLM 轉述，避免
    「建議」講得漂亮但數字是編的。要不要照樣訓練由使用者自己決定。
    """
    if not summary:
        return ""

    name = summary.get("dataset_name") or "—"
    head = (
        f"資料集「{name}」健康檢查：總分 {summary.get('score')} 分、"
        f"標記覆蓋率 {summary.get('coverage')}%"
        f"（{summary.get('labeled_images')}/{summary.get('total_images')} 張）、"
        f"平衡度 {summary.get('balance')} 分。"
    )
    if summary.get("passed"):
        return f"✅ {head}\n\n"

    lines = [f"⚠️ {head}這個資料集**沒有通過健康檢查**："]
    lines += [f"・{c}" for c in summary.get("concerns") or []]
    if summary.get("blocks_training"):
        lines.append(
            "建議先把沒標記的圖片補完再訓練——現在硬跑會在啟動時被 Health Gate 擋下來，等於白等一輪。"
        )
    else:
        lines.append(
            "建議補齊樣本數過少的類別再訓練，效果會好一截；但先跑一輪看基準線也是合理的做法，"
            "訓練不會被擋下。要不要繼續由你決定。"
        )
    return "\n".join(lines) + "\n\n"


def _training_fingerprint(step_id: str, params: dict) -> str:
    payload = json.dumps(
        {"step_id": step_id, "params": params},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_post_augment_training_transition(state: SupervisorState, merge_result: dict, health: dict) -> dict:
    """Build a new confirmed training operation after a successful Roboflow merge."""
    dataset_name = merge_result.get("dataset_name") or ""
    refresh = {
        "dataset_name": dataset_name,
        "added_images": merge_result.get("added_images", 0),
        "total_images_now": merge_result.get("total_images_now", 0),
    }
    coverage = health.get("coverage") or 0
    if health.get("status") != "success" or coverage < 70:
        detail = health.get("message") or f"標記覆蓋率為 {coverage}%（需要至少 70%）"
        return {
            "next": "FINISH",
            "augment_offered": True,
            "roboflow_proposal_handled": True,
            "dataset_refresh": refresh,
            "post_augment_eval_required": False,
            "last_trained_model": None,
            "messages": [AIMessage(content=(
                f"Roboflow 資料已合併到「{dataset_name}」，但重新檢查後仍不適合訓練：{detail}。"
                "請先完成資料整理或標記，再重新提出訓練要求。"
            ))],
        }

    previous = dict((state.get("pending_operation") or {}).get("params") or {})
    base_model_name = str(previous.get("model_name") or "exp")
    params = {
        "dataset_name": dataset_name,
        "variant": previous.get("variant") or "yolov9s.pt",
        "epochs": int(previous.get("epochs") or 20),
        "imgsz": int(previous.get("imgsz") or 320),
        "batch": int(previous.get("batch") or 4),
        "model_name": f"{base_model_name}_aug_{uuid.uuid4().hex[:6]}",
    }
    step_id = f"post-augment-train-{uuid.uuid4().hex[:8]}"
    operation = {
        "operation_id": f"train-{uuid.uuid4().hex[:12]}",
        "type": "training",
        "step_id": step_id,
        "step_description": "Roboflow 補強完成且健康檢查通過，使用補強後資料重新訓練",
        "params": params,
        "fingerprint": _training_fingerprint(step_id, params),
        "status": "pending_confirmation",
    }
    plan = state.get("plan") or {"goal": "補強資料後重新訓練並評估", "steps": []}
    steps = [dict(s) for s in plan.get("steps", [])]
    steps.extend([
        {
            "id": step_id,
            "description": "使用 Roboflow 補強後的資料重新訓練",
            "agent": "model_agent",
            "is_training": True,
            "training_params": params,
            "status": "in_progress",
        },
        {
            "id": f"post-augment-eval-{uuid.uuid4().hex[:8]}",
            "description": "評估補強後重新訓練的模型",
            "agent": "evaluation_agent",
            "is_training": False,
            "training_params": None,
            "status": "pending",
        },
    ])
    return {
        "next": "train_params_gate",
        "augment_offered": True,
        "roboflow_proposal_handled": True,
        "dataset_refresh": refresh,
        "post_augment_eval_required": True,
        "last_trained_model": None,
        "pending_operation": operation,
        "plan": {**plan, "steps": steps},
        "messages": [AIMessage(content=(
            f"已將 Roboflow 資料合併到「{dataset_name}」，目前共有 {refresh['total_images_now']} 張圖片；"
            f"重新檢查後標記覆蓋率為 {coverage}%，已達訓練門檻。接下來我先跟你確認這次重訓要用的參數。"
        ))],
    }


def _build_approved_training_tool(params: dict, operation_id: str):
    """建立只接受已核准參數的 skill_train，避免 LLM 在確認後偷偷改值。"""
    approved = dict(params)

    @tool("skill_train")
    def approved_skill_train(
        dataset_name: str = approved["dataset_name"],
        variant: str = approved["variant"],
        epochs: int = approved["epochs"],
        imgsz: int = approved["imgsz"],
        batch: int = approved["batch"],
        model_name: str = approved["model_name"],
    ) -> dict:
        """執行已由使用者確認的模型訓練；所有參數都必須與確認畫面完全相同。"""
        requested = {
            "dataset_name": dataset_name,
            "variant": variant,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "model_name": model_name,
        }
        if requested != approved:
            return {
                "status": "approval_mismatch",
                "message": (
                    f"操作 {operation_id} 的訓練參數與使用者核准內容不一致，"
                    "已阻止訓練啟動。"
                ),
                "approved_params": approved,
                "requested_params": requested,
            }
        return skill_train.invoke(approved)

    return approved_skill_train


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
    param_reply_model = chat_model.with_structured_output(
        TrainingParamsReply, method="function_calling", include_raw=True
    )
    augment_reply_model = chat_model.with_structured_output(
        AugmentReply, method="function_calling", include_raw=True
    )

    def planner_node(state: SupervisorState) -> dict:
        # 使用者這句話是在回答 Manager 上一輪問的問題（訓練參數怎麼設、要不要去補
        # 資料），不是新需求——不要重新規劃，直接交給 Manager 解析那句回覆。
        if state.get("pending_param_question") or state.get("augment_suggested"):
            return {}
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

    def prepare_train_operation(state: SupervisorState) -> dict:
        """把訓練 Step 固定成一份有唯一 ID 的操作，再交給確認 Gate。

        operation_id 必須在 interrupt() 前先寫入 checkpoint；如果直接在 interrupt
        node 裡產生，resume 時 node 會從頭執行而產生另一個 ID。
        """
        plan = state.get("plan") or {"steps": []}
        steps = [dict(s) for s in plan.get("steps", [])]
        step = next(
            (s for s in steps if s.get("agent") == "model_agent" and s.get("status") == "in_progress"),
            None,
        )
        pending = state.get("pending_operation")

        if step and not step.get("is_training"):
            return {"next": "model_agent", "pending_operation": None}

        if not step:
            if pending and pending.get("type") == "training" and pending.get("status") == "approved":
                return {"next": "model_agent"}
            if (
                pending
                and pending.get("type") == "training"
                and pending.get("status") == "completed"
                and (state.get("last_report") or {}).get("reason_code") == "eval_gate_failed"
            ):
                retry_params = dict(pending.get("params") or {})
                retry_step_id = f"{pending.get('step_id') or 'training'}-retry"
                retry_operation = {
                    "operation_id": f"train-{uuid.uuid4().hex[:12]}",
                    "type": "training",
                    "step_id": retry_step_id,
                    "step_description": "模型評估未達門檻，使用相同參數重新訓練",
                    "params": retry_params,
                    "fingerprint": _training_fingerprint(retry_step_id, retry_params),
                    "status": "pending_confirmation",
                }
                return {
                    "next": "train_params_gate",
                    "pending_operation": retry_operation,
                }
            return {"next": "model_agent", "pending_operation": None}

        params = _normalized_training_params(step)
        if not params.get("dataset_name"):
            return {
                "next": "FINISH",
                "messages": [AIMessage(
                    content="開始訓練前還缺少資料集名稱，請先指定要使用哪一個本機資料集。"
                )],
            }

        step["training_params"] = params
        fingerprint = _training_fingerprint(str(step.get("id", "")), params)
        if (
            pending
            and pending.get("fingerprint") == fingerprint
            and pending.get("status") in {"pending_confirmation", "approved"}
        ):
            operation = pending
        else:
            # 健康檢查在「問參數之前」就跑完：以前只有 skill_train 內部會檢查，使用者
            # 會先被問完一輪參數、按下確認，才在訓練啟動時看到 Gate 沒過，體感上像是
            # 系統默許了這個資料集。這裡只是把數字提早攤開，不改變能不能訓練的判斷。
            operation = {
                "operation_id": f"train-{uuid.uuid4().hex[:12]}",
                "type": "training",
                "step_id": step.get("id"),
                "step_description": step.get("description", ""),
                "params": params,
                "fingerprint": fingerprint,
                "status": "pending_confirmation",
                "health": _training_health_summary(params["dataset_name"]),
            }
        return {
            "next": "train_params_gate",
            "pending_operation": operation,
            "plan": {**plan, "steps": steps},
        }

    def train_params_gate(state: SupervisorState) -> dict:
        """訓練前先用對話講一次資料集健康狀況、問一次參數，不直接跳確認框。

        這裡刻意不用 interrupt()：確認框只有「開始／取消」兩顆按鈕，改不了參數值，
        而使用者要的是能在聊天視窗直接說「epochs 改 50」。所以把即將使用的參數當成
        一般回覆送出、這一輪就結束，等使用者回一句話；下一輪由 decision_engine 的
        pending_param_question 分支解析，再進 train_confirm_gate 做最後確認。

        next 走 ASK_USER 而不是 FINISH：這一輪只是停下來等回話，不是任務收尾，不該
        順手把 save_offer_gate 的「要不要存模型」也問出來。
        """
        operation = state.get("pending_operation") or {}
        params = operation.get("params") or {}
        question = (
            f"{_format_training_health_block(operation.get('health'))}"
            f"準備開始訓練資料集「{params.get('dataset_name') or '—'}」，這是我打算使用的參數：\n"
            f"・模型權重 variant：{params.get('variant') or '—'}\n"
            f"・訓練輪數 epochs：{params.get('epochs') if params.get('epochs') is not None else '—'}\n"
            f"・影像尺寸 imgsz：{params.get('imgsz') if params.get('imgsz') is not None else '—'}\n"
            f"・batch：{params.get('batch') if params.get('batch') is not None else '—'}\n"
            f"・實驗名稱 model_name：{params.get('model_name') or '—'}\n"
            "照這樣跑就回「就用這些」；要改直接說，例如「epochs 改 50、batch 8」；"
            "不想訓練就說「取消」。"
        )
        return {
            "next": "ASK_USER",
            "pending_param_question": {
                "operation": operation,
                "post_augment_eval_required": bool(state.get("post_augment_eval_required")),
            },
            "messages": [AIMessage(content=question)],
        }

    def _latest_user_text(state: SupervisorState) -> str:
        for message in reversed(state["messages"]):
            if isinstance(message, HumanMessage):
                return str(message.content)
        return ""

    def _parse_reply(model, system_prompt: str, state: SupervisorState):
        """把使用者的最後一句話丟給結構化模型解讀，回傳 (parsed, usage)。

        解析失敗不拋例外——呼叫端會當成「這句話跟剛問的問題無關」，退回一般決策流程，
        不要因為一次 LLM 失誤就把使用者的訊息吃掉。
        """
        context = [SystemMessage(content=system_prompt), HumanMessage(content=_latest_user_text(state))]
        try:
            result = _invoke_with_retry(lambda: model.invoke(context))
        except Exception:
            return None, None
        return result["parsed"], _usage_from_message(result["raw"])

    # 使用者回覆參數時允許修改的欄位：(欄位名, 型別, 顯示名稱)。variant／model_name
    # 是字串，其餘三個必須是正整數——LLM 可能把「不改」解讀成 0 或負數，這裡擋掉。
    PARAM_FIELDS = (
        ("variant", str, "模型權重"),
        ("epochs", int, "訓練輪數"),
        ("imgsz", int, "影像尺寸"),
        ("batch", int, "batch"),
        ("model_name", str, "實驗名稱"),
    )

    def _resolve_param_reply(state: SupervisorState, question: dict, reply: TrainingParamsReply,
                             rounds: int, retry_counters: dict, usage) -> dict:
        """把使用者的參數回覆套進待確認的訓練操作，再送去確認 Gate。"""
        operation = dict(question.get("operation") or {})
        params = dict(operation.get("params") or {})
        applied, ignored = [], []

        if reply.intent == "override":
            for field, caster, label in PARAM_FIELDS:
                value = getattr(reply, field, None)
                if value is None:
                    continue
                try:
                    value = caster(value)
                except (TypeError, ValueError):
                    ignored.append(label)
                    continue
                if caster is int and value <= 0:
                    ignored.append(label)
                    continue
                if params.get(field) == value:
                    continue
                params[field] = value
                applied.append(f"{label} {value}")

        if applied:
            # 參數變了就是另一個操作：換 operation_id 與 fingerprint，確認框才會重新顯示。
            operation.update({
                "operation_id": f"train-{uuid.uuid4().hex[:12]}",
                "params": params,
                "fingerprint": _training_fingerprint(str(operation.get("step_id") or "training"), params),
                "status": "pending_confirmation",
            })
            note = "已改成：" + "、".join(applied) + "。"
        else:
            note = "好，就用剛剛列出的參數。"
        if ignored:
            note += "（" + "、".join(ignored) + " 看不出要改成什麼，先沿用原值。）"
        note += "接下來會跳出訓練確認，按「開始訓練」就會用這份參數跑。"

        return {
            "next": "train_confirm_gate",
            "rounds": rounds,
            "retry_counters": retry_counters,
            "pending_operation": operation,
            "pending_param_question": None,
            # 補強後重訓必須自動接評估，這個旗標在新訊息進來時被重置過，這裡復原。
            "post_augment_eval_required": bool(question.get("post_augment_eval_required")),
            "usage": usage,
            "messages": [AIMessage(content=note)],
        }

    def train_confirm_gate(state: SupervisorState) -> dict:
        """manager 決定路由到 model_agent 時，先經過這一關：不是訓練步驟就直接
        放行；是訓練步驟就用 interrupt() 暫停，等待前端使用者確認後才真的進
        model_agent（訓練耗時、耗運算資源，不該讓 Agent 自己說做就做）。

        train_confirmed_operation_id 記錄核准的是哪一個具體操作；同一個操作因 Gate
        失敗或重新規劃而重試時不重複詢問，但只要參數或 operation_id 改變，就必須
        顯示新的確認內容。"""
        operation = state.get("pending_operation") or {}
        if operation.get("type") != "training":
            return {"next": "model_agent"}

        operation_id = operation.get("operation_id")
        if state.get("train_confirmed_operation_id") == operation_id:
            return {
                "next": "model_agent",
                "pending_operation": {**operation, "status": "approved"},
            }

        resume = interrupt({
            "action_type": "training",
            "operation_id": operation_id,
            "reason": "即將啟動模型訓練，這是高成本、耗時的操作，請確認是否繼續。",
            "step_description": operation.get("step_description", ""),
            "training_params": operation.get("params", {}),
        })
        if isinstance(resume, dict) and resume.get("approved"):
            return {
                "next": "model_agent",
                "train_confirmed": True,
                "train_confirmed_operation_id": operation_id,
                "pending_operation": {**operation, "status": "approved"},
            }
        return {
            "next": "FINISH",
            "pending_operation": {**operation, "status": "rejected"},
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

        # 上一輪 Manager 問了問題、這一輪使用者回了一句話時，那句話就是答案，要先解析
        # 完再談路由。若解析結果是「在講別的事」，就把待辦放掉、照一般流程處理新需求，
        # 但不能讓旗標留著（否則下一句話又會被當成在回答同一個問題）——stale_updates
        # 帶著這件事一路傳到最後的 return。
        stale_updates: dict = {}

        param_question = state.get("pending_param_question")
        if param_question:
            suggested = (param_question.get("operation") or {}).get("params") or {}
            reply, usage = _parse_reply(param_reply_model, (
                "Manager 剛問使用者「這次訓練要用什麼參數」，建議值是：\n"
                + json.dumps(suggested, ensure_ascii=False, sort_keys=True)
                + "\n把使用者的回覆翻成結構化意圖：同意照建議跑填 use_suggested；要改參數填"
                  " override 並只填他明確提到的欄位（沒提到的一律留空）；不想訓練填 cancel；"
                  "這句話根本不是在回答參數問題填 unrelated。"
            ), state)
            intent = reply.intent if reply else "unrelated"
            if intent == "cancel":
                return {
                    "next": "FINISH",
                    "rounds": rounds,
                    "retry_counters": retry_counters,
                    "pending_param_question": None,
                    "usage": usage,
                    "messages": [AIMessage(content="好，這次不訓練，已經停下來了。要再訓練隨時跟我說。")],
                }
            if intent in {"use_suggested", "override"}:
                return _resolve_param_reply(state, param_question, reply, rounds, retry_counters, usage)
            stale_updates["pending_param_question"] = None

        if state.get("augment_suggested"):
            reply, usage = _parse_reply(augment_reply_model, (
                "Manager 剛建議使用者「去 Roboflow Universe 找公開資料集，補強目前的訓練資料」，"
                "現在要判斷使用者這句回覆是同意去找（yes）、不要（no），還是在講別的事"
                "（unrelated）。"
            ), state)
            augment_decision = reply.decision if reply else "unrelated"
            if augment_decision == "yes":
                ds_name = _active_dataset_name()
                return {
                    "next": "data_agent",
                    "rounds": rounds,
                    "retry_counters": retry_counters,
                    "augment_suggested": False,
                    "augment_offered": True,
                    "augment_baseline_count": _count_images(DATASET_DIR / ds_name) if ds_name else None,
                    "usage": usage,
                    "messages": [HumanMessage(content=(
                        f"使用者同意補強資料集「{ds_name}」。請呼叫 prepare_roboflow_augmentation "
                        "自動搜尋 Roboflow Universe 候選，下載到暫存區並評估類別對應、重複圖片、"
                        "可用標註與風險。不要修改正式資料集；完成後把 proposal 結果回報給 Manager，"
                        "等待使用者最後確認。"
                    ))],
                }
            if augment_decision == "no":
                return {
                    "next": "FINISH",
                    "rounds": rounds,
                    "retry_counters": retry_counters,
                    "augment_suggested": False,
                    "usage": usage,
                    "messages": [AIMessage(content=(
                        "好，這次不去 Roboflow 補資料。想改參數重訓、或換一個資料集，都可以跟我說。"
                    ))],
                }
            stale_updates["augment_suggested"] = False

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

        # After a confirmed Roboflow merge, training and evaluation are one explicit continuation.
        # Do not ask the LLM to infer this handoff from conversational text.
        if state.get("post_augment_eval_required"):
            report_agent = last_report.get("agent")
            report_status = last_report.get("status")
            if report_agent == "model_agent" and report_status == "success":
                plan = _mark_step_in_progress(state.get("plan"), "evaluation_agent")
                model_name = state.get("last_trained_model") or "最新完成的模型"
                return {
                    "next": "evaluation_agent",
                    "rounds": rounds,
                    "retry_counters": retry_counters,
                    "plan": plan,
                    "messages": [HumanMessage(content=(
                        f"請立即評估剛完成的模型「{model_name}」，回報 mAP50、loss 與是否通過門檻。"
                    ))],
                }
            if report_agent == "evaluation_agent":
                if report_status == "success":
                    message = f"Roboflow 補強後的重新訓練與評估已完成：{last_report.get('reason', '')}"
                else:
                    message = (
                        "Roboflow 補強後已重新訓練並評估，但結果仍未通過門檻："
                        f"{last_report.get('reason', '')}。本輪先停止，避免自動反覆補資料與重訓。"
                    )
                return {
                    "next": "FINISH",
                    "rounds": rounds,
                    "retry_counters": retry_counters,
                    "post_augment_eval_required": False,
                    "messages": [AIMessage(content=message)],
                }

        # Data Agent 已經完成 Roboflow 搜尋與暫存評估時，先呈現完整 proposal；
        # 使用者確認前絕不修改正式資料集。
        proposal = state.get("roboflow_proposal")
        if proposal and not state.get("roboflow_proposal_handled"):
            return {
                "next": "augment_gate",
                "rounds": rounds,
                "retry_counters": retry_counters,
            }

        # 分數不夠時先在對話裡「建議」去補資料，等使用者回一句話再動作——搜尋要下載
        # 外部資料、花時間也花額度，不該在使用者沒表示意見前就先跑。同意與否由上面的
        # augment_suggested 分支解析；同意後才派 Data Agent 搜尋、暫存並評估，接著由
        # proposal 分支進入合併前的最後確認。
        if reason_code in AUGMENTABLE_GATES:
            ds_name = _active_dataset_name()
            if ds_name and not state.get("augment_offered"):
                gate_reason = (last_report.get("reason") or "").strip()
                detail = f"（{gate_reason}）" if gate_reason else ""
                return {
                    "next": "ASK_USER",
                    "rounds": rounds,
                    "retry_counters": retry_counters,
                    "augment_suggested": True,
                    "messages": [AIMessage(content=(
                        f"資料集「{ds_name}」這次沒通過門檻{detail}。\n"
                        "我建議去 Roboflow Universe 找公開資料集，把能對應到你現有類別的已標記"
                        "圖片補進來——只會先下載到暫存區做評估，正式資料集不會被改動，要不要"
                        "真的合併由你在確認畫面決定。\n"
                        "要我去找就回「去找」；不用就回「不用」，我先停在這裡。"
                    ))],
                }
            return {
                "next": "FINISH",
                "rounds": rounds,
                "retry_counters": retry_counters,
                "messages": [AIMessage(
                    content="Agent 已嘗試搜尋並評估 Roboflow 候選，但沒有找到能安全對應目前"
                            f"類別的可用資料。原始原因：{last_report.get('reason', '')}"
                )],
            }

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
        if dataset_names:
            # 名稱一律用「」括起來：以前是逗號分隔的裸字串，資料夾名稱剛好是數字時
            # （例如「2」）模型會把它當成數量或順序編號而不是名稱，列清單給使用者
            # 看的時候直接漏掉；使用者回「2」也會被理解成「清單第 2 項」而選錯資料集。
            listed = "、".join(f"「{name}」" for name in dataset_names)
            dataset_note = (
                f"目前本機已有的資料集共 {len(dataset_names)} 個，"
                f"名稱如下（「」裡面就是完整的資料夾名稱）：{listed}。\n"
                "注意：資料夾名稱本身可能就是一個數字（例如「2」），那是它的名稱、不是數量或"
                "順序編號——列清單給使用者看時絕對不能漏掉這種名稱。使用者只回一個數字或短"
                "名稱時，指的是名稱完全相同的那個資料集，不要理解成「清單的第幾項」。"
            )
        else:
            dataset_note = "目前本機沒有任何含圖片的資料集。"
        context.append(SystemMessage(content=dataset_note))
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
                **stale_updates,
            }
        decision = result["parsed"]
        usage = _usage_from_message(result["raw"])

        if decision.action == "call_agent" and decision.target_agent:
            plan = _mark_step_in_progress(state.get("plan"), decision.target_agent)
            return {
                "next": decision.target_agent,
                "rounds": rounds,
                "retry_counters": retry_counters,
                "plan": plan,
                "usage": usage,
                **stale_updates,
            }

        reply_text = decision.final_reply or decision.rationale
        return {
            "next": "FINISH",
            "rounds": rounds,
            "retry_counters": retry_counters,
            "messages": [AIMessage(content=reply_text)],
            "usage": usage,
            **stale_updates,
        }

    def _run_sub_agent(agent_name, sub_agent, state: SupervisorState) -> dict:
        input_messages = state["messages"]
        result = sub_agent.invoke({"messages": input_messages})
        new_messages = result["messages"][len(input_messages):]
        report = _extract_agent_report(agent_name, new_messages)
        return {"messages": new_messages, "last_report": report}

    def run_data_agent(state: SupervisorState) -> dict:
        result = _run_sub_agent("data_agent", data_agent, state)
        proposal = _extract_roboflow_proposal(result["messages"])
        if proposal:
            result["roboflow_proposal"] = proposal
            result["roboflow_proposal_handled"] = False
        return result

    def run_model_agent(state: SupervisorState) -> dict:
        operation = state.get("pending_operation") or {}
        active_agent = model_agent
        if operation.get("type") == "training" and operation.get("status") == "approved":
            params = operation.get("params") or {}
            approved_tool = _build_approved_training_tool(params, operation.get("operation_id", ""))
            exact_prompt = (
                MODEL_AGENT_PROMPT
                + "\n\n這次訓練已由使用者核准。你必須呼叫 skill_train，且只能使用以下參數，"
                  "不得自行替換或省略：\n"
                + json.dumps(params, ensure_ascii=False, sort_keys=True)
            )
            active_agent = _build_sub_agent(chat_model, [approved_tool, skill_infer], exact_prompt)

        result = _run_sub_agent("model_agent", active_agent, state)
        trained_model = _extract_trained_model_name(result["messages"])
        if trained_model:
            result["last_trained_model"] = trained_model
            result["pending_operation"] = {**operation, "status": "completed"}
        elif operation.get("type") == "training":
            report_status = (result.get("last_report") or {}).get("status")
            if report_status == "failed":
                result["pending_operation"] = {**operation, "status": "failed"}
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
            "action_type": "model_save",
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

    def augment_gate(state: SupervisorState) -> dict:
        """Data Agent 搜尋與暫存評估完成後的唯一人工確認點。

        interrupt payload 直接帶完整 proposal，使用者看得到來源、版本、授權、類別
        對應、可用與跳過數量。確認後才套用；拒絕就刪除暫存資料。
        """
        proposal = state.get("roboflow_proposal") or {}
        proposal_id = proposal.get("proposal_id")
        if not proposal_id:
            return {
                "next": "FINISH",
                "roboflow_proposal_handled": True,
                "messages": [AIMessage(content="Roboflow 評估結果遺失，沒有修改正式資料集。")],
            }
        candidate = proposal.get("candidate") or {}
        resume = interrupt({
            "action_type": "roboflow_proposal",
            "reason": (
                f"Agent 已搜尋並實際檢查 {proposal.get('candidate_count', 0)} 個候選，"
                f"建議使用「{candidate.get('name', candidate.get('project', 'Roboflow 資料集'))}」。"
                "以下資料已在暫存區完成驗證；接受後才會寫入正式資料集。"
            ),
            "dataset_name": proposal.get("dataset_name"),
            "proposal": proposal,
        })
        if isinstance(resume, dict) and resume.get("approved"):
            result = apply_roboflow_proposal.invoke({"proposal_id": proposal_id})
            if result.get("status") == "success":
                health = check_dataset_health.invoke({"dataset_name": result.get("dataset_name", "")})
                return _build_post_augment_training_transition(state, result, health)
            else:
                note = f"Roboflow 合併沒有完成：{result.get('message') or result.get('status')}。"
            return {
                "next": "FINISH",
                "augment_offered": True,
                "roboflow_proposal_handled": True,
                "messages": [AIMessage(content=note)],
            }
        discard_roboflow_proposal(proposal_id)
        return {
            "next": "FINISH",
            "augment_offered": True,
            "roboflow_proposal_handled": True,
            "messages": [AIMessage(content="已拒絕這份 Roboflow 建議並刪除暫存資料，正式資料集沒有變更。")],
        }

    def run_evaluation_agent(state: SupervisorState) -> dict:
        result = _run_sub_agent("evaluation_agent", evaluation_agent, state)
        charts = _extract_eval_charts(result["messages"])
        if charts:
            result["eval_charts"] = charts
        return result

    graph = StateGraph(SupervisorState)
    graph.add_node("planner", planner_node)
    graph.add_node("manager", decision_engine)
    graph.add_node("observer", observer_node)
    graph.add_node("prepare_train_operation", prepare_train_operation)
    graph.add_node("train_params_gate", train_params_gate)
    graph.add_node("train_confirm_gate", train_confirm_gate)
    graph.add_node("save_offer_gate", save_offer_gate)
    graph.add_node("augment_gate", augment_gate)
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
            "model_agent": "prepare_train_operation",
            "evaluation_agent": "evaluation_agent",
            "planner": "planner",
            "augment_gate": "augment_gate",
            # 使用者回覆完訓練參數後直接進最後確認，不必再繞 prepare_train_operation
            # 重新組一份 operation（那會把剛談好的參數蓋掉）。
            "train_confirm_gate": "train_confirm_gate",
            # 這一輪只是停下來等使用者回話，不是任務收尾：不走 save_offer_gate，
            # 免得在等回覆的同時又跳出「要不要把模型存進模型庫」。
            "ASK_USER": END,
            "FINISH": "save_offer_gate",
        },
    )
    graph.add_conditional_edges(
        "prepare_train_operation",
        lambda s: s["next"],
        {
            "train_params_gate": "train_params_gate",
            "model_agent": "model_agent",
            "FINISH": "save_offer_gate",
        },
    )
    graph.add_conditional_edges(
        "train_params_gate",
        lambda s: s["next"],
        {"ASK_USER": END},
    )
    graph.add_conditional_edges(
        "train_confirm_gate",
        lambda s: s["next"],
        {"model_agent": "model_agent", "FINISH": "save_offer_gate"},
    )
    graph.add_conditional_edges(
        "augment_gate",
        lambda s: s["next"],
        {
            "data_agent": "data_agent",
            "train_params_gate": "train_params_gate",
            "FINISH": "save_offer_gate",
        },
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
    plan/evaluation/last_report/train_confirmed/pending_operation/
    train_confirmed_operation_id/last_trained_model/model_save_offered 每次新訊息都明確重置，這些欄位
    是「這一次任務」用的保險絲，不該延續到下一次使用者發的新任務（resume 不算
    新任務，不會重置這些欄位，接著用暫停當下的狀態繼續——train_confirmed 也是
    因為這樣才能在同一個任務裡重試訓練時不用重複跳出確認畫面）。"""
    all_messages = [HumanMessage(content=user_message)] if user_message is not None else []
    seed_len = len(all_messages)
    config = {"configurable": {"thread_id": thread_id}}
    interrupted = False

    try:
        # graph.stream() 的第一個 update 要等 Planner 完成一次 LLM 呼叫才會出現。
        # 先送出一個狀態事件，讓前端在這段等待期間仍能明確顯示 Agent 正在規劃，
        # 而不是只在 Observer 完成後才看到任何回饋。
        yield {
            "type": "status",
            "stage": "planner",
            "message": "Manager 正在理解需求並制定執行計畫…",
        }
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
                "train_confirmed_operation_id": None,
                "pending_operation": None,
                "last_trained_model": None,
                "eval_charts": None,
                "model_save_offered": False,
                "augment_offered": False,
                "augment_baseline_count": None,
                "roboflow_proposal": None,
                "roboflow_proposal_handled": False,
                "dataset_refresh": None,
                "post_augment_eval_required": False,
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
                yield {
                    "type": "interrupt",
                    "interrupt_id": interrupt_obj.id,
                    "payload": interrupt_obj.value,
                }
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

                # 評估圖表（混淆矩陣／F1／P-R 曲線）由後端直接把檔案 URL 送給前端
                # 畫出來，不經過 LLM 轉述——路徑要是讓模型自己寫，寫錯就變成破圖。
                if node_output.get("eval_charts"):
                    yield {"type": "eval_charts", **node_output["eval_charts"]}
                if node_output.get("dataset_refresh"):
                    yield {"type": "dataset_refresh", **node_output["dataset_refresh"]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        yield {"type": "error", "message": str(e)}
        return

    if interrupted:
        return

    reply = _extract_text(all_messages[-1].content) if len(all_messages) > seed_len else ""
    yield {"type": "final", "reply": reply, "usage": total_usage, "llm_calls": llm_calls}
