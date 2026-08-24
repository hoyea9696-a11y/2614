import os
import json
import requests
from openai import OpenAI
from pathlib import Path

# 手動載入 .env
if os.path.exists(".env"):
    with open(".env", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

class AIService:
    def __init__(self):
        pass

    def _get_keys(self):
        # 使用絕對路徑定位根目錄的 .env
        base_dir = Path(__file__).resolve().parent.parent
        env_path = base_dir / ".env"
        
        keys = {"gemini": os.getenv("GEMINI_API_KEY"), "groq": os.getenv("GROQ_API_KEY"), "openai": os.getenv("OPENAI_API_KEY")}
    def _get_config(self):
        # [核心修正] 強制使用絕對路徑定位根目錄的 .env
        base_dir = Path(__file__).resolve().parent.parent
        env_path = base_dir / ".env"
        
        # 初始預設值
        config = {
            "gemini_api_key": os.getenv("GEMINI_API_KEY"),
            "groq_api_key": os.getenv("GROQ_API_KEY"),
            "gemini_flash_model": os.getenv("GEMINI_FLASH_MODEL", "gemini-2.5-flash"),
            "gemini_pro_model": os.getenv("GEMINI_PRO_MODEL", "gemini-2.5-pro"),
            "groq_model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "ollama_model": os.getenv("OLLAMA_MODEL", "llama3.2"),
            "ollama_base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        }
        
        if env_path.exists():
            print(f"--- [AIService] 發現設定檔: {env_path}")
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        config[k.strip().lower()] = v.strip().strip('"').strip("'")
        else:
            print(f"--- [AIService] 警告: 找不到設定檔 {env_path}")
            
        return config

    def chat(self, messages: list, context_data: dict = None, model_type: str = "gemini") -> str:
        platform = model_type.lower()
        conf = self._get_config()
        
        # 決定具體的 Gemini 系列模型
        gemini_model = conf.get("gemini_flash_model", "gemini-2.5-flash")
        if platform == "gemini-pro":
            gemini_model = conf.get("gemini_pro_model", "gemini-2.5-pro")
            platform = "gemini"
        
        # 從 context_data 讀取已由 main.py 準備好的健康數據
        health_info = ""
        health_json = ""
        
        if context_data and "health_data" in context_data:
            stats = context_data["health_data"]
            ds_name = context_data.get("activeDataset", "未知")
            if stats.get("status") == "success":
                # 建立精簡版的摘要供 AI 閱讀，避免 Prompt 爆炸
                summary_list = [f"{v['name']}: {v['count']}個" for v in stats['class_distribution'].values()]
                health_info = f"\n【資料集即時診斷摘要: {ds_name}】\n"
                health_info += f"- 健康總分: {stats['score']}/100\n"
                health_info += f"- 圖片總數: {stats['total_images']} 張\n"
                health_info += f"- 已標記圖片數: {stats['labeled_images']} 張\n"
                health_info += f"- 標記覆蓋率: {stats['coverage']}%\n"
                health_info += f"- 類別分佈摘要: {', '.join(summary_list)}\n"
                health_json = json.dumps(stats, ensure_ascii=False)

        # 從 context_data 讀取已由 main.py 準備好的模型訓練數據
        model_info = ""
        if context_data and "model_eval_data" in context_data:
            m = context_data["model_eval_data"]
            model_info = f"\n【歷史訓練紀錄（最近一次已完成的訓練，非正在進行）- 模型名稱: {m.get('model_name', '未知')}】\n"
            model_info += f"- 已訓練輪次: {m['epoch']}\n"
            model_info += f"- 精準度 (mAP50): {m['map50']*100:.2f}%\n"
            model_info += f"- 嚴格精準度 (mAP50-95): {m['map50_95']*100:.2f}%\n"
            model_info += f"- 訓練損失 (Loss): {m['train_loss']:.4f}\n"
            model_info += f"- 驗證損失 (Val Loss): {m['val_loss']:.4f}\n"

        # 系統提示詞構造
        system_prompt = f"""你是一個專業的 AI 影像辨識專家與資料分析師。
目前系統狀態:
- 當前活動資料集: {context_data.get('activeDataset', '未選擇') if context_data else '未選擇'}
{health_info}
{model_info}

【對話準則】
0. **【絕對禁止臆測數據】**：所有具體數字（圖片張數、類別數、標註率、解析度、mAP、Loss…）
   只能引用上面【】區塊裡真的出現的數值，一個字都不能改。上面沒有提供的數據，就直接說
   「我目前沒有拿到這個資料集／模型的實際數據」，**絕對不可以自己估算、舉例或編造**。
   你也不可以自行輸出 `[HEALTH_DATA: ...]` 標籤——那是系統用來畫健康圖卡的專用標記，
   你冒用會讓使用者看到假數據。
1. **【診斷優先級】**：如果同時提供了 [健康數據] 與 [模型效能數據]，且使用者詢問的是「模型」或「診斷」，請**務必優先分析模型效能**。不要在模型診斷時夾雜過多的資料集健康資訊，除非使用者特別要求。
2. **【詳細報告要求】**：如果提供了 [歷史訓練紀錄]，請評論 mAP 的高低，並觀察 Loss 趨勢（若訓練與驗證 Loss 差距大，代表 Overfitting）。注意那是過去已完成的訓練結果，不是正在進行的訓練。
3. **【標籤處理】**：健康圖卡由系統自行附加，你只要用文字說明即可，不要輸出任何 `[HEALTH_DATA: ...]`。
4. **【長度要求】**：回答要精簡：先講結論，再用條列補充重點，全文以 300 字內為原則；只有使用者明確要求詳細說明時才展開。這是聊天視窗，不是技術文件。
5. **【能力邊界】**：使用者要求訓練模型時，簡短回應即可（一兩句話），不要輸出長篇訓練教學。訓練是否真的啟動由系統自行判斷並告知使用者；你「絕對不要」自己輸出「[系統提示]」或「⚙️」這類字樣，那是系統保留的專用標記，你冒用會讓使用者混淆。
"""

        try:
            if platform == "gemini":
                reply = self._chat_gemini_rest(messages, system_prompt, conf["gemini_api_key"], gemini_model)
            elif platform == "groq":
                # 跟 Manager Agent 的 gpt-oss 引擎用同一顆模型，方便做有無 LangGraph 的對比
                client = OpenAI(api_key=conf["groq_api_key"], base_url="https://api.groq.com/openai/v1")
                reply = self._chat_openai(client, conf.get("gpt_oss_model", "openai/gpt-oss-120b"), messages, system_prompt)
            else:
                client = OpenAI(api_key="ollama", base_url=conf["ollama_base_url"])
                reply = self._chat_openai(client, conf["ollama_model"], messages, system_prompt)
            
            # Debug: 確保我們在後端能看到完整的 reply
            print(f">>> [AIService] Reply length: {len(reply)}")
            return reply
        except Exception as e:
            return f"[{platform.upper()} 錯誤]: {str(e)}"

    def _chat_gemini_rest(self, messages, system_prompt, api_key, model_name):
        # 切換回 v1beta 以支援 Gemini 2.0 模型
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
        contents = []
        
        # 注入系統狀態作為背景
        contents.append({"role": "user", "parts": [{"text": f"【背景資訊與設定】\n{system_prompt}\n\n請根據以上資訊與使用者對話。"}]})
        contents.append({"role": "model", "parts": [{"text": "好的，我已經準備好協助你分析資料集與訓練狀態了。"}]})

        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        # Gemini 2.5 是思考型模型，思考 token 也計入 maxOutputTokens；設太低會讓
        # 正文在思考耗掉額度後被硬生生截斷（finishReason=MAX_TOKENS）。
        payload = {"contents": contents, "generationConfig": {"temperature": 0.8, "maxOutputTokens": 8192}}
        headers = {"Content-Type": "application/json"}
        response = requests.post(url, json=payload, headers=headers)
        res_data = response.json()
        
        if response.status_code != 200:
            return f"Gemini API 錯誤 ({response.status_code}): {json.dumps(res_data)}"
        try:
            return res_data["candidates"][0]["content"]["parts"][0]["text"]
        except:
            return f"解析回應失敗: {json.dumps(res_data)}"

    def chat_with_image(self, prompt: str, base64_image: str) -> str:
        conf = self._get_config()
        # 視覺標記同樣切換回 v1beta
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{conf.get('gemini_flash_model', 'gemini-2.5-flash')}:generateContent?key={conf['gemini_api_key']}"
        
        payload = {
            "contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": base64_image}}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 1000}
        }
        
        try:
            response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
            res_data = response.json()
            return res_data["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            print(f"[Gemini Vision Error]: {str(e)}")
            return ""

    def _chat_openai(self, client, model_name, messages, system_prompt):
        try:
            # 這是 Groq (Llama) 看到資料集的關鍵：將 system_prompt 放入訊息列表首位
            formatted_msgs = [{"role": "system", "content": system_prompt}]
            for msg in messages:
                formatted_msgs.append({"role": msg["role"], "content": msg["content"]})
            
            response = client.chat.completions.create(
                model=model_name,
                messages=formatted_msgs,
                temperature=0.8
            )
            return response.choices[0].message.content
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            if "AuthenticationError" in error_type or "401" in error_msg:
                return f"[Groq 認證錯誤]: 您的 API Key 可能無效或已過期，請檢查 .env 檔案。"
            elif "RateLimitError" in error_type or "429" in error_msg:
                return f"[Groq 額度錯誤]: 您的 Groq 帳戶額度已達上限或餘額不足。"
            elif "NotFoundError" in error_type or "404" in error_msg:
                return f"[Groq 模型錯誤]: 找不到模型 '{model_name}'，請確認該模型是否存在。"
            elif "APIConnectionError" in error_type:
                return f"[Groq 連線錯誤]: 無法連線至 Groq 伺服器，請檢查網路設定或代理伺服器。"
            
            return f"[Groq 未知錯誤]: {error_type} - {error_msg}"

ai_engine = AIService()
