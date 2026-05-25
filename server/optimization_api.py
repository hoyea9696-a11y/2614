from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from ai_analyzer import ai_analyzer
from ai_service import ai_engine
import os

router = APIRouter(prefix="/api/optimization", tags=["optimization"])

@router.get("/diagnose")
async def diagnose_training(exp_name: str = "exp"):
    """
    結合 AIAnalyzer 與 AIService 進行深度診斷
    """
    df = ai_analyzer.get_latest_results(exp_name)
    if df is None:
        return {"status": "error", "message": "找不到訓練結果文件 (results.csv)"}
    
    summary = ai_analyzer.summarize_metrics(df)
    
    # 建立給 LLM 的 Prompt
    prompt = f"""
    身為 AI 訓練專家，請根據以下訓練數據給出專業診斷與下一步優化建議。
    數據摘要：{summary}
    
    請包含以下內容：
    1. 訓練現況評價
    2. 是否建議繼續訓練或停止
    3. 下次訓練的參數調整建議 (LR, Augmentation, etc.)
    """
    
    messages = [{"role": "user", "content": prompt}]
    ai_reply = ai_engine.chat(messages)
    
    return {
        "status": "success",
        "metrics_summary": summary,
        "ai_diagnosis": ai_reply
    }

@router.get("/suggest_params")
async def suggest_params(dataset_name: str):
    """
    根據資料集規模建議初始參數
    """
    # 這裡未來可以串接更多資料集分析邏輯
    prompt = f"我的資料集名稱是 {dataset_name}。請根據一般 YOLO 經驗建議一組適合的訓練參數 (Epochs, Batch, Imgsz)。"
    messages = [{"role": "user", "content": prompt}]
    ai_reply = ai_engine.chat(messages)
    
    return {"status": "success", "suggestions": ai_reply}
