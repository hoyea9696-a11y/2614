import os
import sys
import yaml
import torch
from pathlib import Path
from ultralytics import YOLO

def diagnostic_train():
    print("="*60)
    print("🚀 AI Detector 獨立診斷訓練工具")
    print("="*60)

    # 1. 設定路徑 (請確認此路徑正確)
    dataset_path = r"D:\AI-project\data\datasets\my_custom_model\Flowers Detection.v1-roboflow-instant-1--eval-.yolov8"
    ds_root = Path(dataset_path)
    
    print(f"[*] 檢查資料集根目錄: {ds_root}")
    if not ds_root.exists():
        print(f"❌ 錯誤: 找不到資料夾 {ds_root}")
        return

    # 2. 智慧偵測 (加入防護)
    def safe_check(p):
        try:
            return (ds_root / p).exists()
        except:
            return False

    train_path = "train/images" if safe_check("train/images") else "train"
    val_path = "valid/images" if safe_check("valid/images") else "valid"
    
    print(f"[*] 偵測到結構: train={train_path}, val={val_path}")

    # 3. 讀取類別名稱
    names = {0: 'object'}
    existing_yamls = list(ds_root.glob("*.yaml"))
    if existing_yamls:
        try:
            with open(existing_yamls[0], 'r', encoding='utf-8') as f:
                y_data = yaml.safe_load(f)
                if 'names' in y_data:
                    names = y_data['names']
                    print(f"[*] 已從 {existing_yamls[0].name} 讀取到 {len(names)} 個類別")
        except: pass

    # 4. 產生純淨 YAML (放在專案根目錄，遠離 OneDrive)
    temp_yaml = Path("debug_dataset.yaml").absolute()
    config = {
        "path": str(ds_root.absolute()),
        "train": train_path,
        "val": val_path,
        "names": names
    }
    
    with open(temp_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True)
    
    print(f"[*] 已生成偵錯 YAML: {temp_yaml}")

    # 5. 啟動訓練
    try:
        print("\n[*] 正在載入 YOLO 模型...")
        model = YOLO("yolov9m.pt")
        
        print("[*] 準備開始訓練 (這可能會觸發掃描)...")
        device = 0 if torch.cuda.is_available() else 'cpu'
        
        model.train(
            data=str(temp_yaml),
            epochs=1,        # 偵測用，先跑 1 輪就好
            imgsz=640,
            batch=4,         # 降低負載
            device=device,
            project="runs/debug",
            name="test_run",
            exist_ok=True,
            verbose=True
        )
        print("\n✅ 診斷訓練成功啟動！這代表檔案結構已經沒問題了。")
        
    except Exception as e:
        print(f"\n❌ 訓練失敗！")
        print(f"錯誤類型: {type(e).__name__}")
        print(f"詳細錯誤訊息: {str(e)}")
        print("\n[建議] 如果依然是 images/val 存取被拒，請嘗試手動刪除該資料夾，或將整個資料集移出 OneDrive 範圍。")

if __name__ == "__main__":
    diagnostic_train()
