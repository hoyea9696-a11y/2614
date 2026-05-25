import pandas as pd
import os
from pathlib import Path

class AIAnalyzer:
    def __init__(self, runs_dir=None):
        base_dir = Path(__file__).resolve().parent.parent
        self.runs_dir = Path(runs_dir) if runs_dir else base_dir / "runs" / "train"

    def get_latest_results(self, exp_name="exp"):
        """讀取最新的 YOLO 訓練結果 CSV"""
        csv_path = self.runs_dir / exp_name / "results.csv"
        if not csv_path.exists():
            return None
        
        try:
            df = pd.read_csv(csv_path)
            # 移除列名中的空白
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:
            print(f"Error reading results.csv: {e}")
            return None

    def summarize_metrics(self, df):
        """將數據轉化為 AI 易懂的文字摘要"""
        if df is None or df.empty:
            return "目前尚無訓練數據。"

        last_row = df.iloc[-1]
        summary = {
            "total_epochs": len(df),
            "final_map50": round(last_row.get("metrics/mAP50(B)", 0), 4),
            "final_loss": round(last_row.get("train/box_loss", 0) + last_row.get("train/cls_loss", 0), 4),
            "map_trend": "上升" if len(df) > 5 and df["metrics/mAP50(B)"].iloc[-1] > df["metrics/mAP50(B)"].iloc[-5] else "平穩或波動"
        }
        
        # 簡單的過擬合檢查
        if len(df) > 10:
            val_loss_recent = df["val/box_loss"].iloc[-5:].mean()
            val_loss_prev = df["val/box_loss"].iloc[-10:-5].mean()
            if val_loss_recent > val_loss_prev * 1.05:
                summary["warning"] = "偵測到驗證集損失上升，可能有過擬合風險。"

        return summary

    def get_dataset_classes(self, ds_name, datasets_dir=None):
        """取得特定資料集的類別清單"""
        if datasets_dir is None:
            base_dir = Path(__file__).resolve().parent.parent
            datasets_dir = base_dir / "data" / "datasets"
        
        base_path = Path(datasets_dir) / ds_name
        if not base_path.exists():
            return []
            
        # 1. 先嘗試找純文字格式的類別檔
        class_files = ["classes.txt", "_classes.txt", "obj.names"]
        for cf in class_files:
            p = base_path / cf
            if p.exists():
                with open(p, 'r', encoding='utf-8') as f:
                    return [l.strip() for l in f.readlines() if l.strip()]
                    
        # 2. 如果沒有 txt 檔，則解析 YAML 檔 (標準 YOLO 格式)
        import yaml
        yaml_files = ["data.yaml", "dataset.yaml"]
        for yf in yaml_files:
            p = base_path / yf
            if p.exists():
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                        if data and 'names' in data:
                            names = data['names']
                            if isinstance(names, list):
                                return names
                            elif isinstance(names, dict):
                                return list(names.values())
                except Exception as e:
                    print(f"Error reading YAML {yf}: {e}")
        return []

    def analyze_dataset_health(self, ds_name):
        """深度分析資料集健康度 (類別平衡、標記率等)"""
        base_dir = Path(__file__).resolve().parent.parent
        ds_path = base_dir / "data" / "datasets" / ds_name
        
        if not ds_path.exists():
            return {"status": "error", "message": "資料集不存在"}

        # 1. 取得基本類別對照
        class_names = self.get_dataset_classes(ds_name)
        # 2. 掃描圖片與標籤 (包含更多副檔名以防漏算)
        image_exts = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG"]
        images_set = set()
        for ext in image_exts:
            for p in ds_path.rglob(ext):
                images_set.add(str(p.resolve()))
                
        total_images = len(images_set)
        
        # 建立一個所有圖片主檔名（不含副檔名）的集合，用於驗證標籤
        image_stems = set(Path(p).stem for p in images_set)
        
        labels = list(ds_path.rglob("*.txt"))
        labeled_images = 0
        class_counts = {} # 儲存每個類別出現的次數
        
        # 初始化類別統計
        for i, name in enumerate(class_names):
            class_counts[str(i)] = {"name": name, "count": 0}

        # 3. 讀取所有標籤檔案進行統計
        for lp in labels:
            # 排除非標籤檔或空白檔
            if lp.name in ["classes.txt", "_classes.txt", "notes.txt", "README.txt"] or lp.stat().st_size == 0:
                continue
            
            # 核心修正：只有當這個標籤檔有對應的圖片檔案存在時，才算作「已標記圖片」
            if lp.stem not in image_stems:
                print(f"--- [Health Check] 發現孤立標籤檔（無對應圖片）: {lp.name}，已排除計數 ---")
                continue
            
            labeled_images += 1
            try:
                with open(lp, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = parts[0]
                            if cls_id in class_counts:
                                class_counts[cls_id]["count"] += 1
                            else:
                                # 處理未定義在 classes.txt 裡的 ID
                                if cls_id not in class_counts:
                                    class_counts[cls_id] = {"name": f"Unknown-{cls_id}", "count": 1}
                                else:
                                    class_counts[cls_id]["count"] += 1
            except:
                continue

        # 4. 計算評分 (0-100)
        # 基本分：覆蓋率 (防止因標籤檔比圖片多而超過 100%)
        coverage_score = min(100.0, (labeled_images / total_images * 100)) if total_images > 0 else 0
        
        # 平衡分：計算變異係數 (Coefficient of Variation) 簡化版
        counts = [v["count"] for v in class_counts.values()]
        balance_score = 100
        if len(counts) > 1 and sum(counts) > 0:
            avg = sum(counts) / len(counts)
            diffs = sum(abs(c - avg) for c in counts)
            balance_score = max(0, 100 - (diffs / sum(counts) * 100))

        final_score = (coverage_score * 0.4) + (balance_score * 0.6)

        return {
            "status": "success",
            "score": round(final_score, 1),
            "total_images": total_images,
            "labeled_images": labeled_images,
            "coverage": round(coverage_score, 1),
            "balance": round(balance_score, 1),
            "class_distribution": class_counts
        }

    def get_all_datasets_info(self, datasets_dir=None):
        """掃描所有資料集，取得圖片張數與類別資訊"""
        base_dir = Path(__file__).resolve().parent.parent
        base_path = Path(datasets_dir) if datasets_dir else base_dir / "data" / "datasets"
        
        if not base_path.exists():
            return []
        
        info_list = []
        for ds_dir in base_path.iterdir():
            if ds_dir.is_dir() and not ds_dir.name.startswith("."):
                image_exts = ["*.jpg", "*.jpeg", "*.png", "*.webp", "*.JPG", "*.JPEG", "*.PNG"]
                images_set = set()
                for ext in image_exts:
                    for p in ds_dir.rglob(ext):
                        images_set.add(str(p.resolve()))
                img_count = len(images_set)
                
                classes = self.get_dataset_classes(ds_dir.name, datasets_dir=str(base_path))
                
                info_list.append({
                    "name": ds_dir.name,
                    "image_count": img_count,
                    "classes": classes,
                    "has_labels": len(list(ds_dir.rglob("*.txt"))) > 0
                })
        return info_list

ai_analyzer = AIAnalyzer()
