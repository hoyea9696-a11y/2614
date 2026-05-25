import os
import yaml
import shutil
import random
import threading
from pathlib import Path
from ultralytics import YOLO

class YOLOTrainer:
    def __init__(self, data_root=None):
        self.model = None
        self.progress = {
            "status": "idle",
            "message": "",
            "current_epoch": 0,
            "total_epochs": 0,
            "metrics": []
        }
        self._stop_event = threading.Event()
        self.temp_root = Path("temp_training")
        self.temp_root.mkdir(parents=True, exist_ok=True)

    def prepare_dataset(self, dataset_path, val_split=0.2):
        """同步自 standalone_train.py 的成功邏輯"""
        ds_root = Path(dataset_path)
        
        # 1. 優先找 YAML
        existing_yamls = list(ds_root.glob("*.yaml"))
        class_names = {0: 'object'}
        if existing_yamls:
            try:
                with open(existing_yamls[0], 'r', encoding='utf-8') as f:
                    y_data = yaml.safe_load(f)
                    if 'names' in y_data: class_names = y_data['names']
            except: pass

        # 2. 檢查是否已經有傳統的 train/valid 目錄結構
        def safe_check(p):
            try: return (ds_root / p).exists()
            except: return False

        train_path = "train/images" if safe_check("train/images") else ("train" if safe_check("train") else None)
        val_path = "valid/images" if safe_check("valid/images") else ("valid" if safe_check("valid") else None)
        test_path = "test/images" if safe_check("test/images") else ("test" if safe_check("test") else None)

        # 3. 產生存取最安全的 YAML (放在專案目錄下)
        temp_yaml = Path(f"dataset_{ds_root.stem}.yaml").absolute()
        
        # 4. 如果沒有切好的目錄，啟動「虛擬 TXT 切割法」
        if not train_path or not val_path:
            print("--- [Dataset] 未偵測到預設切割目錄，啟動自動虛擬切割 (TXT Split) ---")
            train_path, val_path, test_path = self._generate_split_txt(ds_root)

        config = {
            "path": str(ds_root.absolute()),
            "train": str(train_path) if train_path else "",
            "val": str(val_path) if val_path else "",
            "names": class_names
        }
        
        if test_path:
            config["test"] = str(test_path)
        
        with open(temp_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True)
            
        print(f"Web Trainer: YAML generated at {temp_yaml}")
        return str(temp_yaml)

    def _generate_split_txt(self, ds_root):
        """自動掃描資料集，隨機分配 80% Train, 10% Val, 10% Test 並寫入 TXT"""
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        valid_images = []
        
        # 找出所有具備標記檔的圖片
        for ext in image_extensions:
            for img_p in ds_root.rglob(f"*{ext}"):
                label_p = img_p.parent.parent / "labels" / (img_p.stem + ".txt") if "images" in str(img_p) else img_p.parent / (img_p.stem + ".txt")
                if label_p.exists() and label_p.stat().st_size > 0:
                    valid_images.append(str(img_p.absolute()))
                    
        if not valid_images:
            raise Exception("資料集內沒有找到任何已標記的圖片！請先進行標記。")
            
        # 打亂順序
        random.seed(42) # 固定隨機種子確保每次切法一致
        random.shuffle(valid_images)
        
        total = len(valid_images)
        train_idx = int(total * 0.8)
        val_idx = int(total * 0.9) # 剩下的 10% 給 test
        
        train_imgs = valid_images[:train_idx]
        val_imgs = valid_images[train_idx:val_idx] if total > 5 else valid_images[train_idx:]
        test_imgs = valid_images[val_idx:] if total > 10 else []
        
        # 寫入 txt 檔案
        train_txt = ds_root / "train.txt"
        val_txt = ds_root / "val.txt"
        test_txt = ds_root / "test.txt"
        
        with open(train_txt, 'w', encoding='utf-8') as f: f.write('\n'.join(train_imgs))
        with open(val_txt, 'w', encoding='utf-8') as f: f.write('\n'.join(val_imgs))
        if test_imgs:
            with open(test_txt, 'w', encoding='utf-8') as f: f.write('\n'.join(test_imgs))
            
        print(f"--- [Split Result] Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)} ---")
        
        return str(train_txt.absolute()), str(val_txt.absolute()), str(test_txt.absolute()) if test_imgs else None

    def _check_stop(self, trainer):
        if self._stop_event.is_set():
            raise Exception("USER_STOPPED")
            
    def _update_progress(self, trainer):
        if self._stop_event.is_set():
            raise Exception("USER_STOPPED")
            
        try:
            epoch = trainer.epoch + 1
            map_val = 0
            if hasattr(trainer, 'validator') and hasattr(trainer.validator, 'metrics'):
                res_data = trainer.validator.metrics.results_dict
                if callable(res_data): results = res_data()
                else: results = res_data
                map_val = results.get('metrics/mAP50(B)', 0) * 100

            loss_val = 0
            if hasattr(trainer, 'loss_items'):
                loss_val = float(trainer.loss_items[0]) if hasattr(trainer.loss_items, '__getitem__') else 0

            self.progress["current_epoch"] = epoch
            self.progress["metrics"].append({
                "epoch": epoch,
                "map": round(float(map_val), 2),
                "loss": round(float(loss_val), 4)
            })
            print(f"Epoch {epoch}: mAP={map_val:.1f}%, Loss={loss_val:.4f}")
        except Exception as e:
            if "USER_STOPPED" in str(e): raise e
            print(f"Callback error: {e}")

    def train(self, variant="yolov9m.pt", data_yaml="dataset.yaml", epochs=50, imgsz=640, batch=16, patience=50, model_name="exp"):
        print(f"\n🚀 WEB 訓練任務啟動: {variant}")
        self.progress.update({"status": "training", "total_epochs": epochs, "metrics": [], "current_epoch": 0})
        
        try:
            project_root = Path(__file__).resolve().parent.parent
            runs_dir = project_root / "runs" / "train"
            
            model_path = variant
            if not os.path.isabs(model_path):
                potential = project_root / "04_資料與模型" / "models" / variant
                if potential.exists(): model_path = str(potential.absolute())

            self.model = YOLO(model_path)
            self._stop_event.clear()
            self.model.add_callback("on_fit_epoch_end", self._update_progress)
            self.model.add_callback("on_train_batch_end", self._check_stop)
            
            import torch
            device = 0 if torch.cuda.is_available() else 'cpu'
            
            self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                patience=patience,
                project=str(runs_dir),
                name=model_name,
                exist_ok=True,
                verbose=True,
                device=device
            )
            self.progress["status"] = "completed"
        except Exception as e:
            if "USER_STOPPED" in str(e):
                self.progress["status"] = "stopped"
                self.progress["message"] = "訓練已由使用者手動終止"
            else:
                print(f"TRAIN ERROR: {e}")
                self.progress["status"] = "error"
                self.progress["message"] = str(e)

    def stop(self):
        self._stop_event.set()
