import cv2
import numpy as np
from ultralytics import YOLO
import os
from pathlib import Path

class YOLOInferencer:
    def __init__(self, model_path=None):
        if model_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            model_path = str(base_dir / "models" / "yolov9m.pt")
        self.model_path = model_path
        self._model = None  # 延遲載入：只有真的用到推論/訓練相關功能時才載入權重，
        # 不要在建構子就載入——main.py 用 run_multi_port.py 開 4 個埠口時，每個埠口
        # 都各自 import 一次 main.py 就會各自建立一個 inferencer，若在這裡就載入權重，
        # 4 個埠口（其中只有推論頁真的需要模型）就會各自載入一次 YOLO，白白拖慢啟動、
        # 佔用 4 倍記憶體。

    @property
    def model(self):
        if self._model is None:
            self.load_model()
        return self._model

    def load_model(self):
        # Force directory creation
        abs_path = os.path.abspath(self.model_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        
        print(f"--- Model Engine Status ---")
        print(f"Target Path: {abs_path}")
        
        if not os.path.exists(abs_path):
            print(f"Weight file not found in models/ folder. Syncing from engine...")
            model_name = os.path.basename(self.model_path)
            
            # Load by name to let YOLO find it (download or cache)
            temp_model = YOLO(model_name) 
            
            # Get the actual path where YOLO found/saved the weights
            # In newer ultralytics, it's often in model.ckpt_path
            source_path = getattr(temp_model, 'ckpt_path', None)
            
            if source_path and os.path.exists(source_path):
                print(f"Found weights at {source_path}. Copying to models/ folder...")
                import shutil
                shutil.copy(source_path, abs_path)
            else:
                # Fallback: check current directory
                if os.path.exists(model_name):
                    import shutil
                    shutil.move(model_name, abs_path)
        
        try:
            self._model = YOLO(self.model_path)
            print(f"Model loaded successfully from: {self.model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    def predict(self, image_source, conf=0.25, iou=0.45, augment=False, imgsz=640):
        """
        image_source: can be a path or a numpy image
        returns: detection results and an annotated image
        """
        results = self.model.predict(image_source, conf=conf, iou=iou, augment=augment, imgsz=imgsz)
        
        # results list of Results objects
        res = results[0]
        # Enhance box visibility with line_width and labels
        annotated_img = res.plot(line_width=3, font_size=1.2) # Returns BGR numpy array
        
        # Convert detections to a serializable format
        detections = []
        for box in res.boxes:
            detections.append({
                "class_id": int(box.cls[0]),
                "name": self.model.names[int(box.cls[0])],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist() # [x1, y1, x2, y2]
            })
            
        return detections, annotated_img

    def switch_model(self, model_path):
        self.model_path = model_path
        self.load_model()

    def auto_label_gemini(self, dataset_dir, target_classes, ai_service, progress_callback=None):
        """
        使用 Gemini 視覺能力進行標記 (真正具備理解力的標記)
        """
        import os, base64, time
        from pathlib import Path
        
        ds_path = Path(dataset_dir)
        image_extensions = [".jpg", ".jpeg", ".png", ".webp"]
        images = [f for f in ds_path.rglob("*") if f.suffix.lower() in image_extensions]
        total = len(images)
        labeled_count = 0
        
        # 建立清單
        class_map_str = ", ".join([f"{i}: {name}" for i, name in enumerate(target_classes)])
        
        print(f"--- [Gemini Auto-Label] 啟動 ---")
        print(f"限制類別: {class_map_str}")
        
        use_yolo_for_remaining = False

        for idx, img_path in enumerate(images):
            # 檢查標籤路徑
            txt_path = img_path.with_suffix(".txt")
            if "images" in str(img_path):
                txt_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
            
            if txt_path.exists() and txt_path.stat().st_size > 0:
                labeled_count += 1
                if progress_callback: progress_callback(idx + 1, total, labeled_count)
                continue

            # [優化] 如果已經偵測到 API 失效，直接用 YOLO
            if use_yolo_for_remaining:
                self._label_single_image_yolo(img_path, txt_path, target_classes)
                labeled_count += 1
                if progress_callback: progress_callback(idx + 1, total, labeled_count)
                continue

            gemini_success = False
            # 嘗試 Gemini
            try:
                with open(img_path, "rb") as f:
                    img_data = base64.b64encode(f.read()).decode("utf-8")

                prompt = f"""作為影像標記專家，請偵測圖中所有屬於以下類別的物件：[{class_map_str}]。
請嚴格按照 YOLO 格式輸出結果：[class_id x_center y_center width height]
規則：座標歸一化, 禁止輸出清單外 ID, 沒發現回 NONE, 僅輸出數字。"""

                res_txt = ai_service.chat_with_image(prompt, img_data)
                
                if not res_txt or "429" in res_txt or "quota" in res_txt.lower() or "error" in res_txt.lower():
                    print(f"  [Alert] Gemini 異常或配額耗盡，當下切換至 YOLO...")
                    use_yolo_for_remaining = True
                    self._label_single_image_yolo(img_path, txt_path, target_classes)
                    labeled_count += 1
                    gemini_success = True 
                else:
                    if "NONE" not in res_txt.upper():
                        lines = [l.strip() for l in res_txt.split("\n") if l.strip() and l.strip()[0].isdigit()]
                        if lines:
                            txt_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(txt_path, "w", encoding="utf-8") as f: f.write("\n".join(lines))
                            labeled_count += 1
                            print(f"  [Smart OK] {img_path.name}")
                    gemini_success = True
            except Exception as e:
                print(f"  [Error] Gemini 標記出錯: {e}，當下切換至 YOLO...")
                use_yolo_for_remaining = True
                self._label_single_image_yolo(img_path, txt_path, target_classes)
                labeled_count += 1
                gemini_success = True

            # 如果 Gemini 失敗且尚未成功標記，則在這裡補執行單張 YOLO
            if not gemini_success:
                self._label_single_image_yolo(img_path, txt_path, target_classes)
                labeled_count += 1

            if progress_callback:
                progress_callback(idx + 1, total, labeled_count)
        
        return labeled_count

    def _label_single_image_yolo(self, img_path, txt_path, target_classes):
        """ YOLO 本地標記輔助方法 """
        detections, _ = self.predict(str(img_path), conf=0.3)
        if not detections: return
        class_map = {name.lower(): i for i, name in enumerate(target_classes)}
        yolo_lines = []
        try:
            import cv2
            img_cv = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
            h_img, w_img = img_cv.shape[:2]
            for det in detections:
                det_name = det["name"].lower()
                if det_name in class_map:
                    fid = class_map[det_name]
                    x1, y1, x2, y2 = det["bbox"]
                    xc, yc = (x1+x2)/2/w_img, (y1+y2)/2/h_img
                    w, h = (x2-x1)/w_img, (y2-y1)/h_img
                    yolo_lines.append(f"{fid} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            if yolo_lines:
                txt_path.parent.mkdir(parents=True, exist_ok=True)
                with open(txt_path, "w", encoding="utf-8") as f: f.write("\n".join(yolo_lines))
        except: pass

    def auto_label(self, dataset_dir, target_classes=None, progress_callback=None):
        """
        遍歷資料集，對未標記的圖片進行自動偵測並儲存
        dataset_dir: 資料集絕對路徑
        target_classes: 資料集原本定義的類別清單 (用於 ID 轉換)
        progress_callback: 用於回報進度的 function(processed, total, labeled_count)
        """
        import os
        from pathlib import Path
        
        ds_path = Path(dataset_dir)
        # 搜尋圖片
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        images = []
        for ext in image_extensions:
            images.extend(list(ds_path.rglob(f"*{ext}")))
            
        total = len(images)
        print(f"--- [Auto Label] 啟動 ---")
        print(f"Target: {dataset_dir}, Found {total} images")
        
        # 確保 target_classes 是一個可變清單
        if target_classes is None:
            target_classes = []
        else:
            target_classes = list(target_classes)
        
        initial_class_count = len(target_classes)

        count = 0
        processed = 0
        for img_p in images:
            processed += 1
            if progress_callback:
                progress_callback(processed, total, count)

            # 決定標籤路徑
            img_rel = img_p.relative_to(ds_path)
            label_rel = str(img_rel).replace("images", "labels")
            label_p = ds_path / (os.path.splitext(label_rel)[0] + ".txt")
            
            # 如果標籤已存在且不為空，跳過
            if label_p.exists() and label_p.stat().st_size > 0:
                continue
                
            # 執行推理
            try:
                # 恢復原始門檻值，確保 YOLO 能正常偵測
                clean_img_path = str(img_p).replace("\\\\", "\\")
                detections, _ = self.predict(clean_img_path, conf=0.25, iou=0.45)
            except Exception as e:
                print(f"Prediction failed for {img_p}: {e}")
                continue
            
            if not detections:
                # 即使沒偵測到，也要回報進度，讓前端進度條前進
                if progress_callback:
                    progress_callback(processed, total, count)
                continue


                
            yolo_lines = []
            try:
                img_array = np.fromfile(str(img_p), dtype=np.uint8)
                img_cv = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img_cv is None: continue
                h_img, w_img = img_cv.shape[:2]
            except: continue
            
            for det in detections:
                det_name = det["name"].strip()
                
                # 智慧配對與自動新增
                found_idx = -1
                for idx, tc in enumerate(target_classes):
                    if det_name.lower() == tc.lower().strip():
                        found_idx = idx
                        break
                
                if found_idx != -1:
                    final_class_id = found_idx
                else:
                    # [核心變更]：如果類別不存在，自動新增它！
                    print(f"--- [Auto Class] 新增類別: {det_name} ---")
                    target_classes.append(det_name)
                    final_class_id = len(target_classes) - 1
                
                x1, y1, x2, y2 = det["bbox"]
                xc = (x1 + x2) / 2 / w_img
                yc = (y1 + y2) / 2 / h_img
                w = (x2 - x1) / w_img
                h = (y2 - y1) / h_img
                yolo_lines.append(f"{final_class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
            
            if yolo_lines:
                label_p.parent.mkdir(parents=True, exist_ok=True)
                with open(label_p, "w", encoding="utf-8") as f:
                    f.write("\n".join(yolo_lines))
                count += 1
        
        # 如果類別有變動，更新 classes.txt
        if len(target_classes) > initial_class_count:
            class_file = ds_path / "classes.txt"
            print(f"--- [Update Classes] 儲存新的類別清單至 {class_file} ---")
            with open(class_file, "w", encoding="utf-8") as f:
                f.write("\n".join(target_classes))
                
        if progress_callback:
            progress_callback(total, total, count)
        return count
