import os
import cv2
import numpy as np
from ultralytics import YOLO, FastSAM

def _create_opencv_tracker():
    if hasattr(cv2, 'TrackerCSRT_create'):
        return cv2.TrackerCSRT_create()
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
        return cv2.legacy.TrackerCSRT_create()
    if hasattr(cv2, 'TrackerMIL_create'):
        return cv2.TrackerMIL_create()
    if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMIL_create'):
        return cv2.legacy.TrackerMIL_create()
    raise RuntimeError("❌ No supported OpenCV C++ tracker module found.")

class IntegratedPerceptionEngine:
    def __init__(self):
        yolo_path = os.path.join("models", "yolov10m_openvino_model")
        fastsam_path = "FastSAM-s_openvino_model"
        
        print("🎮 Initializing OpenVINO YOLOv10 Detector...")
        self.yolo_model = YOLO(yolo_path, task="detect")
        
        print("🧬 Initializing OpenVINO FastSAM...")
        self.sam_model = FastSAM(fastsam_path)
        
        self.tracker = None
        print("✅ Hybrid Perception Engine Ready.")

    def discover_candidates(self, frame, conf_threshold=0.1):
        """
        Scans a frame canvas for human tracking candidates on the Intel Arc GPU.
        """
        candidates = []
        try:
            results = self.yolo_model.predict(
                source=frame,
                device="intel:gpu",
                conf=conf_threshold,
                verbose=False
            )
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                for idx, box in enumerate(boxes):
                    class_id = int(box.cls[0].cpu().numpy())
                    if class_id == 0:  # Class ID 0 = Person
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        candidates.append({
                            "id": idx + 1,
                            "confidence": conf,
                            "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                        })
        except Exception as e:
            print(f"⚠️ Error during YOLOv10 candidate scan: {e}")
            
        return candidates

    def initialize_sam_track(self, frame0, chosen_bbox):
        print(f"🎯 Locking target coordinates: {chosen_bbox}")
        x1, y1, x2, y2 = chosen_bbox
        w = x2 - x1
        h = y2 - y1
        
        self.tracker = _create_opencv_tracker()
        self.tracker.init(frame0, (x1, y1, w, h))
        return chosen_bbox

    def track_next_frame(self, frame, current_bbox):
        if self.tracker is None:
            return None

        success, box = self.tracker.update(frame)
        if success:
            x, y, w, h = [int(v) for v in box]
            return {"bbox": [x, y, x + w, y + h]}
        
        return None