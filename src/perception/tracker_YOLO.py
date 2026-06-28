import os
import cv2
from ultralytics import YOLO

class OpenVINOTracker:
    def __init__(self, model_dir=None):
        """
        Initializes the tracking engine using the compiled OpenVINO IR directory,
        forcing compute paths straight into the hardware acceleration assets.
        """
        if model_dir is None:
            self.model_dir = os.path.join("models", "yolov10m_openvino_model")
        else:
            self.model_dir = model_dir
            
        if not os.path.exists(self.model_dir):
            raise FileNotFoundError(f"❌ Compiled OpenVINO model folder not found at {self.model_dir}")
            
        print(f"🎮 Initializing OpenVINO Inference Engine Target: {self.model_dir}")
        # Load the compiled OpenVINO folder directory structure directly
        self.model = YOLO(self.model_dir, task="detect")
        print("✅ Model loaded successfully into core namespace.")

    def process_frame(self, frame, target_class_id=0, conf_threshold=0.25):
        """
        Runs high-speed inference on a single frame array using the Intel Arc GPU hardware tier.
        Default target_class_id=0 tracks human targets (skiers, rafters, hikers).
        """
        # Run local hardware inference via OpenVINO backend wrapper
        # verbose=False suppresses logging dumps to keep frame loops snappy
        results = self.model.predict(source=frame, device="intel:gpu", conf=conf_threshold, verbose=False)
        
        detections = []
        if not results or len(results) == 0:
            return detections

        # Extract predicted bounding coordinates from frame 0 evaluation matrix
        boxes = results[0].boxes
        for box in boxes:
            cls_id = int(box.cls[0].item())
            
            # Filter results to strictly isolate your tracking target class
            if cls_id == target_class_id:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                
                detections.append({
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": confidence,
                    "class_id": cls_id
                })
                
                # We break after finding the primary target to lock tracking consistency
                break
                
        return detections

if __name__ == "__main__":
    print("🎯 Testing OpenVINO Target Tracking Engine Class...")
    try:
        tracker = OpenVINOTracker()
        # Initialize a blank synthetic canvas frame array to test pipe connectivity
        import numpy as np
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        print("⚡ Executing dry-run hardware inference loop on Arc GPU...")
        test_detections = tracker.process_frame(blank_frame)
        print(f"🎉 Pipeline Functional. Returned {len(test_detections)} visual matches on sample canvas.")
    except Exception as e:
        print(f"❌ Operational tracking failure: {e}")