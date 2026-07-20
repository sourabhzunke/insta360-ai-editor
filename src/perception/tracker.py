import os
import numpy as np
from ultralytics import YOLO, FastSAM

class IntegratedPerceptionEngine:
    def __init__(self):
        """
        Initializes dual-stage AI perception layers, routing execution 
        graphs directly onto the local Intel Arc 140T GPU profile using 
        the OpenVINO runtime framework.
        """
        # Define structural framework weights paths
        yolo_path = os.path.join("models", "yolov10m_openvino_model")
        fastsam_path = "FastSAM-s_openvino_model"
        
        # Verify hardware graph folders exist before spinning up model tensors
        if not os.path.exists(yolo_path):
            print(f"⚠️ Warning: YOLOv10 OpenVINO asset directory not found at {yolo_path}")
        if not os.path.exists(fastsam_path):
            print(f"⚠️ Warning: FastSAM OpenVINO asset directory not found at {fastsam_path}")

        print("🎮 Initializing OpenVINO YOLOv10 Detector on Intel Arc GPU...")
        self.yolo_model = YOLO(yolo_path)
        
        print("🧬 Initializing OpenVINO FastSAM on Intel Arc GPU...")
        self.sam_model = FastSAM(fastsam_path)
        
        print("✅ Dual-stage OpenVINO hardware perception layers successfully initialized on GPU layout.")

    def discover_candidates(self, frame, conf_threshold=0.1):
        """
        Leverages OpenVINO-accelerated YOLOv10 to scan an initial frame canvas 
        and discover trackable human action subject targets.
        """
        candidates = []
        
        try:
            # Using 'intel:gpu' bypasses CUDA guardrails and triggers OpenVINO's Intel Arc path
            results = self.yolo_model.predict(
                source=frame,
                device="intel:gpu",
                conf=conf_threshold,
                verbose=False
            )
            
            if results and len(results) > 0:
                boxes = results[0].boxes
                for idx, box in enumerate(boxes):
                    # Filter for human/person class tracking signatures (Class ID 0 in COCO)
                    class_id = int(box.cls[0].cpu().numpy())
                    if class_id == 0:
                        xyxy = box.xyxy[0].cpu().numpy().tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        
                        candidates.append({
                            "id": idx + 1,
                            "confidence": conf,
                            "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                        })
        except Exception as e:
            print(f"⚠️ Error occurred during automated YOLOv10 candidate discovery scan: {e}")
            
        return candidates

    def initialize_sam_track(self, frame0, chosen_bbox):
        """
        Seeds the FastSAM prompt-guided selection layers on the Intel Arc GPU 
        using OpenVINO runtime graph execution arrays.
        """
        print(f"🎯 Seeding FastSAM tracking mask with bounding coordinates: {chosen_bbox}")
        
        # Route processing straight into your 77 TOPS hardware silicon
        results = self.sam_model.predict(
            source=frame0, 
            bboxes=[chosen_bbox], 
            device="intel:gpu", 
            imgsz=320,
            verbose=False
        )
        return results

    def track_next_frame(self, frame, current_bbox):
        """
        Streams sequential frame slices directly across parallel GPU cores, 
        updating localized prompt mask tracking boundaries frame-by-frame.
        """
        try:
            # Accelerated execution loop mapping straight to Intel Arc hardware
            results = self.sam_model.predict(
                source=frame, 
                bboxes=[current_bbox], 
                device="intel:gpu", 
                imgsz=320,
                verbose=False
            )
            
            # Extract prompt-guided region coordinates out of the results object matrix
            if results and len(results) > 0:
                boxes = results[0].boxes.xyxy
                if len(boxes) > 0:
                    xyxy = boxes[0].cpu().numpy().tolist()
                    return {
                        "bbox": [int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])]
                    }
        except Exception as e:
            print(f"⚠️ Error encountered during sequential tracking loop frame execution: {e}")
            
        return None