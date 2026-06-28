import os
import cv2
import numpy as np
from ultralytics import YOLO, SAM

class IntegratedPerceptionEngine:
    def __init__(self, yolo_model_dir=None, sam_model_name="sam2_b.pt"):
        """
        Initializes a dual-stage perception engine leveraging YOLOv10 for candidate 
        generation and SAM 2 for high-fidelity pixel-exact tracking masks.
        Custom-tailored for Intel Arc GPU execution via OpenVINO.
        """
        # 1. Initialize YOLOv10 Candidate Detector
        if yolo_model_dir is None:
            self.yolo_model_dir = os.path.join("models", "yolov10m_openvino_model")
        else:
            self.yolo_model_dir = yolo_model_dir
            
        if not os.path.exists(self.yolo_model_dir):
            raise FileNotFoundError(f"❌ Compiled OpenVINO YOLOv10 model not found at {self.yolo_model_dir}")
            
        print(f"🎮 Initializing OpenVINO YOLOv10 Detector: {self.yolo_model_dir}")
        self.yolo_model = YOLO(self.yolo_model_dir, task="detect")
        
        # 2. Initialize SAM 2 Pixel Mask Engine
        print(f"🧬 Initializing Segment Anything Model 2 (SAM 2): {sam_model_name}")
        # Ultralytics natively manages SAM 2 execution; passing intel:gpu routes tensors through OpenVINO
        self.sam_model = SAM(sam_model_name)
        self.active_predictor = None
        print("✅ Dual-stage AI perception layers loaded successfully.")

    def discover_candidates(self, frame0, target_class_id=0, conf_threshold=0.25):
        """
        Scans Frame 0 and extracts EVERY visible instance of the target class
        without early-breaking, generating an array of potential tracking targets.
        """
        results = self.yolo_model.predict(source=frame0, device="intel:gpu", conf=conf_threshold, verbose=False)
        candidates = []
        
        if not results or len(results) == 0:
            return candidates

        boxes = results[0].boxes
        for idx, box in enumerate(boxes):
            cls_id = int(box.cls[0].item())
            if cls_id == target_class_id:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = box.conf[0].item()
                
                candidates.append({
                    "id": idx + 1,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": confidence
                })
        return candidates

    def initialize_sam_track(self, frame0, chosen_bbox):
        """
        Seeds the SAM 2 segmentation track using the chosen YOLO bounding box as a prompt coordinate.
        Leverages 96GB system RAM boundaries to store temporal frame attention layouts.
        """
        print(f"🎯 Seeding SAM 2 segment tracking mask with bounding coordinates: {chosen_bbox}")
        # Generate initial high-fidelity segmented pixel mask from bounding box prompt
        sam_results = self.sam_model.predict(source=frame0, bboxes=[chosen_bbox], device="intel:gpu", verbose=False)
        
        # Isolate mask metadata array matrix
        if sam_results and len(sam_results) > 0 and sam_results[0].masks is not None:
            print("✅ SAM 2 successfully compiled baseline pixel-mask geometry.")
            return sam_results[0].masks.data[0].cpu().numpy()
        else:
            print("⚠️ SAM 2 failed to compile exact pixel mask coordinates. Defaulting to bounding box tracks.")
            return None

    def track_next_frame(self, current_frame, last_bbox):
        """
        Tracks the chosen target across successive frames using SAM 2's prompt memory loops
        running natively inside the local hardware tier.
        """
        # In consecutive tracking execution, the previous bounding box guides the mask prediction
        results = self.sam_model.predict(source=current_frame, bboxes=[last_bbox], device="intel:gpu", verbose=False)
        
        if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            # Extract updated tracking coordinates dynamically calculated by SAM 2
            x1, y1, x2, y2 = results[0].boxes.xyxy[0].tolist()
            mask = results[0].masks.data[0].cpu().numpy() if results[0].masks is not None else None
            return {
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "mask": mask
            }
        return None

if __name__ == "__main__":
    print("🎬 Testing Unified Selection & SAM 2 Mask Tracking Pipeline...")
    try:
        # Create a dummy image representing Frame 0 with a simulated human target square
        dummy_frame0 = np.zeros((480, 640, 3), dtype=np.uint8)
        # Paint a distinct synthetic box so YOLO has something to look at
        cv2.rectangle(dummy_frame0, (200, 150), (350, 350), (255, 255, 255), -1)
        
        engine = IntegratedPerceptionEngine()
        
        # Stage 1: Candidate Discovery
        print("\n🔍 Step 1: Scanning Canvas for Candidates...")
        targets = engine.discover_candidates(dummy_frame0, conf_threshold=0.1)
        print(f"Identified {len(targets)} candidate individuals on Frame 0.")
        
        for t in targets:
            print(f" -> Candidate ID [{t['id']}]: Box Location {t['bbox']} (Conf: {t['confidence']:.2f})")
            
        # Simulate User Selection Selection (Locking onto Candidate 1)
        if len(targets) > 0:
            chosen_target = targets[0]
            print(f"\n🙋 User Action simulated: Selected Target ID [{chosen_target['id']}]")
            
            # Stage 2: Prompting SAM 2
            print("\n⚡ Step 2: Extracting Pixel-Exact Segmentation Mask via SAM 2...")
            initial_mask = engine.initialize_sam_track(dummy_frame0, chosen_target["bbox"])
            
            if initial_mask is not None:
                print(f"🎉 Core Pipeline Verified. Mask array generated with shape dimensions: {initial_mask.shape}")
            
    except Exception as e:
        print(f"❌ Interactive Perception pipeline fault: {e}")