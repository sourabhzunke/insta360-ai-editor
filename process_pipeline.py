import sys
import os
import cv2
import numpy as np

# Ensure local packages are resolvable within the execution path environment
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our complete verified architecture stack
from src.ingestion.telemetry import TelemetryExtractor
from src.perception.tracker import IntegratedPerceptionEngine
from src.kinematics.coordinates import SphericalCoordinateTransformer
from src.kinematics.filter import SphericalGimbalFilter
from src.rendering.viewport import ViewportRenderEngine

class ProductionPipelineOrchestrator:
    def __init__(self):
        """
        Orchestrates end-to-end streaming loops, routing frame tensors across 
        the local Intel hardware workspace profile without accumulating memory.
        """
        print("Initializing Unified Streaming Pipeline Orchestration Environment...")
        self.perception = IntegratedPerceptionEngine()
        self.transformer = SphericalCoordinateTransformer(frame_width=640, frame_height=480)
        self.gimbal_filter = SphericalGimbalFilter(dt=1/30, process_noise=0.3, measurement_noise=1.5)
        self.renderer = ViewportRenderEngine(viewport_w=320, viewport_h=180)
        
        self.clicked_bbox = None

    def _mouse_click_handler(self, event, x, y, flags, param):
        """
        OpenCV interactive mouse callback thread. Maps exact pixel clicks 
        to localized target candidate bounding boxes.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            candidates = param["candidates"]
            for target in candidates:
                x1, y1, x2, y2 = target["bbox"]
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.clicked_bbox = target["bbox"]
                    print(f"\n Target Locked Interactively! Candidate ID [{target['id']}] -> Box: {self.clicked_bbox}")
                    break

    def select_target_interactively(self, frame0, candidates):
        """
        Spawns an interactive graphical canvas UI window displaying Frame 0 overlays.
        """
        if not candidates:
            print("No tracking targets discovered on Frame 0 canvas layout.")
            return None

        window_name = "Intel AI Editor - Select Tracking Target"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        
        display_canvas = frame0.copy()
        for t in candidates:
            x1, y1, x2, y2 = t["bbox"]
            cv2.rectangle(display_canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_canvas, f"ID: {t['id']} (Conf: {t['confidence']:.2f})", 
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.setMouseCallback(window_name, self._mouse_click_handler, param={"candidates": candidates})

        print("\n Interactive UI Window Spawned Successfully.")
        print("--> ACTION REQUIRED: Click directly INSIDE the green rectangle to lock target.")
        
        while self.clicked_bbox is None:
            cv2.imshow(window_name, display_canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Manual selection sequence aborted by user.")
                break

        cv2.destroyWindow(window_name)
        return self.clicked_bbox

    def execute_processing_run(self):
        """
        Executes a complete production streaming run across all architecture sectors.
        Streams frames frame-by-frame to support massive multi-gigabyte files safely.
        """
        print("\n=====================================================================")
        print(" EXECUTING HIGH-CAPACITY STREAMING AI REFRAMING PIPELINE")
        print("=====================================================================")

        # Target your raw video source path file here
        real_video_path = "C:/Users/SZ/Downloads/video_maker_project/sample_clip.insv"
        output_video_path = "output/production_reframed_clip.mp4"
        
        if not os.path.exists(real_video_path):
            print(f"❌ Target video file missing at {real_video_path}. Please place your video file there.")
            return

        # Check for a matching low-resolution companion asset (.lrv) to speed up telemetry processing
        telemetry_source_path = real_video_path
        lrv_candidate = real_video_path.replace(".insv", ".lrv")
        if os.path.exists(lrv_candidate):
            print(f"   [INFO] Found matching low-resolution companion asset: {lrv_candidate}")
            print("   -> Routing telemetry parser to the .lrv file to accelerate metadata extraction.")
            telemetry_source_path = lrv_candidate

        # 1. Extract Telemetry Logs via ExifTool
        print("\n Step 1: Running Automated Subprocess Telemetry Extraction...")
        telemetry_parser = TelemetryExtractor()
        imu_stream = telemetry_parser.parse_exported_telemetry(video_path=telemetry_source_path)

        # 2. Initialize Video Capture Stream Pointer
        print("\n Step 2: Instantiating I/O Hardware Video Streaming Pointers...")
        cap = cv2.VideoCapture(real_video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open video file stream pointer at: {real_video_path}")
            return

        # 3. Read Frame 0 Separately for Interactive Target Initialization Hook
        ret, raw_frame0 = cap.read()
        if not ret:
            print("❌ Failed to parse initial Frame 0 frame tensor slice from the container source.")
            cap.release()
            return

        frame0 = cv2.resize(raw_frame0, (640, 480), interpolation=cv2.INTER_AREA)

        # 4. Candidate Discovery via OpenVINO YOLOv10 on GPU
        print("\n Step 3: Running Arc GPU Candidate Scan on Frame 0...")
        candidates = self.perception.discover_candidates(frame0, conf_threshold=0.1)
        
        if not candidates:
            # Fallback tracking bounding box if running initial camera alignment tests
            print("   [INFO] YOLOv10 returned 0 automated matches. Injecting baseline selection layer...")
            candidates = [{
                "id": 1,
                "confidence": 0.99,
                "bbox": [200, 150, 440, 390]
            }]

        # 5. Interactive Graphical Target Handshake
        target_bbox = self.select_target_interactively(frame0, candidates)
        if target_bbox is None:
            print("Pipeline execution aborted: No valid interactive selection locked.")
            cap.release()
            return

        # 6. Seed SAM 2 Mask Track Memory Profile
        print("\n Step 4: Seeding SAM 2 Structural Tracking Layer on Target Coordinate...")
        initial_mask = self.perception.initialize_sam_track(frame0, target_bbox)

        # 7. Open Widescreen Output Video File Encoder Pointer using clean MP4V bindings
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        out_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (self.renderer.vw, self.renderer.vh))
        
        # Process and render Frame 0 initial conditions into the export track file
        raw_angles_0 = self.transformer.bbox_to_angles(target_bbox)
        smooth_angles_0 = self.gimbal_filter.smooth_trajectory(raw_angles_0["yaw"], raw_angles_0["pitch"])
        x1, y1, x2, y2 = self.renderer.calculate_crop_window(smooth_angles_0["yaw"], smooth_angles_0["pitch"], 640, 480)
        out_writer.write(frame0[y1:y2, x1:x2])

        # 8. Commencing Real-Time Sequential Single-Pass Frame-by-Frame Processing Loop
        print("\n Step 5: Executing Continuous Memory-Safe Kinematic Processing Loop...")
        current_tracking_bbox = target_bbox
        frame_idx = 1

        while cap.isOpened():
            ret, raw_frame = cap.read()
            if not ret:
                break # Reached end of video file stream naturally

            # Downsample frame tensor immediately to keep execution memory footprint flat
            pipeline_ready_frame = cv2.resize(raw_frame, (640, 480), interpolation=cv2.INTER_AREA)

            # Pass the individual frame through SAM 2 on the optimized CPU track (imgsz=320)
            tracking_output = self.perception.track_next_frame(pipeline_ready_frame, current_tracking_bbox)

            if tracking_output:
                current_tracking_bbox = tracking_output["bbox"]
                raw_angles = self.transformer.bbox_to_angles(current_tracking_bbox)
                smooth_angles = self.gimbal_filter.smooth_trajectory(raw_angles["yaw"], raw_angles["pitch"])
            else:
                # Occlusion fallback tracking logic if target signature profile drops temporarily
                smooth_angles = self.gimbal_filter.smooth_trajectory(None, None)

            # Compute boundary parameters and slice the virtual tracking viewport out of the canvas
            cx1, cy1, cx2, cy2 = self.renderer.calculate_crop_window(
                smooth_angles["yaw"], smooth_angles["pitch"], 
                frame_w=640, frame_h=480
            )
            viewport_frame = pipeline_ready_frame[cy1:cy2, cx1:cx2]

            # Write frame directly to disk storage file path
            out_writer.write(viewport_frame)

            if frame_idx % 30 == 0:
                print(f"   Streaming Progress: Processed Frame [{frame_idx:04d}] | Locked Target Coordinates")

            frame_idx += 1

        # Close all hardware file streams cleanly
        cap.release()
        out_writer.release()

        print("\n=====================================================================")
        print(f"🎉 SUCCESS: Reframed tracking video compiled across {frame_idx} total frames.")
        print(f"🎯 Full-length production asset exported safely to -> {output_video_path}")
        print("=====================================================================")

if __name__ == "__main__":
    orchestrator = ProductionPipelineOrchestrator()
    orchestrator.execute_processing_run()