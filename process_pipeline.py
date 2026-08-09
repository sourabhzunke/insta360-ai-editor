import os
import sys
import cv2
import numpy as np
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

print("🚀 Launching AI Reframing Pipeline (Ultra-Wide Preview Engine)...")

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.ingestion.telemetry import TelemetryExtractor
from src.perception.tracker import IntegratedPerceptionEngine
from src.kinematics.coordinates import SphericalCoordinateTransformer
from src.kinematics.filter import SphericalGimbalFilter
from src.rendering.dewarp import FisheyeDewarpEngine

class ProductionPipelineOrchestrator:
    def __init__(self):
        print("Initializing Dewarped AI Video Reframing Pipeline...")
        self.perception = IntegratedPerceptionEngine()
        self.transformer = SphericalCoordinateTransformer(frame_width=1280, frame_height=720)
        self.gimbal_filter = SphericalGimbalFilter(dt=1/30, process_noise=0.1, measurement_noise=1.5)
        
        # Standard production dewarper (90° FOV)
        self.dewarper = FisheyeDewarpEngine(output_w=1280, output_h=720, fov_deg=90.0)
        # Ultra-wide preview dewarper (130° FOV) for zoomed-out target selection
        self.wide_dewarper = FisheyeDewarpEngine(output_w=1280, output_h=720, fov_deg=130.0)

    def _eval_yaw_angle(self, raw_frame, yaw):
        """Worker function to scan 130° ultra-wide views across cardinal angles in parallel."""
        wide_view = self.wide_dewarper.extract_perspective_viewport(raw_frame, yaw_deg=yaw, pitch_deg=0.0, roll_deg=0.0)
        candidates = self.perception.discover_candidates(wide_view, conf_threshold=0.15)
        return yaw, wide_view, candidates

    def find_first_frame_with_subjects(self, cap_proxy, max_search_frames=300):
        """
        Scans forward across 360° space using ultra-wide viewports to find visible subjects.
        """
        print("🔍 Multi-threaded scanning (Ultra-Wide 130° FOV) for visible subjects...")
        angles = [0.0, 90.0, 180.0, 270.0]
        
        frame_idx = 0
        with ThreadPoolExecutor(max_workers=4) as scan_executor:
            while cap_proxy.isOpened() and frame_idx < max_search_frames:
                ret, raw_frame = cap_proxy.read()
                if not ret:
                    break

                if frame_idx % 15 == 0:
                    futures = [scan_executor.submit(self._eval_yaw_angle, raw_frame, yaw) for yaw in angles]
                    for fut in futures:
                        yaw, wide_view, candidates = fut.result()
                        if candidates:
                            sec = frame_idx / 30.0
                            print(f"✅ Discovered {len(candidates)} subject(s) at Frame {frame_idx} ({sec:.1f}s) | Base Yaw: {yaw}°")
                            return raw_frame, wide_view, frame_idx, yaw, candidates

                frame_idx += 1

        print("⚠️ No subjects auto-detected in search window. Displaying Ultra-Wide Frame 0...")
        cap_proxy.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, raw_frame0 = cap_proxy.read()
        wide_view0 = self.wide_dewarper.extract_perspective_viewport(raw_frame0, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0)
        return raw_frame0, wide_view0, 0, 0.0, []

    def select_target_interactively(self, wide_canvas, candidates, frame_idx, base_yaw):
        """
        Displays 130° zoomed-out frame and converts user selection to exact Yaw/Pitch angles.
        """
        window_name = f"Intel AI Target Lock - Ultra-Wide 130 FOV View (Frame {frame_idx})"
        display_canvas = wide_canvas.copy()

        for t in candidates:
            x1, y1, x2, y2 = t["bbox"]
            cv2.rectangle(display_canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_canvas, f"Subject {t['id']} ({t['confidence']:.2f})", 
                        (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        print("\n=====================================================================")
        print(f" 🎯 TARGET SELECTION INSTRUCTIONS (Zoomed-Out 130° FOV Canvas):")
        print("   1. An interactive window with a wide field-of-view has opened.")
        print("   2. EITHER click directly on a subject OR click & drag a bounding box.")
        print("   3. Press ENTER or SPACE to confirm selection.")
        print("=====================================================================\n")

        os.makedirs("output", exist_ok=True)
        cv2.imwrite("output/discovered_frame_preview.jpg", display_canvas)

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)

        roi = cv2.selectROI(window_name, display_canvas, showCrosshair=True, fromCenter=False)
        cv2.destroyAllWindows()

        rx, ry, rw, rh = roi
        if rx == 0 and ry == 0 and rw == 0 and rh == 0:
            return None, None

        # Convert selection center on 130° wide canvas to spherical Yaw and Pitch offsets
        cx = (rx + rw / 2.0) if rw > 10 else rx
        cy = (ry + rh / 2.0) if rh > 10 else ry

        wide_fov_h = 130.0
        wide_fov_v = wide_fov_h * (720.0 / 1280.0)

        target_yaw = base_yaw + ((cx / 1280.0) - 0.5) * wide_fov_h
        target_pitch = (0.5 - (cy / 720.0)) * wide_fov_v

        print(f"✅ Target Locked! Calibrated Direction -> Yaw: {target_yaw:.1f}°, Pitch: {target_pitch:.1f}°")
        return target_yaw, target_pitch

    def execute_processing_run(self):
        print("\n=====================================================================")
        print(" EXECUTING HIGH-CAPACITY MULTI-THREADED AI REFRAMING PIPELINE")
        print("=====================================================================")

        real_video_path = os.path.abspath("sample_clip.insv")
        lrv_path = real_video_path.replace(".insv", ".lrv")
        output_video_path = os.path.abspath("output/production_reframed_clip.mp4")

        if not os.path.exists(real_video_path):
            print(f"❌ FATAL ERROR: Target video file missing at '{real_video_path}'.")
            return

        proxy_stream_path = lrv_path if os.path.exists(lrv_path) else real_video_path

        # 1. ASYNC TELEMETRY EXTRACTION
        print("\n Step 1: Launching Background Telemetry Extractor...")
        telemetry_parser = TelemetryExtractor()
        imu_container = []
        
        def extract_telemetry_task():
            data = telemetry_parser.parse_exported_telemetry(video_path=proxy_stream_path)
            imu_container.append(data)

        telemetry_thread = threading.Thread(target=extract_telemetry_task, daemon=True)
        telemetry_thread.start()

        # 2. MULTI-THREADED SUBJECT DISCOVERY
        print("\n Step 2: Running Smart Subject Discovery on Ultra-Wide Canvas...")
        cap_proxy = cv2.VideoCapture(proxy_stream_path)
        raw_frame, wide_frame, start_frame_idx, base_yaw, candidates = self.find_first_frame_with_subjects(cap_proxy)

        # 3. INTERACTIVE TARGET SELECTION
        target_yaw, target_pitch = self.select_target_interactively(wide_frame, candidates, start_frame_idx, base_yaw)
        if target_yaw is None:
            print("❌ Pipeline execution aborted: Target selection canceled.")
            cap_proxy.release()
            return

        telemetry_thread.join()
        imu_stream = imu_container[0] if imu_container else []

        # Center standard 90° tracking viewport on target direction
        init_tracking_frame = self.dewarper.extract_perspective_viewport(
            raw_frame, yaw_deg=target_yaw, pitch_deg=target_pitch, roll_deg=0.0
        )
        init_bbox = [580, 270, 700, 450]  # Center target box
        self.perception.initialize_sam_track(init_tracking_frame, init_bbox)

        # 4. PASS 1: DOUBLE-BUFFERED TRAJECTORY TRACKING PASS
        print(f"\n Step 3: Running Trajectory Tracking starting from Frame {start_frame_idx}...")
        trajectory_history = []
        current_bbox = init_bbox
        
        cap_proxy.set(cv2.CAP_PROP_POS_FRAMES, 0)
        track_queue = queue.Queue(maxsize=30)

        def proxy_frame_reader():
            f_idx = 0
            while cap_proxy.isOpened():
                r, frame = cap_proxy.read()
                if not r:
                    break
                track_queue.put((f_idx, frame))
                f_idx += 1
            track_queue.put(None)

        reader_thread = threading.Thread(target=proxy_frame_reader, daemon=True)
        reader_thread.start()

        while True:
            item = track_queue.get()
            if item is None:
                break
            
            frame_idx, raw_frame_item = item
            imu_sample = imu_stream[frame_idx] if frame_idx < len(imu_stream) else {}
            roll_angle = imu_sample.get("gyro_rad_sec", [0.0, 0.0, 0.0])[2]

            if frame_idx < start_frame_idx:
                smooth_angles = {"yaw": target_yaw, "pitch": target_pitch, "roll": -roll_angle}
            else:
                dewarped_frame = self.dewarper.extract_perspective_viewport(
                    raw_frame_item, yaw_deg=target_yaw, pitch_deg=target_pitch, roll_deg=-roll_angle
                )
                tracking_output = self.perception.track_next_frame(dewarped_frame, current_bbox)
                if tracking_output:
                    current_bbox = tracking_output["bbox"]
                    raw_angles = self.transformer.bbox_to_angles(current_bbox)
                    raw_angles["yaw"] += target_yaw
                    raw_angles["pitch"] += target_pitch
                    smooth_angles = self.gimbal_filter.smooth_trajectory(raw_angles["yaw"], raw_angles["pitch"])
                else:
                    smooth_angles = self.gimbal_filter.smooth_trajectory(None, None)

                smooth_angles["roll"] = -roll_angle

            trajectory_history.append(smooth_angles)

            if frame_idx % 60 == 0:
                print(f"   Tracked Frame [{frame_idx:04d}] | Yaw: {smooth_angles['yaw']:.1f}° | Pitch: {smooth_angles['pitch']:.1f}°")

        cap_proxy.release()

        # 5. PASS 2: MULTI-THREADED MASTER EXPORT (.insv)
        print("\n Step 4: Executing Multi-Threaded Master Export Pass (.insv)...")
        cap_master = cv2.VideoCapture(real_video_path)
        total_frames = len(trajectory_history)
        
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(output_video_path, fourcc, 30.0, (1280, 720))

        read_queue = queue.Queue(maxsize=30)
        write_dict = {}

        def master_frame_reader():
            f_idx = 0
            while cap_master.isOpened() and f_idx < total_frames:
                r, master_raw = cap_master.read()
                if not r:
                    break
                read_queue.put((f_idx, master_raw, trajectory_history[f_idx]))
                f_idx += 1
            read_queue.put(None)

        def process_render_frame(item):
            idx, master_raw, angles = item
            high_res_viewport = self.dewarper.extract_perspective_viewport(
                master_raw,
                yaw_deg=angles["yaw"],
                pitch_deg=angles["pitch"],
                roll_deg=angles["roll"]
            )
            return idx, high_res_viewport

        m_reader_thread = threading.Thread(target=master_frame_reader, daemon=True)
        m_reader_thread.start()

        num_workers = min(8, os.cpu_count() or 4)
        print(f"⚡ Spawning {num_workers} parallel worker threads for master rendering...")

        next_write_idx = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = set()
            while True:
                item = read_queue.get()
                if item is None:
                    break
                
                fut = executor.submit(process_render_frame, item)
                futures.add(fut)

                completed = {f for f in futures if f.done()}
                for fut in completed:
                    idx, frame = fut.result()
                    write_dict[idx] = frame
                    futures.remove(fut)

                while next_write_idx in write_dict:
                    out_writer.write(write_dict.pop(next_write_idx))
                    if next_write_idx % 30 == 0 or next_write_idx == total_frames - 1:
                        progress_pct = (next_write_idx / max(1, total_frames)) * 100
                        print(f"   Render Progress: Frame [{next_write_idx:04d}/{total_frames:04d}] ({progress_pct:.1f}%)")
                    next_write_idx += 1

            for fut in futures:
                idx, frame = fut.result()
                write_dict[idx] = frame

            while next_write_idx in write_dict:
                out_writer.write(write_dict.pop(next_write_idx))
                next_write_idx += 1

        cap_master.release()
        out_writer.release()

        print("\n=====================================================================")
        print(f"🎉 SUCCESS: Exported stabilized high-resolution video to -> {output_video_path}")
        print("=====================================================================")

if __name__ == "__main__":
    try:
        orchestrator = ProductionPipelineOrchestrator()
        orchestrator.execute_processing_run()
    except Exception as e:
        print(f"\n❌ Unhandled Execution Exception: {e}")