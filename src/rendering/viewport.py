import os
import cv2
import numpy as np

class ViewportRenderEngine:
    def __init__(self, viewport_w=320, viewport_h=180):
        """
        Initializes the rendering engine to map smoothed angular trajectories
        back into dynamic pixel-level crop boundaries (viewports).
        """
        self.vw = viewport_w
        self.vh = viewport_h

    def calculate_crop_window(self, smooth_yaw, smooth_pitch, frame_w=640, frame_h=480):
        """
        Translates spherical degrees back into pixel coordinates, centering
        the extraction viewport around the smoothed trajectory target.
        """
        # Reverse the linear mapping from SphericalCoordinateTransformer
        # norm_x = yaw / 360.0 -> x_center = (norm_x + 0.5) * frame_w
        norm_x = smooth_yaw / 360.0
        norm_y = 0.5 - (smooth_pitch / 180.0)

        x_center = int((norm_x + 0.5) * frame_w)
        y_center = int(norm_y * frame_h)

        # Calculate bounding boundaries for top-left crop point
        x1 = x_center - (self.vw // 2)
        y1 = y_center - (self.vh // 2)

        # Enforce frame boundaries to keep crop window strictly within source bounds
        x1 = max(0, min(x1, frame_w - self.vw))
        y1 = max(0, min(y1, frame_h - self.vh))

        x2 = x1 + self.vw
        y2 = y1 + self.vh

        return x1, y1, x2, y2

    def render_stabilized_sequence(self, frame_buffer, trajectory_history, output_path="output_tracked.mp4", fps=30):
        """
        Iterates over input frame matrices, applies dynamic viewport cropping 
        frame-by-frame, and writes out an accelerated compressed video file.
        """
        if not frame_buffer or len(frame_buffer) != len(trajectory_history):
            raise ValueError("❌ Frame buffer length and trajectory history length must be identical.")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use AVC1/H.264 codec mapping for high performance MP4 encoding compliance
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (self.vw, self.vh))

        if not writer.isOpened():
            print("⚠️ AVC1 codec unavailable. Falling back to default MP4V container encoder...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (self.vw, self.vh))

        print(f"🎬 Initializing Video Writer: Exporting to {output_path} ({self.vw}x{self.vh} @ {fps} FPS)...")

        for idx, frame in enumerate(frame_buffer):
            angles = trajectory_history[idx]
            
            # Compute dynamic crop box coordinates
            x1, y1, x2, y2 = self.calculate_crop_window(
                angles["yaw"], angles["pitch"], 
                frame_w=frame.shape[1], frame_h=frame.shape[0]
            )
            
            # Slice the virtual viewport matrix out of the high-res canvas
            viewport_frame = frame[y1:y2, x1:x2]
            
            # Commit frame to video file stream
            writer.write(viewport_frame)

        writer.release()
        print(f"🎉 Rendering Complete! Stabilized tracking output file saved to -> {output_path}")
        return output_path

if __name__ == "__main__":
    print("🎬 STARTING VIEWPORT RENDER ENGINE SECTOR VERIFICATION...")
    
    # Instantiate 16:9 widescreen layout renderer configuration
    renderer = ViewportRenderEngine(viewport_w=320, viewport_h=180)
    
    # Build synthetic test datasets: 60 frames of a 640x480 canvas array
    print("📦 Creating mock frame buffers and trajectory tracks...")
    mock_frames = []
    mock_trajectory = []
    
    for idx in range(60):
        # Generate a distinct structural graphic layout for verification
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        # Paint frame identifier bars to visually verify panning operations
        cv2.line(canvas, (0, idx * 4), (640, 480 - (idx * 4)), (0, 0, 255), 3)
        mock_frames.append(canvas)
        
        # Simulate an active Kalman panning trajectory sweep
        simulated_yaw = -45.0 + (idx * 1.5)
        simulated_pitch = 10.0 - (idx * 0.2)
        mock_trajectory.append({"yaw": simulated_yaw, "pitch": simulated_pitch})

    test_output = "output/render_test.mp4"
    try:
        exported_file = renderer.render_stabilized_sequence(mock_frames, mock_trajectory, output_path=test_output)
        
        if os.path.exists(exported_file) and os.path.getsize(exported_file) > 0:
            print(f"✅ Render Module Passed Validation! File Size: {os.path.getsize(exported_file) / 1024:.2f} KB")
            
        # Clean up transient validation output directory files
        if os.path.exists(exported_file): 
            os.remove(exported_file)
            os.rmdir("output")
    except Exception as e:
        print(f"❌ Render Engine processing validation failure: {e}")