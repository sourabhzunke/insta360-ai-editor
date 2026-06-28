import os
import re
import numpy as np
import cv2
import sys

class VideoPairParser:
    def __init__(self, workspace_dir):
        """
        Initializes the Ingestion Parser to look for matching low-res proxies 
        and high-res master video pairs.
        """
        self.workspace_dir = workspace_dir
        self.supported_extensions = ('.mp4', '.insv', '.lrv')

    def scan_workspace(self):
        """
        Scans the directory and maps matching LRV and Master video profiles.
        """
        all_files = [f for f in os.listdir(self.workspace_dir) if f.lower().endswith(self.supported_extensions)]
        lrv_files = {}
        master_files = {}

        # Categorize files based on core naming tokens
        for file in all_files:
            # Matches standard Insta360 timestamp tokens (e.g., VID_20260516_143000)
            timestamp_match = re.search(r'(VID|LRV)_(\d{8}_\d{6})', file)
            if not timestamp_match:
                continue
                
            prefix, timestamp = timestamp_match.groups()
            full_path = os.path.join(self.workspace_dir, file)

            if prefix == "LRV" or "lrv" in file.lower():
                lrv_files[timestamp] = full_path
            else:
                master_files[timestamp] = full_path

        # Synchronize pairs
        video_pairs = {}
        for timestamp, lrv_path in lrv_files.items():
            if timestamp in master_files:
                video_pairs[timestamp] = {
                    "proxy": lrv_path,
                    "master": master_files[timestamp],
                    "metadata": self._get_video_metadata(lrv_path, master_files[timestamp])
                }
        
        return video_pairs

    def _get_video_metadata(self, proxy_path, master_path):
        """
        Extracts structural frame rates and frame lengths to ensure matching indexes.
        """
        cap_proxy = cv2.VideoCapture(proxy_path)
        cap_master = cv2.VideoCapture(master_path)

        metadata = {
            "proxy_fps": cap_proxy.get(cv2.CAP_PROP_FPS),
            "proxy_frames": int(cap_proxy.get(cv2.CAP_PROP_FRAME_COUNT)),
            "master_fps": cap_master.get(cv2.CAP_PROP_FPS),
            "master_frames": int(cap_master.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

        cap_proxy.release()
        cap_master.release()
        
        # Calculate scaling multiplier factor for tracking transformations
        if metadata["proxy_frames"] > 0:
            metadata["frame_scale_factor"] = metadata["master_frames"] / metadata["proxy_frames"]
        else:
            metadata["frame_scale_factor"] = 1.0

        return metadata
		
class FrameBufferIngestor:
    def __init__(self, video_path, target_width=640, target_height=480, max_cached_chunks=500):
        """
        Utilizes high system RAM capacity to cache decoded video frames in memory arrays,
        preventing continuous disk read latency during AI inference loops.
        """
        self.video_path = video_path
        self.target_width = target_width
        self.target_height = target_height
        self.max_cached_chunks = max_cached_chunks
        self.buffer = []
        
    def load_video_to_ram(self):
        """
        Reads the video proxy target file and populates the high-speed RAM cache matrix.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Error: Unable to open video track at {self.video_path}")
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"🧠 Allocating RAM cache for {total_frames} frames...")
        
        frame_idx = 0
        while True:
            ret, frame = cap.get_cmd() if hasattr(cap, 'get_cmd') else cap.read()
            if not ret:
                break
                
            # Resize frame down to optimized dimensions for fast AI ingestion
            resized_frame = cv2.resize(frame, (self.target_width, self.target_height))
            self.buffer.append(resized_frame)
            
            frame_idx += 1
            if frame_idx >= self.max_cached_chunks:
                print(f"⚠️ Cache ceiling hit at {self.max_cached_chunks} frames.")
                break

        cap.release()
        
        # Convert internal list to a contiguous block of memory for ultra-fast GPU indexing
        self.buffer = np.array(self.buffer, dtype=np.uint8)
        ram_usage_mb = self.buffer.nbytes / (1024 * 1024)
        print(f"✅ Cached frame matrix successfully into RAM: {ram_usage_mb:.2f} MB utilized.")
        return True

    def get_frame(self, frame_index):
        """
        Fetches a frame instantly from memory instead of hitting the storage drive.
        """
        if frame_index >= len(self.buffer):
            return None
        return self.buffer[frame_index]

if __name__ == "__main__":
    # Test Block Execution
    print("🎬 Testing Ingestion Parser Module...")
    # Change this path to a local folder containing your test clips when ready
    test_parser = VideoPairParser(workspace_dir=".")
    found_pairs = test_parser.scan_workspace()
    print(f"✅ Scanning complete. Found {len(found_pairs)} matching proxy/master sets.")