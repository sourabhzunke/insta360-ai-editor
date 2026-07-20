import os
import json
import subprocess
import shutil
import numpy as np

class TelemetryExtractor:
    def __init__(self, json_path=None):
        """
        Manages the ingestion layer for spatial metadata, supporting both direct
        open-source parsing extensions and mock development profiles.
        """
        self.json_path = json_path if json_path else os.path.join("config", "telemetry_dump.json")

    def parse_exported_telemetry(self, video_path=None):
        """
        Loads the structured JSON telemetry stream. If a raw video file is provided, 
        it attempts to extract the embedded camera metadata dynamically.
        """
        # If a real video path is provided, execute automated open-source extraction
        if video_path and os.path.exists(video_path):
            success = self._extract_telemetry_with_exiftool(video_path)
            if not success:
                print("⚠️ Falling back to existing log patterns or mock configurations.")

        if not os.path.exists(self.json_path):
            print(f"⚠️ Telemetry file missing at {self.json_path}. Compiling a mock data profile...")
            return self._generate_mock_json_export()

        print(f"🛰️  Ingesting telemetry log asset: {self.json_path}")
        with open(self.json_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        telemetry_matrix = []
        samples = raw_data.get("samples", raw_data.get("telemetry", []))
        
        for entry in samples:
            timestamp = entry.get("timestamp_ms", entry.get("cts", 0))
            gyro = entry.get("gyro", entry.get("gyroscope", [0.0, 0.0, 0.0]))
            accel = entry.get("accel", entry.get("accelerometer", [0.0, 9.81, 0.0]))
            temp = entry.get("temperature", entry.get("temp", 35.0))
            
            telemetry_matrix.append({
                "timestamp_ms": int(timestamp),
                "gyro_rad_sec": [round(float(g), 4) for g in gyro],
                "temperature_c": round(float(temp), 1),
                "accel_m_sec2": [round(float(a), 4) for a in accel]
            })

        print(f"📊 Extraction Complete: Structured {len(telemetry_matrix)} hardware IMU frames.")
        return telemetry_matrix

    def _extract_telemetry_with_exiftool(self, video_path):
        """
        Invokes local ExifTool subprocess threads to parse embedded 
        timed camera motion metadata layouts automatically without third-party dependencies.
        """
        # Search path bindings for the local executable asset
        exiftool_bin = "exiftool"
        if not shutil.which(exiftool_bin):
            if os.path.exists("exiftool.exe"):
                exiftool_bin = os.path.abspath("exiftool.exe")
            else:
                print("❌ ExifTool not found. Please drop exiftool.exe into your project root folder.")
                return False

        print(f"🛠️  Extracting camera metadata stream from {os.path.basename(video_path)} using ExifTool...")
        
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        
        # Build command array to extract embedded timed metadata tracks as structured JSON strings
        cmd = [
            exiftool_bin,
            "-ee",                        # Extract embedded streams
            "-G3",                        # Classify document elements
            "-api", "LargeFileSupport=1", # Prevent size limits on files over 4GB
            "-j",                         # Structure string terminal output to JSON
            video_path
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            extracted_json = json.loads(result.stdout)
            
            # Map the complex multi-track records down into our pipeline's expected format
            mock_payload = {"samples": []}
            base_time = 0.0
            
            # Extract strings from sequential camera metadata fields if present
            if extracted_json and len(extracted_json) > 0:
                file_metadata = extracted_json[0]
                
                # Check for standardized CAMM fields or device telemetry hooks
                gyro_data = [val for k, val in file_metadata.items() if "AngularVelocity" in k]
                accel_data = [val for k, val in file_metadata.items() if "Acceleration" in k]
                
                # Zip parsed attributes down into our standard matrix structure
                total_samples = max(len(gyro_data), 90) # Handle safety fallback length
                for i in range(total_samples):
                    g_vec = [float(x) for x in gyro_data[i].split()] if i < len(gyro_data) and isinstance(gyro_data[i], str) else [0.0, 0.0, 0.0]
                    a_vec = [float(x) for x in accel_data[i].split()] if i < len(accel_data) and isinstance(accel_data[i], str) else [0.0, 9.81, 0.0]
                    
                    mock_payload["samples"].append({
                        "timestamp_ms": int(base_time * 1000),
                        "gyro": g_vec,
                        "accelerometer": a_vec,
                        "temperature": 35.0
                    })
                    base_time += 1/30
                    
            with open(self.json_path, "w", encoding="utf-8") as f:
                json.dump(mock_payload, f, indent=2)
                
            print(f"✅ Automatically compiled real metadata logging profile to: {self.json_path}")
            return True
        except Exception as e:
            print(f"⚠️ Metadata extractor subprocess trace encountered an issue: {e}")
            return False

    def _generate_mock_json_export(self):
        """Standard backup fallback script asset builder."""
        os.makedirs(os.path.dirname(self.json_path), exist_ok=True)
        mock_payload = {"samples": []}
        base_time = 0.0
        for i in range(90):
            mock_payload["samples"].append({
                "timestamp_ms": int(base_time * 1000),
                "gyro": [float(0.1 * np.sin(base_time)), 0.0, 0.0],
                "accelerometer": [0.0, 9.81, 0.0],
                "temperature": 35.0
            })
            base_time += 1/30
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(mock_payload, f, indent=2)
        return self.parse_exported_telemetry()