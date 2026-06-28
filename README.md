# insta360-ai-editor

Automated AI-based video reframing and object tracking pipeline for Insta360 footage optimized for Intel OpenVINO and QuickSync.

## 🚀 System Architecture Framework
- **Ingestion Module:** Dual-track synchronization mapping low-res `.lrv` proxy metrics to high-res `.insv` stitched source binaries.
- **Perception Engine:** Optimized via **Intel OpenVINO** running target detection (YOLOv10 / RT-DETR) and pixel matching (SAM 2) paths natively on Intel Arc graphics assets.
- **Kinematic Stabilizer:** Discrete Kalman filtering loops resolving target blackouts and Savitzky-Golay path-smoothing logic.
- **Render Engine:** Direct hardware-accelerated **FFmpeg Intel QuickSync Video (QSV)** compilation exporting custom 16:9 and 9:16 video payloads.

## 🛠️ Local Development Installation
1. Install OpenVINO core dependencies and FFmpeg locally.
2. Initialize environment:
   pip install -r config/requirements.txt
