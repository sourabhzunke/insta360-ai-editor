import os
import urllib.request
from ultralytics import YOLO

def compile_model_for_intel():
    """
    Checks for raw weights, downloads them if missing, and compiles the framework
    into an OpenVINO Intermediate Representation (IR) for Intel GPU/NPU acceleration.
    """
    model_dir = "models"
    model_path = os.path.join(model_dir, "yolov10m.pt")
    
    # 1. Automatic Download Safeguard
    if not os.path.exists(model_path):
        os.makedirs(model_dir, exist_ok=True)
        model_url = "https://github.com/THU-MIG/YOLOv10/releases/download/v1.1/yolov10m.pt"
        print(f"📥 Raw weights missing. Downloading YOLOv10m (~50MB) to {model_path}...")
        try:
            urllib.request.urlretrieve(model_url, model_path)
            print("✅ Download complete!")
        except Exception as e:
            print(f"❌ Download failed: {e}")
            return

    # 2. Load the PyTorch Model
    print("🧠 Loading PyTorch baseline framework...")
    model = YOLO(model_path)

    # 3. Export to OpenVINO IR
    print("⚡ Compiling model framework to OpenVINO IR format...")
    # half=True forces FP16 weight quantization, optimized for local Intel Arc processing
    exported_path = model.export(format="openvino", half=True, dynamic=False)
    
    print(f"🎉 Success! OpenVINO IR assets compiled at: {exported_path}")

if __name__ == "__main__":
    print("🚀 STARTING AUTOMATED OPENVINO OPTIMIZATION PIPELINE...")
    compile_model_for_intel()