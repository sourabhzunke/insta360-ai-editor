import os
import sys
import subprocess
import requests

# Repository Configuration
REPO_NAME = "insta360-ai-editor"
REPO_DESCRIPTION = "Automated AI-based video reframing and object tracking pipeline for Insta360 footage optimized for Intel OpenVINO and QuickSync."

# Fetch PAT from environment variable
GITHUB_PAT = os.environ.get("GITHUB_PAT")
if not GITHUB_PAT:
    print("❌ Error: GITHUB_PAT environment variable not found.")
    print("Please set it in PowerShell using: $env:GITHUB_PAT='your_token'")
    sys.exit(1)

# 1. Define Project Directory Structure
DIRECTORIES = [
    "config",
    "models",
    "notebooks",
    "src/ingestion",
    "src/perception",
    "src/kinematics",
    "src/render",
    "ui"
]

def create_workspace():
    print("🏗️ Creating local workspace directories...")
    for folder in DIRECTORIES:
        os.makedirs(folder, exist_ok=True)
        # Create an __init__.py file for Python package directories
        if "src" in folder:
            with open(os.path.join(folder, "__init__.py"), "w") as f:
                pass
        # Add a placeholder file to empty directories so Git tracks them
        elif folder in ["models", "notebooks"]:
            with open(os.path.join(folder, ".gitkeep"), "w") as f:
                pass
    print("✅ Local directory layout initialized.")

# 2. Populate Baseline Artifact Files (Using safe arrays to avoid triple-quote bugs)
def generate_base_files():
    print("📄 Generating base configuration and markdown files...")
    
    # .gitignore lines
    gitignore_lines = [
        "# Python baseline",
        "__pycache__/",
        "*.py[cod]",
        "*$py.class",
        ".venv/",
        "venv/",
        "ENV/",
        "env/",
        "",
        "# Data and Models",
        "models/*.xml",
        "models/*.bin",
        "models/*.pt",
        "*.insv",
        "*.lrv",
        "*.mp4",
        "",
        "# IDE files",
        ".vscode/",
        ".idea/"
    ]
    with open(".gitignore", "w", encoding="utf-8") as f:
        f.write("\n".join(gitignore_lines) + "\n")

    # config/requirements.txt lines
    requirements_lines = [
        "openvino==2024.1.0",
        "openvino-telemetry",
        "streamlit",
        "opencv-python",
        "ffmpeg-python",
        "numpy",
        "pandas",
        "requests",
        "torch --index-url https://download.pytorch.org/whl/cpu"
    ]
    with open("config/requirements.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(requirements_lines) + "\n")

    # README.md lines
    readme_lines = [
        f"# {REPO_NAME}",
        "",
        REPO_DESCRIPTION,
        "",
        "## 🚀 System Architecture Framework",
        "- **Ingestion Module:** Dual-track synchronization mapping low-res `.lrv` proxy metrics to high-res `.insv` stitched source binaries.",
        "- **Perception Engine:** Optimized via **Intel OpenVINO** running target detection (YOLOv10 / RT-DETR) and pixel matching (SAM 2) paths natively on Intel Arc graphics assets.",
        "- **Kinematic Stabilizer:** Discrete Kalman filtering loops resolving target blackouts and Savitzky-Golay path-smoothing logic.",
        "- **Render Engine:** Direct hardware-accelerated **FFmpeg Intel QuickSync Video (QSV)** compilation exporting custom 16:9 and 9:16 video payloads.",
        "",
        "## 🛠️ Local Development Installation",
        "1. Install OpenVINO core dependencies and FFmpeg locally.",
        "2. Initialize environment:",
        "   pip install -r config/requirements.txt"
    ]
    with open("README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines) + "\n")
        
    print("✅ Baseline configuration files written.")

# 3. Create Remote Repository via GitHub API
def create_github_repo():
    print(f"🌐 Creating remote GitHub repository: '{REPO_NAME}'...")
    url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"token {GITHUB_PAT}",
        "Accept": "application/vnd.github.v3+json"
    }
    payload = {
        "name": REPO_NAME,
        "description": REPO_DESCRIPTION,
        "private": True,
        "auto_init": False
    }
    
    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 201:
        repo_data = response.json()
        print(f"✅ Successfully created remote repository!")
        return repo_data["clone_url"]
    elif response.status_code == 422:
        print(f"⚠️ Repository '{REPO_NAME}' already exists on GitHub. Proceeding with sync.")
        user_url = "https://api.github.com/user"
        user_res = requests.get(user_url, headers=headers).json()
        return f"https://github.com/{user_res['login']}/{REPO_NAME}.git"
    else:
        print(f"❌ Failed to create repository. Status Code: {response.status_code}")
        print(response.text)
        sys.exit(1)

# 4. Git Initialization and Initial Commit/Push
def initialize_git(clone_url):
    print("🚀 Initializing Git repository and pushing baseline modules...")
    try:
        authenticated_url = clone_url.replace("https://", f"https://{GITHUB_PAT}@")
        
        subprocess.run(["git", "init"], check=True)
        subprocess.run(["git", "add", "."], check=True)
        subprocess.run(["git", "commit", "-m", "Initial baseline commit: Structural system layout and config initialization"], check=True)
        subprocess.run(["git", "branch", "-M", "main"], check=True)
        
        remotes = subprocess.run(["git", "remote"], capture_output=True, text=True).stdout
        if "origin" in remotes:
            subprocess.run(["git", "remote", "remove", "origin"], check=True)
            
        subprocess.run(["git", "remote", "add", "origin", authenticated_url], check=True)
        subprocess.run(["git", "push", "-u", "origin", "main"], check=True)
        print(f"🎉 Success! Project repository is active at: {clone_url}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Git automation step execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_workspace()
    generate_base_files()
    remote_url = create_github_repo()
    initialize_git(remote_url)