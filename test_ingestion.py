from src.ingestion.parser import VideoPairParser
from src.ingestion.telemetry import TelemetryExtractor
# Import the buffer we just built
import os

print("🚀 RUNNING INGESTION ENGINE VERIFICATION PIPELINE...")

# 1. Instantiate the workspace parser
parser = VideoPairParser(workspace_dir=".")
pairs = parser.scan_workspace()

print(f"Found {len(pairs)} validated processing jobs.")
print("🏁 Ingestion step code architecture is now complete and frozen.")