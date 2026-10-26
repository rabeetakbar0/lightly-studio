import sys
from pathlib import Path

# The pipeline package lives next to this folder and is not installed.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
