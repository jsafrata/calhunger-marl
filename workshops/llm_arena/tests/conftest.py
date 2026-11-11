import sys
from pathlib import Path

WORKSHOPS_ROOT = Path(__file__).resolve().parents[2]
if str(WORKSHOPS_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSHOPS_ROOT))
