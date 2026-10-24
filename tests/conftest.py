"""Make the top-level `pneumonia` package importable when pytest runs from any folder."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
