"""Launch MedQuanv Workbench desktop application from Terminal:
    python scripts/medquanv_app.py
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.medquanv_workbench import main  # noqa: E402

if __name__ == "__main__":
    main()
