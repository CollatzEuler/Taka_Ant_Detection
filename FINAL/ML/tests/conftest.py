from __future__ import annotations

import sys
from pathlib import Path


# Let the copied pipeline's tests import its local ML package without an
# editable install or a PYTHONPATH inherited from another checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
