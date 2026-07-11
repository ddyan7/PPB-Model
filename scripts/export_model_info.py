"""Export the static model-info JSON consumed by the web UI on GitHub Pages.

`MODEL_INFO` in serve/model_info.py is the canonical source; this writes an
identical docs/model_info.json so the Pages frontend can render the model's
provenance, metrics, and attribution without calling the (scale-to-zero) API.

Run after changing MODEL_INFO:
    python scripts/export_model_info.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "serve"))
from model_info import MODEL_INFO  # noqa: E402

DST = ROOT / "docs" / "model_info.json"


def main() -> None:
    DST.write_text(json.dumps(MODEL_INFO, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"Wrote {DST} ({DST.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
