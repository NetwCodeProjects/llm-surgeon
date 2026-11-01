#!/usr/bin/env python3
"""Acceptance or release record of a finished run (src/llm_surgeon/release_record.py; docs/acceptance.md).

python3 scripts/acceptance_record.py --workspace WS --run RUN [--setup-run A --reuse-run B] [--kind release]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_surgeon.release_record import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
