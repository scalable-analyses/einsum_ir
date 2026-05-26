"""``python -m etops.textir`` entry point."""

from __future__ import annotations

import sys

from etops.textir.cli import main

if __name__ == "__main__":
    sys.exit(main())
