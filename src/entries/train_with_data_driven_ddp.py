#!/usr/bin/env python3
"""DDP entry wrapper forwarding to ``entries.train_with_data_driven.main``."""
import os
import sys

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from entries.train_with_data_driven import main

if __name__ == "__main__":
    main()
