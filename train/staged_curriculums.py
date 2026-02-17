"""train/staged_curriculums.py (LEGACY)

This file used to contain a separate curriculum dict.
It is kept as a tiny shim so existing imports keep working.

New canonical location:
  config/curriculum.py
"""

from config.curriculum import CURRICULUM  # re-export (backwards compatible)
