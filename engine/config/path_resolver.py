"""
PRISM ENGINE — PATH RESOLVER (FINAL VERSION)
Automatically finds the working_copy folder and builds correct paths.
"""

from pathlib import Path

# ------------------------------------------------------------
# STEP 1 — THIS FILE LIVES HERE:
# prism-engine/working_copy/engine/config/path_resolver.py
#
# So parents[3] = working_copy
# ------------------------------------------------------------
WORKING_COPY = Path(__file__).resolve().parents[3]

# ------------------------------------------------------------
# STEP 2 — BUILD ALL CORE PATHS
# ------------------------------------------------------------
class PATHS:
    base = WORKING_COPY

    # Data folders
    data = WORKING_COPY / "data"
    data_raw = WORKING_COPY / "data" / "raw"
    data_clean = WORKING_COPY / "data" / "clean"
    data_registry = WORKING_COPY / "data" / "registry" / "prism_metric_registry.json"

    # Engine + Modules
    engine = WORKING_COPY / "engine"
    results = WORKING_COPY / "results"
    notebooks = WORKING_COPY / "notebooks"
    visualization = WORKING_COPY / "visualization"
    interpretation = WORKING_COPY / "interpretation"

    @staticmethod
    def debug():
        print("WORKING_COPY =", WORKING_COPY)
        print("data_raw =", PATHS.data_raw)
        print("data_registry =", PATHS.data_registry)
