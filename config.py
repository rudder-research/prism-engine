'''
PRISM Engine Configuration
===========================
'''

from pathlib import Path
import os

# Always use relative paths based on this file's location
# This makes the project portable - drop it anywhere and it works
BASE_DIR = Path(__file__).parent.absolute()

# Allow override via environment variable (optional)
if "PRISM_ENGINE_BASE_DIR" in os.environ:
    BASE_DIR = Path(os.environ["PRISM_ENGINE_BASE_DIR"])

DATA_RAW = BASE_DIR / "data_raw"
DATA_CLEAN = BASE_DIR / "data_clean"
REGISTRY_DIR = BASE_DIR / "registry"
OUTPUTS_DIR = BASE_DIR / "outputs"

for directory in [DATA_RAW, DATA_CLEAN, REGISTRY_DIR, OUTPUTS_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

METRIC_REGISTRY = REGISTRY_DIR / "prism_metric_registry.json"

PILOT_INPUTS = ['SPY_50d_200d_MA', 'DGS10', 'DXY', 'AGG']

FREQUENCY_BANDS = {
    'fast': {'range': (1, 3)},
    'medium': {'range': (6, 12)},
    'slow': {'range': (24, 36)}
}

REGIME_THRESHOLDS = {
    'coherence_min': 0.3,
    'magnitude_low': 33,
    'magnitude_high': 67,
}

__version__ = "1.0.0"
__author__ = "Jason Rudder"
__project__ = "PRISM Engine"
