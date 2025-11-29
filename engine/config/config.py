'''
PRISM Engine Configuration
===========================
'''

from pathlib import Path
import os

BASE_DIR = Path("/content/drive/MyDrive/prism_engine")

DATA_RAW = BASE_DIR / "./data/raw"
DATA_CLEAN = BASE_DIR / "./data/cleaned"
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
