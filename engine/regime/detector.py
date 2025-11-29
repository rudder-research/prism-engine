"""PRISM Regime Detector"""

import numpy as np
import pandas as pd

class RegimeDetector:
    def __init__(self, coherence_threshold: float = 0.3):
        self.coherence_threshold = coherence_threshold
        
    def detect_regime(self, state_vector: np.ndarray, history: pd.DataFrame):
        magnitude = np.linalg.norm(state_vector)
        
        if len(history) > 1:
            prev_state = history.iloc[-2].values
            prev_norm = prev_state / (np.linalg.norm(prev_state) + 1e-10)
            curr_norm = state_vector / (magnitude + 1e-10)
            cos_angle = np.clip(np.dot(prev_norm, curr_norm), -1.0, 1.0)
            rotation = np.arccos(cos_angle)
        else:
            rotation = 0.0
        
        if magnitude > 2.5 or rotation > 1.0:
            regime, confidence = 'CRISIS', 0.9
        elif magnitude > 1.5:
            regime, confidence = 'STRESS', 0.7
        elif rotation > 0.5:
            regime, confidence = 'TRANSITION', 0.6
        else:
            regime, confidence = 'NORMAL', 0.5
        
        return {
            'regime': regime,
            'confidence': confidence,
            'magnitude': magnitude,
            'rotation': rotation
        }
