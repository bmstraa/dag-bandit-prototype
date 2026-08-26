# src/drift_detector.py
import numpy as np
from typing import List

class CUSUMDriftDetector:
    """
    CUSUM change-point detector for monitoring a stream of confidence scores.
    Designed to detect negative shifts (drops in performance).
    """

    def __init__(self, k_factor: float = 0.5, h_factor: float = 3.0):
        self.k_factor = k_factor
        self.h_factor = h_factor
        self.s_lo = 0.0
        self.baseline_mean = None
        self.baseline_std = None

    def calibrate(self, scores: List[float]) -> None:
        """
        Establish baseline statistics (mean and std) during a burn-in phase.
        """
        self.baseline_mean = np.mean(scores)
        self.baseline_std = np.std(scores)
        if self.baseline_std == 0.0:
            self.baseline_std = 0.1  # guard against division by zero

    def update(self, score: float) -> bool:
        """
        Update the CUSUM statistic with a new confidence score.
        Returns True if a drift (drop) is detected.
        """
        if self.baseline_mean is None:
            raise ValueError("CUSUM must be calibrated first.")

        z_score = (self.baseline_mean - score) / self.baseline_std
        k = self.k_factor * self.baseline_std
        h = self.h_factor * self.baseline_std

        self.s_lo = max(0, self.s_lo + z_score - k)
        return self.s_lo > h
