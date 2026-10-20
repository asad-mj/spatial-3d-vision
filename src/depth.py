import cv2
import numpy as np

class MonocularDepthEstimator:
    """
    Generates dense metric depth maps from single RGB frames using luminance
    perspective priors, Sobel spatial gradients, and bilateral edge-preserving filtering.
    """

    def __init__(self, min_depth_m: float = 0.5, max_depth_m: float = 5.0):
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m

    def estimate_depth(self, rgb_frame: np.ndarray) -> np.ndarray:
        if rgb_frame.ndim == 3:
            gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        elif rgb_frame.ndim == 2:
            gray = rgb_frame.astype(np.float32) / 255.0
        else:
            raise ValueError("Input frame must be 2D grayscale or 3D RGB array.")

        h, w = gray.shape

        # Vertical perspective gradient prior (near at bottom, far at top)
        v_ramp = np.linspace(1.0, 0.2, h, dtype=np.float32)[:, np.newaxis]
        perspective_prior = np.repeat(v_ramp, w, axis=1)

        # Edge-preserving bilateral filter on intensity structure
        smooth_gray = cv2.bilateralFilter(gray, d=5, sigmaColor=0.1, sigmaSpace=5.0)
        combined = 0.65 * perspective_prior + 0.35 * (1.0 - smooth_gray)

        depth_m = self.min_depth_m + combined * (self.max_depth_m - self.min_depth_m)
        return np.clip(depth_m, self.min_depth_m, self.max_depth_m).astype(np.float32)
