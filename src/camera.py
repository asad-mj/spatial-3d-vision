import numpy as np
from typing import Tuple, List
from pydantic import BaseModel

class CameraIntrinsics(BaseModel):
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 319.5
    cy: float = 239.5

    def to_matrix(self) -> np.ndarray:
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

class PinholeProjector:
    """
    Implements perspective projection and inverse 2D-to-3D back-projection
    using camera intrinsic parameters K.
    """

    def __init__(self, intrinsics: CameraIntrinsics | None = None):
        self.intrinsics = intrinsics or CameraIntrinsics()

    def pixel_to_point3d(self, u: float, v: float, depth_m: float) -> Tuple[float, float, float]:
        """
        Back-projects a 2D pixel (u, v) with metric depth Z (meters) into 3D camera space (X, Y, Z).
        """
        if depth_m <= 0:
            raise ValueError("Depth Z must be strictly positive.")

        x = (u - self.intrinsics.cx) * depth_m / self.intrinsics.fx
        y = (v - self.intrinsics.cy) * depth_m / self.intrinsics.fy
        z = float(depth_m)
        return round(float(x), 5), round(float(y), 5), round(z, 5)

    def depth_map_to_point_cloud(
        self,
        depth_map: np.ndarray,
        stride: int = 2
    ) -> np.ndarray:
        """
        Vectorized conversion of a 2D depth map (H, W) into an (N, 3) 3D point cloud.
        """
        if depth_map.ndim != 2:
            raise ValueError("Expected 2D depth map array (H, W).")

        h, w = depth_map.shape
        v_grid, u_grid = np.mgrid[0:h:stride, 0:w:stride]
        z = depth_map[0:h:stride, 0:w:stride].astype(np.float64)

        valid = z > 1e-4
        u_valid = u_grid[valid]
        v_valid = v_grid[valid]
        z_valid = z[valid]

        x = (u_valid - self.intrinsics.cx) * z_valid / self.intrinsics.fx
        y = (v_valid - self.intrinsics.cy) * z_valid / self.intrinsics.fy

        return np.stack([x, y, z_valid], axis=-1)
