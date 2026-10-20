import numpy as np
from typing import List, Tuple
from pydantic import BaseModel

class DistanceMeasurement(BaseModel):
    point_a_3d: List[float]
    point_b_3d: List[float]
    distance_meters: float
    distance_cm: float

class PlaneFitResult(BaseModel):
    normal_vector: List[float]
    plane_offset_d: float
    inlier_count: int
    inlier_ratio: float

class SpatialGeometryEngine:
    """
    Provides 3D Euclidean metric measurements, bounding-box volume estimation,
    and RANSAC 3D plane fitting (aX + bY + cZ + d = 0).
    """

    @staticmethod
    def compute_metric_distance(
        pt_a: Tuple[float, float, float],
        pt_b: Tuple[float, float, float]
    ) -> DistanceMeasurement:
        a = np.array(pt_a, dtype=np.float64)
        b = np.array(pt_b, dtype=np.float64)
        dist_m = float(np.linalg.norm(a - b))

        return DistanceMeasurement(
            point_a_3d=[round(float(x), 4) for x in a],
            point_b_3d=[round(float(x), 4) for x in b],
            distance_meters=round(dist_m, 4),
            distance_cm=round(dist_m * 100.0, 2)
        )

    @staticmethod
    def fit_plane_ransac(
        points: np.ndarray,
        iterations: int = 100,
        distance_threshold: float = 0.03,
        seed: int = 42
    ) -> PlaneFitResult:
        """
        Fits a 3D plane (n · X + d = 0) to an (N, 3) point cloud via RANSAC.
        """
        n_pts = points.shape[0]
        if n_pts < 3:
            raise ValueError("At least 3 non-collinear 3D points are required for plane fitting.")

        rng = np.random.default_rng(seed)
        best_inliers = -1
        best_normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        best_d = 0.0

        for _ in range(iterations):
            idx = rng.choice(n_pts, size=3, replace=False)
            p1, p2, p3 = points[idx[0]], points[idx[1]], points[idx[2]]

            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_len = float(np.linalg.norm(normal))
            if norm_len < 1e-9:
                continue

            normal = normal / norm_len
            if normal[2] < 0:
                normal = -normal
            d = -float(np.dot(normal, p1))

            distances = np.abs(np.dot(points, normal) + d)
            inliers = int(np.sum(distances <= distance_threshold))

            if inliers > best_inliers:
                best_inliers = inliers
                best_normal = normal
                best_d = d

        return PlaneFitResult(
            normal_vector=[round(float(c), 4) for c in best_normal],
            plane_offset_d=round(best_d, 4),
            inlier_count=best_inliers,
            inlier_ratio=round(best_inliers / float(n_pts), 4)
        )
