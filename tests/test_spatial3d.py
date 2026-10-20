import numpy as np
from fastapi.testclient import TestClient
from main import app
from src.camera import CameraIntrinsics, PinholeProjector
from src.geometry import SpatialGeometryEngine
from src.depth import MonocularDepthEstimator

def test_api_health():
    with TestClient(app) as client:
        res = client.get("/health")
        assert res.status_code == 200
        assert res.json()["status"] == "healthy"

def test_pinhole_backprojection_and_metric_distance():
    intrinsics = CameraIntrinsics(fx=500.0, fy=500.0, cx=320.0, cy=240.0)
    projector = PinholeProjector(intrinsics)

    # Principal point ray (u=cx, v=cy) at Z=2.0m must map to (0, 0, 2.0)
    pt_origin = projector.pixel_to_point3d(320.0, 240.0, 2.0)
    assert pt_origin == (0.0, 0.0, 2.0)

    # 100 pixels right at Z=2.0m with fx=500 -> X = (100 * 2.0) / 500 = 0.40m (40 cm)
    pt_shifted = projector.pixel_to_point3d(420.0, 240.0, 2.0)
    assert pt_shifted == (0.4, 0.0, 2.0)

    dist = SpatialGeometryEngine.compute_metric_distance(pt_origin, pt_shifted)
    assert dist.distance_meters == 0.4
    assert dist.distance_cm == 40.0

def test_ransac_3d_plane_recovery():
    rng = np.random.default_rng(42)
    # 90 inliers on plane Z = 1.50
    xy = rng.uniform(-2.0, 2.0, size=(90, 2))
    z = np.full((90, 1), 1.50) + rng.normal(0.0, 0.002, size=(90, 1))
    inliers = np.hstack([xy, z])

    # 10 extreme outliers
    outliers = rng.uniform(5.0, 10.0, size=(10, 3))
    cloud = np.vstack([inliers, outliers])

    fit = SpatialGeometryEngine.fit_plane_ransac(cloud, iterations=120, distance_threshold=0.02)
    assert abs(fit.normal_vector[2]) > 0.99
    assert fit.inlier_count >= 88
    assert fit.inlier_ratio >= 0.88

def test_monocular_depth_map_and_cloud_generation():
    synth_img = np.zeros((64, 64, 3), dtype=np.uint8)
    estimator = MonocularDepthEstimator(min_depth_m=0.5, max_depth_m=4.0)
    depth_map = estimator.estimate_depth(synth_img)

    assert depth_map.shape == (64, 64)
    assert float(np.min(depth_map)) >= 0.5
    assert float(np.max(depth_map)) <= 4.0

    projector = PinholeProjector()
    cloud = projector.depth_map_to_point_cloud(depth_map, stride=4)
    assert cloud.shape == (256, 3)

def test_api_measure_3d_endpoint():
    with TestClient(app) as client:
        payload = {
            "point_a": {"u": 319.5, "v": 239.5, "depth_m": 1.5},
            "point_b": {"u": 319.5, "v": 239.5, "depth_m": 2.5}
        }
        res = client.post("/measure-3d", json=payload)
        assert res.status_code == 200
        body = res.json()
        assert body["distance_meters"] == 1.0
        assert body["distance_cm"] == 100.0
