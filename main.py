import numpy as np
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.camera import CameraIntrinsics, PinholeProjector
from src.geometry import SpatialGeometryEngine, DistanceMeasurement, PlaneFitResult

app = FastAPI(
    title="3D Spatial Intelligence & Metric Measurement Engine",
    description="Pinhole camera back-projection, 3D point-cloud reconstruction, and RANSAC geometric plane fitting API.",
    version="1.0.0"
)

class PixelCoord(BaseModel):
    u: float
    v: float
    depth_m: float = Field(..., gt=0.0)

class MeasurePixelsRequest(BaseModel):
    point_a: PixelCoord
    point_b: PixelCoord
    intrinsics: CameraIntrinsics = CameraIntrinsics()

class PlaneFitRequest(BaseModel):
    points_3d: List[List[float]] = Field(..., min_length=3)
    distance_threshold: float = Field(default=0.03, gt=0.0)

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "service": "spatial-3d-vision", "version": "1.0.0"}

@app.post("/measure-3d", response_model=DistanceMeasurement, tags=["Spatial Metrology"])
def measure_real_world_distance(payload: MeasurePixelsRequest):
    projector = PinholeProjector(payload.intrinsics)
    pt_a = projector.pixel_to_point3d(payload.point_a.u, payload.point_a.v, payload.point_a.depth_m)
    pt_b = projector.pixel_to_point3d(payload.point_b.u, payload.point_b.v, payload.point_b.depth_m)
    return SpatialGeometryEngine.compute_metric_distance(pt_a, pt_b)

@app.post("/fit-plane", response_model=PlaneFitResult, tags=["3D Segmentation"])
def segment_dominant_plane(payload: PlaneFitRequest):
    arr = np.array(payload.points_3d, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise HTTPException(status_code=400, detail="points_3d must have shape (N, 3).")
    return SpatialGeometryEngine.fit_plane_ransac(arr, distance_threshold=payload.distance_threshold)
