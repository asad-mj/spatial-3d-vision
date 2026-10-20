import streamlit as st
import numpy as np
import plotly.graph_objects as go
from src.camera import CameraIntrinsics, PinholeProjector
from src.geometry import SpatialGeometryEngine

st.set_page_config(page_title="3D Spatial Intelligence", page_icon="🧊", layout="wide")
st.title("🧊 3D Spatial Intelligence & Point-Cloud Metrology Workbench")
st.markdown("Interactive **Pinhole Camera Back-Projection**, **RANSAC 3D Plane Segmentation**, and **Real-World Centimeter Measurement**.")

projector = PinholeProjector(CameraIntrinsics(fx=500.0, fy=500.0, cx=160.0, cy=120.0))

col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("📏 Virtual 3D Caliper")
    u1 = st.number_input("Point A — Pixel U", value=100.0)
    v1 = st.number_input("Point A — Pixel V", value=120.0)
    z1 = st.slider("Point A — Depth Z (m)", 0.5, 5.0, 2.0, 0.1)

    u2 = st.number_input("Point B — Pixel U", value=220.0)
    v2 = st.number_input("Point B — Pixel V", value=120.0)
    z2 = st.slider("Point B — Depth Z (m)", 0.5, 5.0, 2.0, 0.1)

    p3d_a = projector.pixel_to_point3d(u1, v1, z1)
    p3d_b = projector.pixel_to_point3d(u2, v2, z2)
    measurement = SpatialGeometryEngine.compute_metric_distance(p3d_a, p3d_b)

    st.metric("Physical 3D Distance", f"{measurement.distance_cm:.2f} cm", f"{measurement.distance_meters:.4f} m")

with col2:
    rng = np.random.default_rng(42)
    # Generate synthetic ground plane + object cluster
    plane_xy = rng.uniform(-1.0, 1.0, size=(350, 2))
    plane_z = np.full((350, 1), 2.0) + rng.normal(0, 0.01, size=(350, 1))
    cloud = np.hstack([plane_xy, plane_z])

    ransac_res = SpatialGeometryEngine.fit_plane_ransac(cloud)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2],
        mode="markers",
        marker=dict(size=3, color=cloud[:, 2], colorscale="Viridis", opacity=0.7),
        name="3D Point Cloud"
    ))
    fig.add_trace(go.Scatter3d(
        x=[p3d_a[0], p3d_b[0]],
        y=[p3d_a[1], p3d_b[1]],
        z=[p3d_a[2], p3d_b[2]],
        mode="lines+markers",
        line=dict(color="red", width=6),
        marker=dict(size=6, color="red"),
        name="Measured Span"
    ))
    fig.update_layout(height=480, margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"RANSAC Dominant Plane Normal: `{ransac_res.normal_vector}` | Inlier Ratio: `{ransac_res.inlier_ratio * 100:.1f}%`")
