# 🧊 spatial-3d-vision: 3D Point-Cloud Reconstruction & Spatial Metrology

[![Spatial 3D Vision CI](https://github.com/asad-mj/spatial-3d-vision/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/asad-mj/spatial-3d-vision/actions/workflows/ci-cd.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python)](https://python.org)
[![3D WebGL](https://img.shields.io/badge/Visualizer-Plotly_3D-3F4F75?style=flat-square)](https://plotly.com)

A 3D computer vision and spatial metrology engine implementing **pinhole camera inverse projection**, **monocular depth map synthesis**, **RANSAC 3D plane segmentation**, and **sub-centimeter Euclidean physical measurement**.

---

## 📐 Projective Geometry & RANSAC Formulation

### 1. Pinhole Camera Back-Projection ($\mathbb{P}^2 \to \mathbb{R}^3$)
Given the camera intrinsic calibration matrix $\mathbf{K}$ and metric depth $Z$, a 2D homogeneous pixel $\tilde{\mathbf{u}} = [u, v, 1]^\top$ is back-projected into 3D camera coordinates $\mathbf{X}_c = [X, Y, Z]^\top$:

$$\mathbf{X}_c = Z \cdot \mathbf{K}^{-1} \tilde{\mathbf{u}} = \begin{bmatrix} \frac{(u - c_x) Z}{f_x} \\ \frac{(v - c_y) Z}{f_y} \\ Z \end{bmatrix}, \quad \mathbf{K} = \begin{bmatrix} f_x & 0 & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}$$

### 2. 3D RANSAC Plane Segmentation
Minimal 3-point subsets $\{\mathbf{p}_1, \mathbf{p}_2, \mathbf{p}_3\}$ define candidate surface normals $\hat{\mathbf{n}} = \frac{(\mathbf{p}_2 - \mathbf{p}_1) \times (\mathbf{p}_3 - \mathbf{p}_1)}{\|(\mathbf{p}_2 - \mathbf{p}_1) \times (\mathbf{p}_3 - \mathbf{p}_1)\|_2}$, maximizing the consensus inlier set $\mathcal{I} = \{ \mathbf{p}_i \mid |\hat{\mathbf{n}}^\top \mathbf{p}_i + d| \le \tau \}$.

---

## 🚀 Quick Start

```bash
git clone [https://github.com/asad-mj/spatial-3d-vision.git](https://github.com/asad-mj/spatial-3d-vision.git)
cd spatial-3d-vision

pip install -r requirements.txt
python -m pytest tests/ -v

# Start REST API Server
uvicorn main:app --reload --port 8000

# Launch Interactive 3D WebGL Workbench
streamlit run app.py
