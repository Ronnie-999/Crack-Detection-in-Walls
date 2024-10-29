# Essential Matrix and Camera Calibration

## Overview

This project provides a comprehensive pipeline for computing the Essential Matrix, Camera Calibration, and triangulation of 3D points from two sets of corresponding 2D image points. The code is translated from MATLAB to Python and leverages several key computer vision techniques to achieve this.

The steps include:

1. **Extraction of Calibration Points:** Loading 2D and 3D points from a data file.
2. **Fundamental Matrix Calculation:** Estimating the fundamental matrix using corresponding points.
3. **Camera Calibration Matrices (K1, K2):** Deriving the camera calibration matrices from the fundamental matrix.
4. **Essential Matrix Calculation:** Computing the essential matrix using calibrated 2D points.
5. **Fourfold Ambiguity Resolution:** Resolving the inherent fourfold ambiguity in essential matrix-based 3D reconstruction.
6. **Epipolar Geometry Visualization:** Displaying the epipolar lines for the given points.
7. **Geometric Error Calculation:** Evaluating the geometric error in the reconstruction.
8. **Levenberg-Marquardt Optimization:** Refining the essential matrix to minimize geometric error.

## Dependencies

This project uses Python's `numpy`, `scipy`, and `matplotlib` libraries.

Install the necessary packages using:

```bash
pip install numpy scipy matplotlib
