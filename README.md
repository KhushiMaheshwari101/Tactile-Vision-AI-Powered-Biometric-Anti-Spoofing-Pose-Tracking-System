# Tactile-Vision: AI-Powered Biometric Anti-Spoofing & Pose Tracking System

A computer vision system that performs real-time facial pose estimation and anti-spoofing detection using MediaPipe's face landmarking model and OpenCV.

**Run on Google Colab:** [Open Notebook]
(https://colab.research.google.com/drive/1TE8zMjX4buBE-6a-mLFOAntl5IA3wp5i)

---

## Overview

This project detects whether a face in an image is a real, three-dimensional subject or a static spoof (such as a printed photograph or screen display). It uses 478 facial landmarks to estimate head pose angles — yaw, pitch, and roll — and applies geometric analysis to distinguish live faces from flat, static ones.

---

## How It Works

1. **Face Landmark Detection** — MediaPipe's Face Landmarker model detects 478 facial landmarks from the input image
2. **Pose Estimation** — Six key landmarks (nose, chin, eyes, mouth corners) are mapped to a standard 3D face geometry model using OpenCV's `solvePnP` function
3. **Euler Angle Extraction** — Yaw, pitch, and roll angles are computed from the rotation matrix
4. **Anti-Spoofing Decision** — A face with near-zero yaw and pitch angles is classified as a static spoof; otherwise it is classified as a real 3D subject
5. **Visual Overlay** — Pose angles and detection status are rendered directly onto the output image

---

## Features

- 478-point facial landmark detection using MediaPipe
- 3D head pose estimation (yaw, pitch, roll) via PnP solver
- Anti-spoofing classification based on geometric pose analysis
- Visual mesh overlay with real-time status annotation
- Runs entirely in Google Colab — no local setup required

---

## Tech Stack

| Technology | Purpose |
|-----------|---------|
| MediaPipe (Face Landmarker) | Facial landmark detection |
| OpenCV | Image processing and pose estimation |
| NumPy | Matrix operations and geometric calculations |
| Google Colab | Execution environment |
| Python | Core language |

---

## Getting Started

**1. Open the notebook in Google Colab**

**2. Install dependencies**
```python
!pip install mediapipe opencv-python-headless
```

**3. Download the Face Landmarker model**
```python
!wget -O face_landmarker.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

**4. Upload a test image**

Upload a `.jpg` image to Colab and update the `image_path` variable in the notebook:
```python
image_path = "your_image.jpg"
```

**5. Run all cells**

The output will display the annotated image with pose angles and anti-spoofing status.

---

## Output

- **STATUS: REAL 3D TARGET** — Face has non-zero pose angles; classified as a live subject
- **STATUS: SPOOF / STATIC** — Face has near-zero yaw and pitch; classified as a static image

---

## Project Structure

```
tactile-vision-biometric/
├── Tactile-Vision_AI-Powered_Biometric_Anti-Spoofing_Pose_Tracking_System.ipynb
└── README.md
```

---

## Author

**Khushi Maheshwari**

[GitHub](https://github.com/KhushiMaheshwari101)
