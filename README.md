# Multimodal In-Bed Pose Estimation with Risk-Based Monitoring for Elderly Care

## Overview
This project presents a real-time system for in-bed pose estimation using computer vision. It detects patient posture and tracks duration to provide basic risk monitoring.

## Features
- Real-time pose detection using MediaPipe
- Posture classification (Supine, Left, Right)
- Duration tracking
- Risk monitoring (PURI, RCI, CQS)
- Fall detection
- Voice alerts
- Patient report generation

## Tech Stack
- Python
- OpenCV
- MediaPipe
- NumPy
- Matplotlib
- pyttsx3

## How to Run
```
pip install -r requirements.txt
python main.py
```

## Output
- Real-time monitoring dashboard
- Movement graph
- Patient report (PDF)

## Note
This system focuses on pose estimation with a basic risk monitoring layer and is not intended for clinical use.
