# Emotion Recognition and Bounding Box

Minimal runtime package for integrating face detection and live emotion recognition into another application.

## Included files

- `yolov8_face.py`: webcam face detection + live emotion overlay
- `inference.py`: image and tensor emotion inference helpers
- `models.py`: emotion classification model builder
- `transforms.py`: image preprocessing transforms
- `stability.py`: temporal smoothing and stable-label logic
- `config.py`: runtime paths and defaults
- `yolov8n-face.pt`: YOLOv8 face detector weights
- `artifacts/checkpoints/live_webcam_run1/best.pt`: live webcam emotion model checkpoint

## Intended use

This package is the integration subset from the full training project. It excludes dataset folders, training scripts, and evaluation artifacts.

## Main runtime assets

- Face detection model: `YOLOv8n-face`
- Emotion backbone: `ConvNeXtV2 Pico`
- Live deployment checkpoint: `artifacts/checkpoints/live_webcam_run1/best.pt`
