# Emotion Recognition and Bounding Box

Minimal runtime package for integrating face detection and live emotion recognition into another application.

## Included files

- `yolov8_face.py`: webcam face detection + live emotion overlay
- `emotion_runtime_api.py`: JSON-ready runtime wrapper for chatbot integration
- `emotional_state.py`: derived emotional-state layer
- `face_detectors.py`: shared face detector adapters
- `inference.py`: image and tensor emotion inference helpers
- `models.py`: emotion classification model builder
- `transforms.py`: image preprocessing transforms
- `stability.py`: temporal smoothing and stable-label logic
- `config.py`: runtime paths and defaults
- `yolov8n-face.pt`: YOLOv8 face detector weights
- `artifacts/checkpoints/live_webcam_run1/best.pt`: live webcam emotion model checkpoint

## Intended use

This package is the integration subset from the full training project. 6 emotion : anger, fear, happy, neutral, sad, suprise

## Main runtime assets

- Face detection model: `YOLOv8n-face`
- Emotion backbone: `ConvNeXtV2 Pico`
- Live deployment checkpoint: `artifacts/checkpoints/live_webcam_run1/best.pt`

## Emotional state update

The runtime now derives a higher-level emotional state from rolling emotion probabilities. Supported states:

- `calm`
- `positive`
- `stressed`
- `frustrated`
- `overwhelmed`

This is a rule-based temporal layer on top of the six-class emotion classifier, not a separate trained model.

For chatbot integration, use `EmotionRuntime` from `emotion_runtime_api.py`. It returns a dictionary with:

- `face_detected`
- `bbox`
- `emotion`
- `emotion_confidence`
- `emotional_state`
- `state_confidence`
- `top_predictions`

## Benchmark results

The emotion model benchmark table is included under:

- `artifacts/benchmarks/emotion_models/latest/benchmark_summary.csv`
- `artifacts/benchmarks/emotion_models/latest/benchmark_summary.md`
- `artifacts/benchmarks/emotion_models/latest/benchmark_summary.json`

## future development
- try another emotion model
- try combine emotion to make another label
