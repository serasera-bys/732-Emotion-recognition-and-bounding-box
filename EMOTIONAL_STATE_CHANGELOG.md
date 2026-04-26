# Emotional State Update

This update adds a higher-level emotional state layer on top of the existing six-class webcam emotion model.

## Base Emotion Model

- Backbone: ConvNeXtV2 Pico
- Checkpoint: `artifacts/checkpoints/live_webcam_run1/best.pt`
- Base classes: `anger`, `fear`, `happy`, `neutral`, `sad`, `surprise`

## Derived Emotional States

The emotional state layer is not a separate trained model. It reads rolling emotion probabilities from the base classifier and maps them into:

- `calm`
- `positive`
- `stressed`
- `frustrated`
- `overwhelmed`

## Runtime Integration

- `yolov8_face.py` now displays the derived state in the webcam overlay.
- `state_explanation` is written to CSV logs but is no longer shown on-screen, keeping the overlay readable.
- `emotion_runtime_api.py` exposes a JSON-ready runtime interface for chatbot integration.

Example output:

```json
{
  "face_detected": true,
  "face_status": "ok",
  "bbox": [120, 80, 420, 430],
  "face_confidence": 0.86,
  "emotion": "neutral",
  "emotion_confidence": 0.95,
  "emotional_state": "calm",
  "state_confidence": 0.96
}
```

## Model Benchmark Result

The emotion model benchmark is stored under:

- `artifacts/benchmarks/emotion_models/latest/benchmark_summary.csv`
- `artifacts/benchmarks/emotion_models/latest/benchmark_summary.md`
- `artifacts/benchmarks/emotion_models/latest/benchmark_summary.json`

The best overall model from the benchmark is `convnextv2_pico`.
