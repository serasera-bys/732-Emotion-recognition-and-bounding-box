from __future__ import annotations

from pathlib import Path
from typing import Any

from config import (
    DEFAULT_LIVE_CONFIDENCE_THRESHOLD,
    DEFAULT_MIN_BRIGHTNESS,
    DEFAULT_MIN_FACE_AREA_RATIO,
    DEFAULT_MIN_SHARPNESS,
    DEFAULT_PARTIAL_FACE_BORDER_MARGIN,
    DEFAULT_STATE_PERSISTENCE_FRAMES,
    DEFAULT_STATE_WINDOW_SECONDS,
    YOLO_WEIGHTS_PATH,
)
from emotional_state import EmotionalStateTracker
from face_detectors import build_face_detector, choose_face_from_candidates
from inference import Inferencer
from stability import NON_EMOTION_STATES
from yolov8_face import apply_neutral_bias, estimate_brightness, estimate_sharpness


class EmotionRuntime:
    def __init__(
        self,
        *,
        checkpoint: str | Path,
        detector_weights: str | Path = YOLO_WEIGHTS_PATH,
        device: str = "auto",
        detector_confidence: float = 0.25,
        emotion_confidence_threshold: float = DEFAULT_LIVE_CONFIDENCE_THRESHOLD,
        min_face_area_ratio: float = DEFAULT_MIN_FACE_AREA_RATIO,
        partial_face_border_margin: int = DEFAULT_PARTIAL_FACE_BORDER_MARGIN,
        min_brightness: float = DEFAULT_MIN_BRIGHTNESS,
        min_sharpness: float = DEFAULT_MIN_SHARPNESS,
        top_k: int = 5,
        state_window_seconds: float = DEFAULT_STATE_WINDOW_SECONDS,
        state_persistence_frames: int = DEFAULT_STATE_PERSISTENCE_FRAMES,
        fps: int = 30,
    ) -> None:
        self.detector = build_face_detector("yolov8n-face", weights=str(detector_weights), conf=detector_confidence)
        self.inferencer = Inferencer(checkpoint, device=device)
        self.emotion_confidence_threshold = float(emotion_confidence_threshold)
        self.min_face_area_ratio = float(min_face_area_ratio)
        self.partial_face_border_margin = int(partial_face_border_margin)
        self.min_brightness = float(min_brightness)
        self.min_sharpness = float(min_sharpness)
        self.top_k = max(1, int(top_k))
        window_size = max(3, int(state_window_seconds * max(1, fps)))
        self.state_tracker = EmotionalStateTracker(
            window_size=window_size,
            persistence_frames=state_persistence_frames,
        )

    def analyze_frame(self, frame) -> dict[str, Any]:
        candidates = self.detector.detect(frame)
        face_info = choose_face_from_candidates(
            candidates,
            frame.shape,
            min_face_area_ratio=self.min_face_area_ratio,
            border_margin=self.partial_face_border_margin,
        )

        result: dict[str, Any] = {
            "face_detected": face_info["status"] == "ok",
            "face_status": face_info["status"],
            "bbox": face_info.get("bbox"),
            "face_confidence": float(face_info.get("face_confidence", 0.0)),
            "emotion": "unknown",
            "emotion_confidence": 0.0,
            "emotional_state": "unknown",
            "state_confidence": 0.0,
            "state_explanation": "",
            "top_predictions": [],
            "state_dominant_emotions": [],
        }

        bbox = face_info.get("bbox")
        if face_info["status"] != "ok" or bbox is None:
            state = self.state_tracker.update(None)
            result["emotional_state"] = state.label
            result["state_explanation"] = state.explanation
            return result

        x1, y1, x2, y2 = bbox
        face_img = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
        if face_img.size == 0:
            result["face_status"] = "small_face"
            self.state_tracker.update(None)
            return result

        brightness = estimate_brightness(face_img)
        sharpness = estimate_sharpness(face_img)
        if brightness < self.min_brightness:
            result["face_status"] = "low_light"
            self.state_tracker.update(None)
            return result
        if sharpness < self.min_sharpness:
            result["face_status"] = "blurry_face"
            self.state_tracker.update(None)
            return result

        tensor = self.inferencer.preprocess_image(face_img)
        predicted_idx, confidence, score_map, top_predictions = self.inferencer.predict_distribution_tensor(
            tensor,
            top_k=self.top_k,
        )
        emotion = self.inferencer.class_names[predicted_idx]
        overridden_label, _ = apply_neutral_bias(
            score_map,
            neutral_bias_threshold=0.35,
            neutral_bias_margin=0.12,
            neutral_fallback_threshold=0.18,
            neutral_fallback_max_confidence=0.96,
        )
        if overridden_label is not None:
            emotion = overridden_label
            confidence = float(score_map.get("neutral", confidence))
        if emotion in NON_EMOTION_STATES or confidence < self.emotion_confidence_threshold:
            emotion = "unknown"

        state = self.state_tracker.update(score_map if emotion != "unknown" else None)
        result.update(
            {
                "emotion": emotion,
                "emotion_confidence": float(confidence),
                "emotional_state": state.label,
                "state_confidence": float(state.confidence),
                "state_explanation": state.explanation,
                "top_predictions": top_predictions,
                "state_dominant_emotions": state.dominant_emotions,
            }
        )
        return result
