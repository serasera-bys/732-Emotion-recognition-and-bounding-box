from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2


@dataclass
class FaceCandidate:
    bbox: list[int]
    confidence: float


def bbox_area(box: list[int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def choose_face_from_candidates(
    candidates: list[FaceCandidate],
    frame_shape,
    *,
    min_face_area_ratio: float,
    border_margin: int,
) -> dict[str, Any]:
    if not candidates:
        return {"status": "no_face", "bbox": None, "face_confidence": 0.0, "face_count": 0}

    ordered = sorted(candidates, key=lambda item: bbox_area(item.bbox), reverse=True)
    best = ordered[0]
    frame_height, frame_width = frame_shape[:2]
    frame_area = max(1, frame_height * frame_width)
    area_ratio = bbox_area(best.bbox) / frame_area
    x1, y1, x2, y2 = best.bbox

    if len(ordered) > 1:
        return {
            "status": "multiple_faces",
            "bbox": best.bbox,
            "face_confidence": best.confidence,
            "face_count": len(ordered),
            "area_ratio": area_ratio,
        }
    if area_ratio < min_face_area_ratio:
        return {
            "status": "small_face",
            "bbox": best.bbox,
            "face_confidence": best.confidence,
            "face_count": 1,
            "area_ratio": area_ratio,
        }
    if x1 <= border_margin or y1 <= border_margin or x2 >= frame_width - border_margin or y2 >= frame_height - border_margin:
        return {
            "status": "partial_face",
            "bbox": best.bbox,
            "face_confidence": best.confidence,
            "face_count": 1,
            "area_ratio": area_ratio,
        }
    return {
        "status": "ok",
        "bbox": best.bbox,
        "face_confidence": best.confidence,
        "face_count": 1,
        "area_ratio": area_ratio,
    }


class BaseFaceDetector:
    name = "base"
    integration_complexity = "Unknown"

    def detect(self, frame) -> list[FaceCandidate]:
        raise NotImplementedError


class YoloV8FaceDetector(BaseFaceDetector):
    name = "yolov8n-face"
    integration_complexity = "Low"

    def __init__(self, *, weights: str, conf: float = 0.25) -> None:
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.conf = conf

    def detect(self, frame) -> list[FaceCandidate]:
        results = self.model.predict(source=frame, verbose=False, conf=self.conf)
        result = results[0] if results else None
        if result is None or result.boxes is None:
            return []
        return [
            FaceCandidate(
                bbox=[int(value) for value in xyxy.tolist()],
                confidence=float(conf),
            )
            for xyxy, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy())
        ]


class RetinaFaceDetector(BaseFaceDetector):
    name = "retinaface"
    integration_complexity = "High"

    def __init__(self) -> None:
        try:
            from retinaface import RetinaFace
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("retinaface package is not installed") from exc
        self._detector = RetinaFace

    def detect(self, frame) -> list[FaceCandidate]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self._detector.detect_faces(rgb)
        if not detections or isinstance(detections, tuple):
            return []
        candidates: list[FaceCandidate] = []
        for detection in detections.values():
            facial_area = detection.get("facial_area", [])
            if len(facial_area) != 4:
                continue
            x1, y1, x2, y2 = [int(value) for value in facial_area]
            confidence = float(detection.get("score", detection.get("confidence", 0.0)))
            candidates.append(FaceCandidate(bbox=[x1, y1, x2, y2], confidence=confidence))
        return candidates


class MediaPipeFaceDetector(BaseFaceDetector):
    name = "mediapipe-face-detection"
    integration_complexity = "Medium"

    def __init__(self, *, min_detection_confidence: float = 0.5) -> None:
        try:
            import mediapipe as mp
        except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("mediapipe package is not installed") from exc
        self._mp_face_detection = mp.solutions.face_detection
        self._detector = self._mp_face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, frame) -> list[FaceCandidate]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._detector.process(rgb)
        if not result.detections:
            return []
        frame_height, frame_width = frame.shape[:2]
        candidates: list[FaceCandidate] = []
        for detection in result.detections:
            relative_box = detection.location_data.relative_bounding_box
            x1 = max(0, int(relative_box.xmin * frame_width))
            y1 = max(0, int(relative_box.ymin * frame_height))
            x2 = min(frame_width, x1 + int(relative_box.width * frame_width))
            y2 = min(frame_height, y1 + int(relative_box.height * frame_height))
            candidates.append(
                FaceCandidate(
                    bbox=[x1, y1, x2, y2],
                    confidence=float(detection.score[0]) if detection.score else 0.0,
                )
            )
        return candidates


def build_face_detector(name: str, *, weights: str, conf: float = 0.25):
    normalized = name.strip().lower()
    if normalized in {"yolo", "yolov8", "yolov8-face", "yolov8n-face"}:
        return YoloV8FaceDetector(weights=weights, conf=conf)
    if normalized in {"retinaface", "retina-face"}:
        return RetinaFaceDetector()
    if normalized in {"mediapipe", "mediapipe-face", "mediapipe-face-detection"}:
        return MediaPipeFaceDetector(min_detection_confidence=conf)
    raise ValueError(f"Unsupported face detector: {name}")
