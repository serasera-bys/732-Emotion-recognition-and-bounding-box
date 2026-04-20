from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
from ultralytics import YOLO

from config import (
    DEFAULT_DEBOUNCE_FRAMES,
    DEFAULT_FLICKER_HOLD_FRAMES,
    DEFAULT_LIVE_CONFIDENCE_THRESHOLD,
    DEFAULT_MIN_BRIGHTNESS,
    DEFAULT_MIN_FACE_AREA_RATIO,
    DEFAULT_MIN_SHARPNESS,
    DEFAULT_PARTIAL_FACE_BORDER_MARGIN,
    DEFAULT_SMOOTHING_WINDOW,
    LIVE_LOGS_DIR,
    YOLO_WEIGHTS_PATH,
    ensure_project_dirs,
)
from inference import Inferencer
from stability import EmotionStabilizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stable YOLOv8 face detection and emotion classification on webcam frames.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--weights", default=str(YOLO_WEIGHTS_PATH))
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--confidence-threshold", type=float, default=DEFAULT_LIVE_CONFIDENCE_THRESHOLD)
    parser.add_argument("--smoothing-window", type=int, default=DEFAULT_SMOOTHING_WINDOW)
    parser.add_argument("--debounce-frames", type=int, default=DEFAULT_DEBOUNCE_FRAMES)
    parser.add_argument("--flicker-hold-frames", type=int, default=DEFAULT_FLICKER_HOLD_FRAMES)
    parser.add_argument("--min-face-area-ratio", type=float, default=DEFAULT_MIN_FACE_AREA_RATIO)
    parser.add_argument("--partial-face-border-margin", type=int, default=DEFAULT_PARTIAL_FACE_BORDER_MARGIN)
    parser.add_argument("--min-brightness", type=float, default=DEFAULT_MIN_BRIGHTNESS)
    parser.add_argument("--min-sharpness", type=float, default=DEFAULT_MIN_SHARPNESS)
    parser.add_argument("--class-threshold", action="append", default=[], help="Override confidence threshold per class, e.g. sad=0.7")
    parser.add_argument("--top-k-overlay", type=int, default=3)
    parser.add_argument("--neutral-bias-threshold", type=float, default=0.35)
    parser.add_argument("--neutral-bias-margin", type=float, default=0.12)
    parser.add_argument("--neutral-fallback-threshold", type=float, default=0.18)
    parser.add_argument("--neutral-fallback-max-confidence", type=float, default=0.96)
    parser.add_argument("--log-path", default="")
    return parser.parse_args()


def open_webcam(camera_index: int, width: int, height: int, fps: int):
    capture = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    if capture.isOpened():
        ok, frame = capture.read()
        if ok and frame is not None:
            return capture
        capture.release()

    capture = cv2.VideoCapture(camera_index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    if capture.isOpened():
        ok, frame = capture.read()
        if ok and frame is not None:
            return capture
        capture.release()
    return None


def bbox_area(box: list[int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def choose_face(result, frame_shape, min_face_area_ratio: float, border_margin: int) -> dict:
    if result.boxes is None or len(result.boxes) == 0:
        return {"status": "no_face", "bbox": None, "face_confidence": 0.0, "face_count": 0}

    boxes = []
    for xyxy, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        box = [int(value) for value in xyxy.tolist()]
        boxes.append((box, float(conf)))

    boxes.sort(key=lambda item: bbox_area(item[0]), reverse=True)
    best_box, best_conf = boxes[0]
    frame_height, frame_width = frame_shape[:2]
    frame_area = max(1, frame_height * frame_width)
    area_ratio = bbox_area(best_box) / frame_area
    x1, y1, x2, y2 = best_box

    if len(boxes) > 1:
        return {
            "status": "multiple_faces",
            "bbox": best_box,
            "face_confidence": best_conf,
            "face_count": len(boxes),
            "area_ratio": area_ratio,
        }
    if area_ratio < min_face_area_ratio:
        return {
            "status": "small_face",
            "bbox": best_box,
            "face_confidence": best_conf,
            "face_count": 1,
            "area_ratio": area_ratio,
        }
    if x1 <= border_margin or y1 <= border_margin or x2 >= frame_width - border_margin or y2 >= frame_height - border_margin:
        return {
            "status": "partial_face",
            "bbox": best_box,
            "face_confidence": best_conf,
            "face_count": 1,
            "area_ratio": area_ratio,
        }
    return {
        "status": "ok",
        "bbox": best_box,
        "face_confidence": best_conf,
        "face_count": 1,
        "area_ratio": area_ratio,
    }


def default_log_path() -> Path:
    ensure_project_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return LIVE_LOGS_DIR / f"live_emotion_log_{timestamp}.csv"


def parse_class_thresholds(entries: list[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for entry in entries:
        if "=" not in entry:
            continue
        label, value = entry.split("=", 1)
        try:
            thresholds[label.strip()] = float(value)
        except ValueError:
            continue
    return thresholds


def estimate_brightness(face_img) -> float:
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return float(gray.mean())


def estimate_sharpness(face_img) -> float:
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def draw_text_with_outline(frame, text: str, origin: tuple[int, int], color: tuple[int, int, int], *, scale: float = 0.7) -> None:
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (0, 0, 0),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_status(frame, face_info: dict, stabilized: dict) -> None:
    bbox = face_info.get("bbox")
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label_lines = [
        f"raw: {stabilized['raw_label']} ({stabilized['confidence']:.2f})",
        f"stable: {stabilized['stable_label']}",
        f"status: {stabilized['status']}",
    ]
    if stabilized.get("override_note"):
        label_lines.append(f"bias: {stabilized['override_note']}")
    for item in stabilized.get("top_k_predictions", []):
        label_lines.append(f"{item['class_name']}: {item['confidence']:.2f}")
    origin_x = 12
    origin_y = 28
    line_colors = [
        (0, 140, 255),
        (0, 255, 255),
        (80, 255, 80),
    ]
    for index, text in enumerate(label_lines):
        color = line_colors[index] if index < len(line_colors) else (255, 180, 0)
        draw_text_with_outline(
            frame,
            text,
            (origin_x, origin_y + index * 28),
            color,
            scale=0.7,
        )


def apply_neutral_bias(
    score_map: dict[str, float],
    *,
    neutral_bias_threshold: float,
    neutral_bias_margin: float,
    neutral_fallback_threshold: float,
    neutral_fallback_max_confidence: float,
) -> tuple[str | None, str]:
    if not score_map:
        return None, ""
    ranked_items = sorted(score_map.items(), key=lambda item: item[1], reverse=True)
    top_label, top_confidence = ranked_items[0]
    if top_label not in {"sad", "fear", "contempt", "surprise"}:
        return None, ""

    neutral_confidence = float(score_map.get("neutral", 0.0))
    if neutral_confidence >= neutral_bias_threshold and (top_confidence - neutral_confidence) <= neutral_bias_margin:
        return "neutral", f"neutral-close({top_label})"
    if top_label in {"sad", "fear", "contempt"} and neutral_confidence >= neutral_fallback_threshold and top_confidence <= neutral_fallback_max_confidence:
        return "neutral", f"neutral-fallback({top_label})"
    return None, ""


def main() -> None:
    args = parse_args()
    ensure_project_dirs()
    detector = YOLO(args.weights)
    inferencer = Inferencer(args.checkpoint, device=args.device)
    stabilizer = EmotionStabilizer(
        smoothing_window=args.smoothing_window,
        debounce_frames=args.debounce_frames,
        flicker_hold_frames=args.flicker_hold_frames,
        confidence_threshold=args.confidence_threshold,
    )
    class_thresholds = parse_class_thresholds(args.class_threshold)

    capture = open_webcam(args.camera_index, args.width, args.height, args.fps)
    if capture is None:
        raise RuntimeError("Could not open webcam.")

    log_path = Path(args.log_path) if args.log_path else default_log_path()
    with log_path.open("w", encoding="utf-8", newline="") as log_handle:
        writer = csv.DictWriter(
            log_handle,
            fieldnames=[
                "timestamp",
                "frame_index",
                "raw_label",
                "stable_label",
                "status",
                "raw_status",
                "emotion_confidence",
                "face_confidence",
                "face_count",
                "bbox",
                "brightness",
                "sharpness",
                "effective_threshold",
                "override_note",
                "top_predictions",
            ],
        )
        writer.writeheader()

        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_index += 1

                results = detector.predict(source=frame, verbose=False, conf=args.conf)
                result = results[0] if results else None
                if result is None:
                    face_info = {"status": "no_face", "bbox": None, "face_confidence": 0.0, "face_count": 0}
                else:
                    face_info = choose_face(
                        result,
                        frame.shape,
                        min_face_area_ratio=args.min_face_area_ratio,
                        border_margin=args.partial_face_border_margin,
                    )

                raw_label = face_info["status"]
                emotion_confidence = 0.0
                brightness = 0.0
                sharpness = 0.0
                effective_threshold = args.confidence_threshold
                top_k_predictions: list[dict[str, float | str]] = []
                override_note = ""
                bbox = face_info.get("bbox")
                if face_info["status"] == "ok" and bbox is not None:
                    x1, y1, x2, y2 = bbox
                    face_img = frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
                    if face_img.size == 0:
                        raw_label = "small_face"
                    else:
                        brightness = estimate_brightness(face_img)
                        sharpness = estimate_sharpness(face_img)
                        if brightness < args.min_brightness:
                            raw_label = "low_light"
                        elif sharpness < args.min_sharpness:
                            raw_label = "blurry_face"
                        else:
                            tensor = inferencer.preprocess_image(face_img)
                            predicted_idx, emotion_confidence, score_map, top_k_predictions = inferencer.predict_distribution_tensor(
                                tensor,
                                top_k=max(args.top_k_overlay, 5),
                            )
                            raw_label = inferencer.class_names[predicted_idx]
                            overridden_label, override_note = apply_neutral_bias(
                                score_map,
                                neutral_bias_threshold=args.neutral_bias_threshold,
                                neutral_bias_margin=args.neutral_bias_margin,
                                neutral_fallback_threshold=args.neutral_fallback_threshold,
                                neutral_fallback_max_confidence=args.neutral_fallback_max_confidence,
                            )
                            if overridden_label is not None:
                                raw_label = overridden_label
                                emotion_confidence = float(score_map.get("neutral", emotion_confidence))
                            effective_threshold = class_thresholds.get(raw_label, args.confidence_threshold)
                            if emotion_confidence < effective_threshold:
                                raw_label = "low_confidence"

                stabilized = stabilizer.update(raw_label, emotion_confidence)
                stabilized_row = {
                    "raw_label": stabilized.raw_label,
                    "stable_label": stabilized.stable_label,
                    "status": stabilized.status,
                    "raw_status": stabilized.raw_status,
                    "confidence": stabilized.confidence,
                    "top_k_predictions": top_k_predictions[: args.top_k_overlay],
                    "override_note": override_note,
                }

                writer.writerow(
                    {
                        "timestamp": time.time(),
                        "frame_index": frame_index,
                        "raw_label": stabilized.raw_label,
                        "stable_label": stabilized.stable_label,
                        "status": stabilized.status,
                        "raw_status": stabilized.raw_status,
                        "emotion_confidence": f"{stabilized.confidence:.6f}",
                        "face_confidence": f"{face_info.get('face_confidence', 0.0):.6f}",
                        "face_count": face_info.get("face_count", 0),
                        "bbox": json.dumps(face_info.get("bbox")),
                        "brightness": f"{brightness:.3f}",
                        "sharpness": f"{sharpness:.3f}",
                        "effective_threshold": f"{effective_threshold:.3f}",
                        "override_note": override_note,
                        "top_predictions": json.dumps(top_k_predictions[: args.top_k_overlay]),
                    }
                )

                draw_status(frame, face_info, stabilized_row)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    draw_text_with_outline(
                        frame,
                        f"face {face_info.get('face_confidence', 0.0):.2f}",
                        (x1, min(frame.shape[0] - 12, y2 + 20)),
                        (255, 255, 0),
                        scale=0.6,
                    )

                cv2.imshow("Stable Face Detection and Emotion Classification", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
        finally:
            capture.release()
            cv2.destroyAllWindows()

    print(json.dumps({"log_path": str(log_path.resolve()), "stabilizer": stabilizer.to_summary()}, indent=2))


if __name__ == "__main__":
    main()
