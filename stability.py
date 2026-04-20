from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any


NON_EMOTION_STATES = {"no_face", "multiple_faces", "low_confidence", "small_face", "partial_face", "low_light", "blurry_face"}


@dataclass
class StabilizedEmotionResult:
    raw_label: str
    stable_label: str
    confidence: float
    status: str
    raw_status: str
    changed: bool
    history_size: int


class EmotionStabilizer:
    def __init__(
        self,
        *,
        smoothing_window: int = 5,
        debounce_frames: int = 3,
        flicker_hold_frames: int = 4,
        confidence_threshold: float = 0.6,
    ) -> None:
        self.smoothing_window = max(1, int(smoothing_window))
        self.debounce_frames = max(1, int(debounce_frames))
        self.flicker_hold_frames = max(0, int(flicker_hold_frames))
        self.confidence_threshold = float(confidence_threshold)
        self.history: deque[str] = deque(maxlen=self.smoothing_window)
        self.stable_label = "unknown"
        self._pending_label: str | None = None
        self._pending_count = 0
        self._hold_count = 0

    def update(self, raw_label: str, confidence: float) -> StabilizedEmotionResult:
        raw_status = "ok"
        normalized_label = raw_label
        if raw_label in NON_EMOTION_STATES:
            raw_status = raw_label
            normalized_label = "unknown"
        elif confidence < self.confidence_threshold:
            raw_status = "low_confidence"
            normalized_label = "unknown"

        if normalized_label != "unknown":
            self.history.append(normalized_label)
            majority_label = Counter(self.history).most_common(1)[0][0]
            self._hold_count = 0
        else:
            majority_label = self.stable_label if self.stable_label != "unknown" else "unknown"
            if self.flicker_hold_frames > 0 and self.stable_label != "unknown" and self._hold_count < self.flicker_hold_frames:
                self._hold_count += 1
            else:
                self.history.clear()
                self._pending_label = None
                self._pending_count = 0
                changed = self.stable_label != "unknown"
                self.stable_label = "unknown"
                return StabilizedEmotionResult(
                    raw_label=raw_label,
                    stable_label=self.stable_label,
                    confidence=confidence,
                    status=raw_status,
                    raw_status=raw_status,
                    changed=changed,
                    history_size=len(self.history),
                )

        changed = False
        if majority_label != self.stable_label:
            if self._pending_label == majority_label:
                self._pending_count += 1
            else:
                self._pending_label = majority_label
                self._pending_count = 1
            if self._pending_count >= self.debounce_frames:
                self.stable_label = majority_label
                self._pending_label = None
                self._pending_count = 0
                changed = True
        else:
            self._pending_label = None
            self._pending_count = 0

        status = "stable" if raw_status == "ok" else raw_status
        return StabilizedEmotionResult(
            raw_label=raw_label,
            stable_label=self.stable_label,
            confidence=confidence,
            status=status,
            raw_status=raw_status,
            changed=changed,
            history_size=len(self.history),
        )

    def to_summary(self) -> dict[str, Any]:
        return {
            "stable_label": self.stable_label,
            "history_size": len(self.history),
            "smoothing_window": self.smoothing_window,
            "debounce_frames": self.debounce_frames,
            "flicker_hold_frames": self.flicker_hold_frames,
            "confidence_threshold": self.confidence_threshold,
        }
