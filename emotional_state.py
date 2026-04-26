from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any


EMOTIONAL_STATE_LABELS = ("calm", "positive", "stressed", "frustrated", "overwhelmed")


@dataclass
class EmotionalStateResult:
    label: str
    confidence: float
    explanation: str
    dominant_emotions: list[dict[str, float]]
    changed: bool
    history_size: int


class EmotionalStateTracker:
    def __init__(
        self,
        *,
        window_size: int = 90,
        persistence_frames: int = 10,
    ) -> None:
        self.window_size = max(1, int(window_size))
        self.persistence_frames = max(1, int(persistence_frames))
        self.history: deque[dict[str, float]] = deque(maxlen=self.window_size)
        self.state_label = "unknown"
        self._pending_label: str | None = None
        self._pending_count = 0

    def update(self, score_map: dict[str, float] | None) -> EmotionalStateResult:
        if not score_map:
            changed = self.state_label != "unknown"
            self.state_label = "unknown"
            self.history.clear()
            self._pending_label = None
            self._pending_count = 0
            return EmotionalStateResult(
                label=self.state_label,
                confidence=0.0,
                explanation="no reliable emotion distribution",
                dominant_emotions=[],
                changed=changed,
                history_size=0,
            )

        normalized = {label: float(score_map.get(label, 0.0)) for label in score_map}
        self.history.append(normalized)
        averages = self._average_scores()
        volatility = self._estimate_volatility()
        proposed_label, confidence, explanation = self._classify(averages, volatility)
        dominant_emotions = self._dominant_emotions(averages)

        changed = False
        if proposed_label != self.state_label:
            if self._pending_label == proposed_label:
                self._pending_count += 1
            else:
                self._pending_label = proposed_label
                self._pending_count = 1
            if self._pending_count >= self.persistence_frames:
                self.state_label = proposed_label
                self._pending_label = None
                self._pending_count = 0
                changed = True
        else:
            self._pending_label = None
            self._pending_count = 0

        return EmotionalStateResult(
            label=self.state_label,
            confidence=confidence,
            explanation=explanation,
            dominant_emotions=dominant_emotions,
            changed=changed,
            history_size=len(self.history),
        )

    def _average_scores(self) -> dict[str, float]:
        labels = set()
        for item in self.history:
            labels.update(item.keys())
        if not labels:
            return {}
        return {
            label: sum(frame_scores.get(label, 0.0) for frame_scores in self.history) / len(self.history)
            for label in labels
        }

    def _estimate_volatility(self) -> float:
        if len(self.history) < 2:
            return 0.0
        dominant_labels = [
            max(frame_scores.items(), key=lambda item: item[1])[0]
            for frame_scores in self.history
            if frame_scores
        ]
        if not dominant_labels:
            return 0.0
        dominant_switches = sum(
            1 for previous, current in zip(dominant_labels, dominant_labels[1:]) if previous != current
        )
        return dominant_switches / max(1, len(dominant_labels) - 1)

    def _classify(self, averages: dict[str, float], volatility: float) -> tuple[str, float, str]:
        neutral = averages.get("neutral", 0.0)
        happy = averages.get("happy", 0.0)
        anger = averages.get("anger", 0.0)
        sad = averages.get("sad", 0.0)
        fear = averages.get("fear", 0.0)
        surprise = averages.get("surprise", 0.0)

        calm_score = neutral - 0.25 * volatility
        positive_score = happy + 0.45 * neutral
        stressed_score = fear + 0.8 * surprise - 0.35 * neutral
        frustrated_score = anger + 0.8 * sad
        overwhelmed_score = fear + sad + 0.8 * surprise

        candidates = {
            "calm": calm_score,
            "positive": positive_score,
            "stressed": stressed_score,
            "frustrated": frustrated_score,
            "overwhelmed": overwhelmed_score,
        }
        label, confidence = max(candidates.items(), key=lambda item: item[1])
        confidence = max(0.0, min(1.0, float(confidence)))

        if label == "calm":
            explanation = f"neutral dominant with low volatility ({volatility:.2f})"
        elif label == "positive":
            explanation = "happy and neutral remain dominant"
        elif label == "stressed":
            explanation = "fear/surprise trend is rising while neutral stays low"
        elif label == "frustrated":
            explanation = "anger and sad remain consistently elevated"
        else:
            explanation = "fear, sad, and surprise remain elevated together"
        return label, confidence, explanation

    def _dominant_emotions(self, averages: dict[str, float]) -> list[dict[str, float]]:
        return [
            {"class_name": label, "confidence": float(score)}
            for label, score in sorted(averages.items(), key=lambda item: item[1], reverse=True)[:3]
        ]

    def to_summary(self) -> dict[str, Any]:
        return {
            "state_label": self.state_label,
            "history_size": len(self.history),
            "window_size": self.window_size,
            "persistence_frames": self.persistence_frames,
        }
