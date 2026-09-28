import re

from app.models import Emotion, Gesture, PerformanceBeat, PerformancePlan


_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class ActingDirector:
    """Deterministic baseline director.

    This deliberately works without an LLM. Later we can replace/augment it with a model
    that returns the same PerformancePlan schema.
    """

    def plan(self, script: str) -> PerformancePlan:
        sentences = [s.strip() for s in _SENTENCE_RE.split(script.strip()) if s.strip()]
        beats = [self._direct(index, text) for index, text in enumerate(sentences)]
        return PerformancePlan(beats=beats)

    def _direct(self, index: int, text: str) -> PerformanceBeat:
        low = text.lower()
        emotion = Emotion.confident
        gesture = Gesture.explain
        energy = 0.68

        if "?" in text:
            emotion, gesture, energy = Emotion.skeptical, Gesture.open_palms, 0.74
        if any(token in low for token in ("ridiculous", "insane", "unbelievable", "crazy")):
            emotion, gesture, energy = Emotion.disbelief, Gesture.open_palms, 0.88
        if any(token in low for token in ("honestly", "i think", "my take", "in my opinion")):
            emotion, gesture, energy = Emotion.confident, Gesture.lean_forward, 0.78
        if any(token in low for token in ("serious", "lost", "dead", "failed", "bankrupt")):
            emotion, gesture, energy = Emotion.serious, Gesture.hands_together, 0.58
        if any(token in low for token in ("first", "second", "third", "three", "two reasons")):
            gesture = Gesture.count
        if any(token in low for token in ("look at", "this right here", "right there")):
            gesture = Gesture.point
        if "!" in text:
            energy = min(1.0, energy + 0.12)

        return PerformanceBeat(
            index=index,
            text=text,
            emotion=emotion,
            gesture=gesture,
            energy=energy,
            eye_contact=0.94,
            head_motion=min(0.65, 0.25 + energy * 0.35),
            brow_motion=min(0.75, 0.20 + energy * 0.45),
        )
