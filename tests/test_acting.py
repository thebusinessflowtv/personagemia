from app.acting import ActingDirector
from app.models import Emotion, Gesture


def test_opinion_line_gets_confident_direction():
    plan = ActingDirector().plan("Honestly, I think this is a terrible decision.")
    beat = plan.beats[0]
    assert beat.emotion == Emotion.confident
    assert beat.gesture == Gesture.lean_forward
    assert beat.eye_contact >= 0.9


def test_question_uses_open_palms():
    plan = ActingDirector().plan("Would you actually pay for this?")
    assert plan.beats[0].gesture == Gesture.open_palms
