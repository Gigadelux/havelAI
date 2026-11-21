from src.analytics.summary import summarize
from src.data.taxonomy import AttackType
from src.detection.base import Detection, Detector


def _d(attack: AttackType) -> Detection:
    return Detection(text="x", attack_type=attack, score=0.9)


def test_summarize_counts():
    s = summarize(
        [_d(AttackType.BENIGN), _d(AttackType.PROMPT_LEAKING), _d(AttackType.PROMPT_LEAKING)]
    )
    assert s.total == 3
    assert s.injections == 2
    assert s.by_attack_type == {"benign": 1, "prompt_leaking": 2}
    assert s.injection_rate == 2 / 3


def test_summarize_empty():
    s = summarize([])
    assert s.total == 0 and s.injection_rate == 0.0


def test_detector_protocol():
    class Dummy:
        def detect(self, text: str) -> Detection:
            return _d(AttackType.BENIGN)

    assert isinstance(Dummy(), Detector)
