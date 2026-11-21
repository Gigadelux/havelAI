import re

from src.data.taxonomy import (
    ATTACK_CRITERIA,
    ID2LABEL,
    INTENT_CRITERIA,
    LABEL2ID,
    LABELS,
    AttackType,
)


def test_every_attack_except_benign_has_a_criterion():
    assert set(ATTACK_CRITERIA) == {t for t in AttackType if t is not AttackType.BENIGN}
    assert AttackType.BENIGN not in ATTACK_CRITERIA


def test_labels_are_unique_snake_case():
    assert len(set(LABELS)) == len(LABELS)
    assert all(re.fullmatch(r"[a-z]+(_[a-z]+)*", label) for label in LABELS)
    assert all(re.fullmatch(r"[a-z]+(_[a-z]+)*", k) for k in INTENT_CRITERIA)


def test_label_ids_round_trip():
    assert all(ID2LABEL[LABEL2ID[label]] == label for label in LABELS)
    assert LABELS[0] == "benign"


def test_criteria_are_non_empty_sentences():
    assert all(text.strip().endswith(".") for text in ATTACK_CRITERIA.values())
