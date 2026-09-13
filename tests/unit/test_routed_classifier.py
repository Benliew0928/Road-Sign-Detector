from __future__ import annotations

import numpy as np

from roadsign_assist.classification.routed_backend import RoutedSignClassifier
from roadsign_assist.inference.models import ClassificationModel


class FakeClassifier:
    def __init__(self, name: str, result: ClassificationModel, *, available: bool = True) -> None:
        self._name = name
        self.result = result
        self._available = available
        self.warmed = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def available(self) -> bool:
        return self._available

    def warmup(self) -> bool:
        self.warmed = True
        return self.available

    def classify(self, crop: object) -> ClassificationModel:
        del crop
        return self.result


def result(label: str, *, accepted: bool, confidence: float = 0.9) -> ClassificationModel:
    return ClassificationModel(
        semantic_sign_id=label if accepted else "unknown_sign",
        confidence=confidence,
        accepted=accepted,
        model_name="fake",
        top_k=[(label, confidence)],
        unknown_score=1.0 - confidence,
    )


def route(primary: ClassificationModel, fallback: ClassificationModel) -> ClassificationModel:
    classifier = RoutedSignClassifier(
        FakeClassifier("primary", primary),
        FakeClassifier("fallback", fallback),
        repair_labels={"roadway_diverges", "divided_road_begins"},
    )
    return classifier.classify(np.zeros((8, 8, 3), dtype=np.uint8))


def test_agreement_uses_primary() -> None:
    selected = route(result("stop", accepted=True), result("stop", accepted=True, confidence=0.8))
    assert selected.confidence == 0.9
    assert "models_agree" in selected.model_name


def test_primary_repair_label_wins_disagreement() -> None:
    selected = route(
        result("divided_road_begins", accepted=True),
        result("roadway_diverges", accepted=True),
    )
    assert selected.semantic_sign_id == "divided_road_begins"
    assert "primary_repair_label" in selected.model_name


def test_primary_only_accepted_wins() -> None:
    selected = route(result("tractors", accepted=True), result("stop", accepted=False))
    assert selected.semantic_sign_id == "tractors"
    assert "primary_only_accepted" in selected.model_name


def test_fallback_wins_ordinary_disagreement() -> None:
    selected = route(result("tractors", accepted=True), result("pedestrian_crossing", accepted=True))
    assert selected.semantic_sign_id == "pedestrian_crossing"
    assert "retain_legacy_on_disagreement" in selected.model_name


def test_availability_and_warmup_require_both_models() -> None:
    primary = FakeClassifier("primary", result("stop", accepted=True))
    fallback = FakeClassifier("fallback", result("stop", accepted=True), available=False)
    classifier = RoutedSignClassifier(primary, fallback, repair_labels=set())
    assert not classifier.available
    assert not classifier.warmup()
    assert primary.warmed and fallback.warmed
