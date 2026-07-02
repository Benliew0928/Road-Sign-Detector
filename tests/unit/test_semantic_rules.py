from roadsign_assist.catalogue.repository import load_default_catalogue
from roadsign_assist.inference.models import BoundingBoxModel, OCRModel
from roadsign_assist.semantics.rules import SemanticRuleEngine
from roadsign_assist.tracking.iou_tracker import TrackState


def test_speed_limit_uses_ocr_parameter() -> None:
    engine = SemanticRuleEngine()
    meaning, severity, action = engine.action_for(
        "maximum_speed",
        0.95,
        OCRModel(text="50", confidence=0.99, numeric_value=50, unit="KM/H"),
    )
    assert meaning.en == "Maximum speed"
    assert severity == "critical"
    assert action.target_speed_kmh == 50


def test_speed_limit_advisory_is_human_readable_and_parameterized() -> None:
    engine = SemanticRuleEngine()
    meaning, _, action = engine.action_for(
        "maximum_speed",
        0.95,
        OCRModel(text="50", confidence=0.99, numeric_value=50, unit="KM/H"),
    )

    advisory = engine.advisory_for("maximum_speed", meaning, 0.95, action)

    assert advisory.headline.en == "Speed limit 50 km/h"
    assert "Keep your speed at or below the limit" in advisory.instruction.en
    assert advisory.safe_to_announce is True


def test_low_confidence_action_degrades_to_caution() -> None:
    engine = SemanticRuleEngine()
    meaning, _, action = engine.action_for("stop", 0.40, OCRModel())
    assert action.code == "UNKNOWN_CAUTION"

    advisory = engine.advisory_for("stop", meaning, 0.40, action)
    assert "not confident enough" in advisory.instruction.en
    assert advisory.safe_to_announce is False


def test_ocr_is_conditional_on_sign_semantics() -> None:
    engine = SemanticRuleEngine()
    assert engine.requires_ocr("maximum_speed") is True
    assert engine.requires_ocr("height_restriction") is True
    assert engine.requires_ocr("stop") is False
    assert engine.requires_ocr("unknown_sign") is False


def test_invalid_or_low_confidence_numeric_values_cannot_trigger_speed_action() -> None:
    engine = SemanticRuleEngine()
    for ocr in (
        OCRModel(text="300", confidence=0.99, numeric_value=300, unit="KM/H"),
        OCRModel(text="50", confidence=0.40, numeric_value=50, unit="KM/H"),
        OCRModel(text="50 T", confidence=0.99, numeric_value=50, unit="T"),
    ):
        _, severity, action = engine.action_for("maximum_speed", 0.99, ocr)
        assert severity == "caution"
        assert action.code == "UNKNOWN_CAUTION"


def test_dimension_units_are_required_and_centimetres_are_normalized() -> None:
    engine = SemanticRuleEngine()
    _, _, action = engine.action_for(
        "height_restriction",
        0.99,
        OCRModel(text="450 CM", confidence=0.99, numeric_value=450, unit="CM"),
    )
    assert action.code == "HEIGHT_RESTRICTION"
    assert action.restriction_value is not None
    assert abs(action.restriction_value - 4.5) < 1e-9
    assert action.restriction_unit == "M"

    _, _, unsafe = engine.action_for(
        "height_restriction",
        0.99,
        OCRModel(text="4.5", confidence=0.99, numeric_value=4.5),
    )
    assert unsafe.code == "UNKNOWN_CAUTION"


def test_every_catalogue_class_produces_an_advisory_deterministic_action() -> None:
    engine = SemanticRuleEngine()
    for definition in load_default_catalogue():
        first = engine.action_for(definition.semantic_sign_id, 1.0, OCRModel())
        second = engine.action_for(definition.semantic_sign_id, 1.0, OCRModel())
        assert first == second
        assert first[2].advisory_only is True


def test_duplicate_warning_cooldown_requires_stability() -> None:
    engine = SemanticRuleEngine(duplicate_warning_seconds=8.0)
    track = TrackState(
        track_id=1,
        bbox=BoundingBoxModel(x1=0, y1=0, x2=10, y2=10),
    )
    assert engine.should_announce(track, stable=False, now=10.0) is False
    assert engine.should_announce(track, stable=True, now=10.0) is True
    assert engine.should_announce(track, stable=True, now=15.0) is False
    assert engine.should_announce(track, stable=True, now=19.0) is True
