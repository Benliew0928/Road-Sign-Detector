from __future__ import annotations

import time

from roadsign_assist.catalogue.models import (
    ActionCode,
    LocalizedText,
    ParameterType,
    Severity,
    SignCategory,
    SignDefinition,
)
from roadsign_assist.catalogue.repository import catalogue_by_id
from roadsign_assist.inference.models import ADASActionModel, OCRModel
from roadsign_assist.tracking.iou_tracker import TrackState

UNKNOWN_MEANING = LocalizedText(
    en="Unknown road sign",
    ms="Papan tanda tidak dikenali",
    zh="未知交通标志",
)


class SemanticRuleEngine:
    def __init__(
        self,
        *,
        critical_confidence: float = 0.88,
        normal_confidence: float = 0.75,
        duplicate_warning_seconds: float = 8.0,
    ) -> None:
        self.catalogue = catalogue_by_id()
        self.critical_confidence = critical_confidence
        self.normal_confidence = normal_confidence
        self.duplicate_warning_seconds = duplicate_warning_seconds

    def requires_ocr(self, semantic_sign_id: str) -> bool:
        definition = self.catalogue.get(semantic_sign_id)
        return bool(
            definition
            and (
                definition.category is SignCategory.TEXT
                or definition.parameter_type not in {ParameterType.NONE, ParameterType.DIRECTION}
            )
        )

    def resolve_label(
        self,
        classifier_label: str,
        classifier_confidence: float,
        ocr: OCRModel,
    ) -> tuple[str, float, list[str]]:
        evidence: list[str] = []
        label = classifier_label
        confidence = classifier_confidence
        if label in self.catalogue:
            evidence.append(f"classifier:{label}:{confidence:.3f}")
        else:
            label = "unknown_sign"
        if ocr.semantic_sign_id in self.catalogue and ocr.confidence >= 0.65:
            evidence.append(f"ocr_alias:{ocr.semantic_sign_id}:{ocr.confidence:.3f}")
            if label == "unknown_sign":
                label = ocr.semantic_sign_id
                confidence = ocr.confidence
            elif label == ocr.semantic_sign_id:
                confidence = min(1.0, 0.65 * confidence + 0.35 * ocr.confidence + 0.10)
        return label, confidence, evidence

    def action_for(
        self,
        semantic_sign_id: str,
        confidence: float,
        ocr: OCRModel,
    ) -> tuple[LocalizedText, Severity, ADASActionModel]:
        definition = self.catalogue.get(semantic_sign_id)
        if definition is None:
            return (
                UNKNOWN_MEANING,
                Severity.CAUTION,
                ADASActionModel(code=ActionCode.UNKNOWN_CAUTION),
            )
        threshold = (
            self.critical_confidence
            if definition.severity is Severity.CRITICAL
            else self.normal_confidence
        )
        if confidence < threshold:
            return (
                definition.names,
                Severity.CAUTION,
                ADASActionModel(code=ActionCode.UNKNOWN_CAUTION),
            )

        action_kwargs: dict[str, object] = {"code": definition.base_action}
        parameter = self._validated_parameter(definition, ocr)
        requires_value = definition.base_action in {
            ActionCode.SET_TARGET_SPEED,
            ActionCode.HEIGHT_RESTRICTION,
            ActionCode.WIDTH_RESTRICTION,
            ActionCode.WEIGHT_RESTRICTION,
        }
        if requires_value and parameter is None:
            return (
                definition.names,
                Severity.CAUTION,
                ADASActionModel(code=ActionCode.UNKNOWN_CAUTION),
            )
        value, unit = parameter if parameter is not None else (None, None)
        if definition.parameter_type is ParameterType.SPEED and value is not None:
            action_kwargs["target_speed_kmh"] = value
        elif (
            definition.parameter_type
            in {
                ParameterType.HEIGHT,
                ParameterType.WIDTH,
                ParameterType.WEIGHT,
                ParameterType.AXLE_WEIGHT,
            }
            and value is not None
        ):
            action_kwargs["restriction_value"] = value
            action_kwargs["restriction_unit"] = unit
        elif definition.parameter_type is ParameterType.DIRECTION:
            action_kwargs["direction"] = (
                str(definition.default_parameter) if definition.default_parameter else None
            )
        return definition.names, definition.severity, ADASActionModel.model_validate(action_kwargs)

    @staticmethod
    def _validated_parameter(
        definition: SignDefinition,
        ocr: OCRModel,
    ) -> tuple[float, str | None] | None:
        default_parameter = definition.default_parameter
        if isinstance(default_parameter, (int, float)):
            value = float(default_parameter)
        else:
            value = ocr.numeric_value if ocr.confidence >= 0.65 else None
        if value is None:
            return None

        unit = ocr.unit.upper() if ocr.unit else None
        if definition.parameter_type is ParameterType.SPEED:
            if unit not in {None, "KM/H"} or not 5.0 <= value <= 160.0:
                return None
            return value, "KM/H"
        if definition.parameter_type in {ParameterType.HEIGHT, ParameterType.WIDTH}:
            if unit == "CM":
                value /= 100.0
                unit = "M"
            if unit != "M" or not 0.5 <= value <= 10.0:
                return None
            return value, unit
        if definition.parameter_type in {
            ParameterType.WEIGHT,
            ParameterType.AXLE_WEIGHT,
        }:
            if unit != "T" or not 0.5 <= value <= 100.0:
                return None
            return value, unit
        if definition.parameter_type is ParameterType.DISTANCE:
            if unit not in {"M", "KM"} or not 1.0 <= value <= 100_000.0:
                return None
            return value, unit
        if definition.parameter_type is ParameterType.TIME:
            return value, unit
        return None

    def should_announce(self, track: TrackState, *, stable: bool, now: float | None = None) -> bool:
        if not stable:
            return False
        current = time.monotonic() if now is None else now
        if (
            track.last_announced_at is not None
            and current - track.last_announced_at < self.duplicate_warning_seconds
        ):
            return False
        track.last_announced_at = current
        return True
