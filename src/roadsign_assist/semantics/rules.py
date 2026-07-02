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
from roadsign_assist.inference.models import ADASActionModel, ADASAdvisoryModel, OCRModel
from roadsign_assist.tracking.iou_tracker import TrackState

UNKNOWN_MEANING = LocalizedText(
    en="Unknown road sign",
    ms="Papan tanda tidak dikenali",
    zh="未知交通标志",
)


def _localized(en: str, ms: str | None = None, zh: str | None = None) -> LocalizedText:
    return LocalizedText(en=en, ms=ms or en, zh=zh or en)


def _format_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.1f}".rstrip("0").rstrip(".")


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

    def advisory_for(
        self,
        semantic_sign_id: str,
        meaning: LocalizedText,
        confidence: float,
        action: ADASActionModel,
    ) -> ADASAdvisoryModel:
        if action.code is ActionCode.UNKNOWN_CAUTION:
            if semantic_sign_id == "unknown_sign" or semantic_sign_id not in self.catalogue:
                return ADASAdvisoryModel(
                    headline=UNKNOWN_MEANING,
                    instruction=_localized(
                        "A road sign was detected, but its meaning is uncertain. Keep monitoring the road.",
                        "Papan tanda dikesan, tetapi maksudnya belum pasti. Terus pantau jalan.",
                    ),
                    safe_to_announce=False,
                )
            return ADASAdvisoryModel(
                headline=meaning,
                instruction=_localized(
                    "This sign is not confident enough for a strong command. Slow down and verify it visually.",
                    "Keyakinan tanda ini belum cukup untuk arahan kuat. Perlahankan kenderaan dan sahkan secara visual.",
                ),
                safe_to_announce=False,
            )

        if action.code is ActionCode.SET_TARGET_SPEED and action.target_speed_kmh is not None:
            speed = _format_number(float(action.target_speed_kmh))
            return ADASAdvisoryModel(
                headline=_localized(
                    f"Speed limit {speed} km/h",
                    f"Had laju {speed} km/j",
                ),
                instruction=_localized(
                    f"This road has a speed limit of {speed} km/h. Keep your speed at or below the limit.",
                    f"Jalan ini mempunyai had laju {speed} km/j. Pastikan kelajuan tidak melebihi had.",
                ),
            )

        if (
            action.code
            in {
                ActionCode.HEIGHT_RESTRICTION,
                ActionCode.WIDTH_RESTRICTION,
                ActionCode.WEIGHT_RESTRICTION,
            }
            and action.restriction_value is not None
            and action.restriction_unit
        ):
            value = _format_number(float(action.restriction_value))
            unit = action.restriction_unit.lower()
            dimension = {
                ActionCode.HEIGHT_RESTRICTION: "height",
                ActionCode.WIDTH_RESTRICTION: "width",
                ActionCode.WEIGHT_RESTRICTION: "weight",
            }[action.code]
            return ADASAdvisoryModel(
                headline=_localized(f"{dimension.title()} limit {value} {unit}"),
                instruction=_localized(
                    f"Check the vehicle {dimension}. Do not continue if it exceeds {value} {unit}.",
                    f"Semak {dimension} kenderaan. Jangan teruskan jika melebihi {value} {unit}.",
                ),
            )

        fixed: dict[ActionCode, tuple[str, str]] = {
            ActionCode.STOP_REQUEST: (
                "Stop ahead",
                "Prepare to stop safely and check traffic before moving again.",
            ),
            ActionCode.YIELD: (
                "Give way",
                "Slow down and give way to traffic with priority.",
            ),
            ActionCode.REDUCE_SPEED: (
                "Reduce speed",
                "Slow down smoothly and scan the road ahead.",
            ),
            ActionCode.PROHIBIT_ENTRY: (
                "Do not enter",
                "Do not enter this road. Choose a permitted route.",
            ),
            ActionCode.PROHIBIT_LEFT_TURN: (
                "No left turn",
                "Do not turn left here. Continue until a permitted turn.",
            ),
            ActionCode.PROHIBIT_RIGHT_TURN: (
                "No right turn",
                "Do not turn right here. Continue until a permitted turn.",
            ),
            ActionCode.PROHIBIT_U_TURN: (
                "No U-turn",
                "Do not make a U-turn at this location.",
            ),
            ActionCode.PROHIBIT_DIRECTION: (
                "Movement prohibited",
                "Do not follow the prohibited direction shown by the sign.",
            ),
            ActionCode.PROHIBIT_LANE_CHANGE: (
                "No lane change",
                "Stay in your lane until lane changing is allowed.",
            ),
            ActionCode.PROHIBIT_OVERTAKING: (
                "No overtaking",
                "Do not overtake until the restriction ends.",
            ),
            ActionCode.PROHIBIT_VEHICLE: (
                "Vehicle restriction",
                "This vehicle type is restricted. Use an allowed route.",
            ),
            ActionCode.PROHIBIT_PARKING: (
                "No parking",
                "Do not park in this area.",
            ),
            ActionCode.PROHIBIT_STOPPING: (
                "No stopping",
                "Do not stop here unless required for safety.",
            ),
            ActionCode.PROHIBIT_HORN: (
                "No horn",
                "Avoid using the horn in this area.",
            ),
            ActionCode.KEEP_LEFT: (
                "Keep left",
                "Keep to the left side as directed.",
            ),
            ActionCode.KEEP_RIGHT: (
                "Keep right",
                "Keep to the right side as directed.",
            ),
            ActionCode.FOLLOW_DIRECTION: (
                "Follow direction",
                "Follow the direction shown by the sign.",
            ),
            ActionCode.SOUND_HORN: (
                "Sound horn",
                "Sound the horn if needed to warn others safely.",
            ),
            ActionCode.WATCH_PEDESTRIANS: (
                "Pedestrian crossing",
                "Slow down and watch for pedestrians crossing ahead.",
            ),
            ActionCode.WATCH_CHILDREN: (
                "Watch for children",
                "Slow down and be ready for children near the road.",
            ),
            ActionCode.WATCH_CYCLISTS: (
                "Watch for cyclists",
                "Give cyclists space and be ready to slow down.",
            ),
            ActionCode.WATCH_ANIMALS: (
                "Watch for animals",
                "Slow down and watch for animals near the road.",
            ),
            ActionCode.WATCH_TRAFFIC_SIGNAL: (
                "Traffic signal ahead",
                "Prepare to obey the traffic signal ahead.",
            ),
            ActionCode.WATCH_RAILWAY: (
                "Railway crossing",
                "Slow down and check for trains before crossing.",
            ),
            ActionCode.WATCH_ROAD_HAZARD: (
                "Road hazard ahead",
                "Slow down and prepare for a road hazard ahead.",
            ),
            ActionCode.INFORMATION_ONLY: (
                meaning.en,
                "Use this sign as information and continue monitoring the road.",
            ),
        }
        headline, instruction = fixed.get(
            action.code,
            (meaning.en, "Continue carefully and monitor the road ahead."),
        )
        return ADASAdvisoryModel(
            headline=_localized(headline),
            instruction=_localized(instruction),
            safe_to_announce=confidence >= self.normal_confidence,
        )

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
