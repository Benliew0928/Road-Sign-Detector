from __future__ import annotations

from collections.abc import Set
from dataclasses import dataclass

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.classification.base import SignClassifier
from roadsign_assist.inference.models import ClassificationModel


@dataclass(frozen=True)
class RoutedClassifierDecision:
    selected_model: str
    reason: str


class RoutedSignClassifier:
    """Route between a repair classifier and a retained fallback.

    This backend is intentionally configuration-agnostic. Merely constructing it
    does not alter the active runtime; callers must explicitly supply both frozen
    classifiers and the repair-label allowlist.
    """

    def __init__(
        self,
        primary: SignClassifier,
        fallback: SignClassifier,
        *,
        repair_labels: Set[str],
        name: str = "routed_classifier",
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.repair_labels = frozenset(repair_labels)
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def available(self) -> bool:
        return self.primary.available and self.fallback.available

    @property
    def loaded(self) -> bool:
        return bool(getattr(self.primary, "loaded", True) and getattr(self.fallback, "loaded", True))

    @property
    def active_providers(self) -> tuple[str, ...]:
        providers: list[str] = []
        for member in (self.primary, self.fallback):
            for provider in getattr(member, "active_providers", ()):
                if provider not in providers:
                    providers.append(provider)
        return tuple(providers)

    @property
    def runtime_metadata(self) -> dict[str, object]:
        return {
            "release_status": "routed_candidate_not_promoted",
            "primary": dict(getattr(self.primary, "runtime_metadata", {})),
            "fallback": dict(getattr(self.fallback, "runtime_metadata", {})),
        }

    def warmup(self) -> bool:
        primary_ready = self.primary.warmup()
        fallback_ready = self.fallback.warmup()
        return primary_ready and fallback_ready

    @staticmethod
    def _raw_label(result: ClassificationModel) -> str:
        if result.top_k:
            return result.top_k[0][0]
        return result.semantic_sign_id

    def decide(
        self,
        primary_result: ClassificationModel,
        fallback_result: ClassificationModel,
    ) -> RoutedClassifierDecision:
        primary_label = self._raw_label(primary_result)
        fallback_label = self._raw_label(fallback_result)
        if primary_label == fallback_label:
            return RoutedClassifierDecision("primary", "models_agree")
        if primary_label in self.repair_labels:
            return RoutedClassifierDecision("primary", "primary_repair_label")
        if primary_result.accepted and not fallback_result.accepted:
            return RoutedClassifierDecision("primary", "primary_only_accepted")
        return RoutedClassifierDecision("fallback", "retain_legacy_on_disagreement")

    def classify(self, crop: UInt8Image) -> ClassificationModel:
        primary_result = self.primary.classify(crop)
        fallback_result = self.fallback.classify(crop)
        decision = self.decide(primary_result, fallback_result)
        selected = primary_result if decision.selected_model == "primary" else fallback_result
        return selected.model_copy(update={"model_name": f"{self.name}:{decision.selected_model}:{decision.reason}"})
