from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownArgumentType=false
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
import json
from pathlib import Path
from typing import Any

import numpy as np

from roadsign_assist.baseline.models import UInt8Image
from roadsign_assist.classification.embedding import EmbeddingGate
from roadsign_assist.classification.preprocessing import preprocess_classifier_numpy
from roadsign_assist.inference.models import ClassificationModel
from roadsign_assist.paths import project_path


class ONNXSignClassifier:
    def __init__(
        self,
        model_path: str | Path,
        labels_path: str | Path,
        *,
        calibration_path: str | Path | None = None,
        confidence_threshold: float = 0.72,
        image_size: int = 224,
        providers: list[str] | None = None,
    ) -> None:
        self.model_path = project_path(model_path)
        self.labels_path = project_path(labels_path)
        self.calibration_path = (
            project_path(calibration_path) if calibration_path is not None else None
        )
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.configured_image_size = image_size
        self.providers = providers
        self._session: Any | None = None
        self._labels: list[str] = []
        self._temperature = 1.0
        self._embedding_gate: EmbeddingGate | None = None
        self._active_providers: list[str] = []
        self._runtime_metadata: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return f"onnx:{self.model_path.name}"

    @property
    def available(self) -> bool:
        return self.model_path.exists() and self.labels_path.exists()

    @property
    def loaded(self) -> bool:
        return self._session is not None

    @property
    def active_providers(self) -> tuple[str, ...]:
        return tuple(self._active_providers)

    @property
    def runtime_metadata(self) -> dict[str, Any]:
        return dict(self._runtime_metadata)

    @property
    def embedding_gate_active(self) -> bool:
        return self._embedding_gate is not None

    def _load(self) -> Any:
        if self._session is None:
            if not self.available:
                raise FileNotFoundError("Classifier model or labels file is missing")
            try:
                import torch

                _ = torch.__version__
            except ImportError:
                pass
            import onnxruntime as ort

            self._labels = list(json.loads(self.labels_path.read_text(encoding="utf-8")))
            if self.calibration_path is not None and self.calibration_path.is_file():
                calibration = json.loads(self.calibration_path.read_text(encoding="utf-8"))
                self._runtime_metadata = calibration
                self._temperature = max(
                    0.05,
                    min(10.0, float(calibration.get("temperature", 1.0))),
                )
                if calibration.get("runtime_threshold_authoritative") is True:
                    self.confidence_threshold = max(
                        0.0,
                        min(1.0, float(calibration.get("confidence_threshold", self.confidence_threshold))),
                    )
                embedding_payload = calibration.get("embedding_gate")
                if isinstance(embedding_payload, dict):
                    self._embedding_gate = EmbeddingGate.from_payload(embedding_payload)
            providers = self.providers or [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
            installed = set(ort.get_available_providers())
            selected = [provider for provider in providers if provider in installed]
            if not selected:
                selected = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(str(self.model_path), providers=selected)
            self._active_providers = list(self._session.get_providers())
            input_shape = self._session.get_inputs()[0].shape
            if len(input_shape) == 4 and isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
                if input_shape[2] != input_shape[3]:
                    raise ValueError("Classifier ONNX input must be square")
                self.image_size = input_shape[2]
        return self._session

    def warmup(self) -> bool:
        if not self.available:
            return False
        session = self._load()
        input_name = session.get_inputs()[0].name
        sample = np.zeros(
            (1, 3, self.image_size, self.image_size),
            dtype=np.float32,
        )
        session.run(None, {input_name: sample})
        return True

    def _preprocess(self, crop: UInt8Image) -> np.ndarray[Any, np.dtype[np.float32]]:
        return preprocess_classifier_numpy(crop, self.image_size)

    def classify(self, crop: UInt8Image) -> ClassificationModel:
        session = self._load()
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: self._preprocess(crop)})
        logits = np.asarray(outputs[0])[0]
        calibrated_logits = logits / self._temperature
        exp = np.exp(calibrated_logits - np.max(calibrated_logits))
        probabilities = exp / np.sum(exp)
        order = np.argsort(probabilities)[::-1][:5]
        top_k = [(self._labels[int(index)], float(probabilities[index])) for index in order]
        label, confidence = top_k[0]
        accepted = confidence >= self.confidence_threshold
        embedding_distance: float | None = None
        nearest_prototype: str | None = None
        rejection_reasons: list[str] = []
        unknown_score = 1.0 - confidence
        if not accepted:
            rejection_reasons.append("confidence")
        if self._embedding_gate is not None:
            if len(outputs) < 2:
                raise RuntimeError(
                    "Embedding calibration is configured but the ONNX model has no embedding output"
                )
            decision = self._embedding_gate.decide(np.asarray(outputs[1])[0], label)
            embedding_distance = decision.distance
            nearest_prototype = decision.nearest_label
            rejection_reasons.extend(decision.reasons)
            accepted = accepted and decision.accepted
            unknown_score = max(unknown_score, decision.unknown_score)
        return ClassificationModel(
            semantic_sign_id=label if accepted else "unknown_sign",
            confidence=confidence,
            accepted=accepted,
            model_name=self.name,
            top_k=top_k,
            unknown_score=min(1.0, unknown_score),
            embedding_distance=embedding_distance,
            nearest_prototype=nearest_prototype,
            rejection_reasons=rejection_reasons,
        )
