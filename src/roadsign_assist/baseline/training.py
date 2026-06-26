from __future__ import annotations

# pyright: reportArgumentType=false, reportUnknownArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ClassifierName = Literal["svm", "random_forest"]


@dataclass(frozen=True)
class TrainingMetrics:
    model: str
    feature_set: str
    accuracy: float
    macro_f1: float
    train_count: int
    test_count: int


def build_classifier(name: ClassifierName, seed: int = 2513) -> object:
    if name == "svm":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    SVC(  # pyright: ignore[reportArgumentType]
                        C=10.0,
                        kernel="rbf",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported classifier: {name}")


def train_and_evaluate(
    *,
    classifier_name: ClassifierName,
    feature_set: str,
    x_train: NDArray[np.float32],
    y_train: NDArray[np.str_],
    x_test: NDArray[np.float32],
    y_test: NDArray[np.str_],
    model_path: Path,
    metrics_path: Path,
) -> TrainingMetrics:
    if not len(x_train) or not len(x_test):
        raise ValueError("Training and test data must both be non-empty")
    model = build_classifier(classifier_name)
    model.fit(x_train, y_train)  # type: ignore[attr-defined]
    predictions = model.predict(x_test)  # type: ignore[attr-defined]
    metrics = TrainingMetrics(
        model=classifier_name,
        feature_set=feature_set,
        accuracy=float(accuracy_score(y_test, predictions)),
        macro_f1=float(f1_score(y_test, predictions, average="macro")),
        train_count=len(x_train),
        test_count=len(x_test),
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    report = {
        **metrics.__dict__,
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,  # pyright: ignore[reportArgumentType]
        ),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return metrics
