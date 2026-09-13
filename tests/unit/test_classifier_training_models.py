from roadsign_assist.classification.training import _build_with_pretrained_weight_retry


def test_pretrained_model_build_retries_transient_download_failure(monkeypatch) -> None:
    attempts = 0

    def factory(*, weights: object) -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise OSError("connection reset")
        return "model"

    monkeypatch.setattr("roadsign_assist.classification.training.time.sleep", lambda _: None)

    result = _build_with_pretrained_weight_retry(
        factory,
        architecture="convnext_tiny",
        weights=object(),
        pretrained=True,
    )

    assert result == "model"
    assert attempts == 3


def test_non_pretrained_model_build_does_not_retry() -> None:
    calls: list[object] = []

    def factory(*, weights: object) -> str:
        calls.append(weights)
        return "model"

    result = _build_with_pretrained_weight_retry(
        factory,
        architecture="convnext_tiny",
        weights=object(),
        pretrained=False,
    )

    assert result == "model"
    assert calls == [None]
