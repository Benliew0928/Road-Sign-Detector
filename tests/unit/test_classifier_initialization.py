from pathlib import Path

import pytest
import torch

from roadsign_assist.classification.initialization import initialize_classifier_weights


def test_initialization_loads_weights_without_optimizer_state(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2)
    path = tmp_path / "best.pt"
    target = {"weight": torch.ones(2, 2), "bias": torch.zeros(2)}
    torch.save(
        {
            "state_dict": target,
            "labels": ["left", "right"],
            "config": {"architecture": "toy", "image_size": 320},
            "optimizer_state_dict": {"must_not_be_loaded": True},
            "next_epoch": 90,
        },
        path,
    )
    report = initialize_classifier_weights(
        model, path, labels=["left", "right"], architecture="toy", image_size=320
    )
    assert torch.equal(model.weight, target["weight"])
    optimizer = torch.optim.AdamW(model.parameters())
    assert not optimizer.state
    assert report["mode"] == "weights_only_initialization_fresh_optimizer_and_schedule"


def test_initialization_expands_label_head_and_preserves_shared_rows(tmp_path: Path) -> None:
    model = torch.nn.Sequential()
    model.add_module("features", torch.nn.Linear(2, 2))
    model.add_module("classifier", torch.nn.Linear(2, 3))
    source = torch.nn.Sequential()
    source.add_module("features", torch.nn.Linear(2, 2))
    source.add_module("classifier", torch.nn.Linear(2, 2))
    with torch.no_grad():
        source.features.weight.fill_(2.0)
        source.features.bias.fill_(3.0)
        source.classifier.weight[0].fill_(4.0)
        source.classifier.weight[1].fill_(5.0)
        source.classifier.bias.copy_(torch.tensor([6.0, 7.0]))
    new_weight = model.classifier.weight[1].detach().clone()
    new_bias = model.classifier.bias[1].detach().clone()
    path = tmp_path / "best.pt"
    torch.save(
        {
            "state_dict": source.state_dict(),
            "labels": ["left", "right"],
            "config": {"architecture": "toy", "image_size": 320},
        },
        path,
    )

    report = initialize_classifier_weights(
        model,
        path,
        labels=["left", "new", "right"],
        architecture="toy",
        image_size=320,
        allow_label_expansion=True,
    )

    assert torch.equal(model.features.weight, source.features.weight)
    assert torch.equal(model.classifier.weight[0], source.classifier.weight[0])
    assert torch.equal(model.classifier.weight[2], source.classifier.weight[1])
    assert torch.equal(model.classifier.bias[[0, 2]], source.classifier.bias)
    assert torch.equal(model.classifier.weight[1], new_weight)
    assert torch.equal(model.classifier.bias[1], new_bias)
    assert report["mode"] == "label_aware_superset_expansion_shared_rows_copied"
    assert report["added_labels"] == ["new"]


@pytest.mark.parametrize(
    "labels,architecture,size",
    [
        (["right", "left"], "toy", 320),
        (["left", "right"], "other", 320),
        (["left", "right"], "toy", 224),
    ],
)
def test_initialization_rejects_semantic_or_model_mismatch(
    tmp_path: Path, labels: list[str], architecture: str, size: int
) -> None:
    model = torch.nn.Linear(2, 2)
    before = model.weight.detach().clone()
    path = tmp_path / "best.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "labels": ["left", "right"],
            "config": {"architecture": "toy", "image_size": 320},
        },
        path,
    )
    with pytest.raises(ValueError):
        initialize_classifier_weights(
            model, path, labels=labels, architecture=architecture, image_size=size
        )
    assert torch.equal(model.weight, before)
