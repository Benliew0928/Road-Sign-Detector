import json
from pathlib import Path


def test_emtd_class_35_is_camera_enforcement_after_p5_owner_review() -> None:
    mapping = json.loads(Path("configs/catalogue/emtd_class_mapping.v0.1.json").read_text())

    assert mapping["review_status"] == "p5_partial_owner_review"
    assert mapping["reviewer_2"] == "owner"
    assert mapping["classes"]["35"]["semantic_sign_id"] == "camera_enforcement"
    assert "not animal crossing" in mapping["classes"]["35"]["notes"]
    assert mapping["classes"]["59"]["semantic_sign_id"] == "animal_crossing"
    assert mapping["classes"]["50"]["semantic_sign_id"] == "staggered_junction"
    assert mapping["classes"]["56"]["semantic_sign_id"] == "side_road_right"
    assert mapping["classes"]["57"]["semantic_sign_id"] == "side_road_left"


def test_rebuilt_emtd_classification_uses_camera_enforcement_label() -> None:
    root = Path("data/processed/emtd_classification")
    labels = json.loads((root / "labels.json").read_text())
    metadata = json.loads((root / "dataset_metadata.json").read_text())

    assert "camera_enforcement" in labels
    assert "animal_crossing" in labels
    assert "staggered_junction" in labels
    assert "merge_left" not in labels
    assert "merge_right" not in labels
    assert metadata["class_mapping_status"] == "p5_partial_owner_review"
    assert metadata["classification_crops"] == 1064
    assert sum(
        split_counts.get("camera_enforcement", 0)
        for split_counts in metadata["classification_split_label_counts"].values()
    ) == 18
    assert sum(
        split_counts.get("animal_crossing", 0)
        for split_counts in metadata["classification_split_label_counts"].values()
    ) == 6


def test_p5_qc_report_has_no_catalogue_or_cross_split_duplicate_issues() -> None:
    report = json.loads(Path("outputs/audit/p5_label_qc_report.json").read_text())

    assert report["label_qc_status"] == "generated_no_seeded_corrections"
    assert report["labels_not_in_p2_catalogue"] == []
    assert report["exact_duplicate_hashes_cross_split"] == {}
    assert report["seeded_corrections"] == 0
