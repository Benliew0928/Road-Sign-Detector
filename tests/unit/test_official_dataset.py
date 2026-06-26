import csv

from roadsign_assist.datasets.official import (
    inventory_official_images,
    write_coursework_review_manifest,
)


def test_official_dataset_contains_exactly_84_valid_images() -> None:
    rows = inventory_official_images()
    assert len(rows) == 84
    assert all(row.width > 0 and row.height > 0 for row in rows)
    assert all(len(row.sha256) == 64 for row in rows)


def test_coursework_mapping_covers_every_candidate_id() -> None:
    rows = inventory_official_images()
    output = write_coursework_review_manifest()
    with output.open(encoding="utf-8") as handle:
        mapped = list(csv.DictReader(handle))
    assert len(mapped) == len(rows) == 84
    assert {row["coursework_id_candidate"] for row in mapped} == {
        image.coursework_id_candidate for image in rows
    }
