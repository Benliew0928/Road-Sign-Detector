from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts/collect_no_left_or_right_turn_web_pilot.py"
)


def load_collector() -> ModuleType:
    spec = importlib.util.spec_from_file_location("no_left_or_right_turn_pilot", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def collector() -> ModuleType:
    return load_collector()


def valid_lead(candidate_id: str, source_group: str = "scene-a") -> dict[str, str]:
    return {
        "candidate_id": candidate_id,
        "class": "no_left_or_right_turn",
        "collect": "true",
        "source_type": "direct_image",
        "source_page_url": f"https://example.test/{candidate_id}",
        "direct_image_url": f"https://example.test/{candidate_id}.jpg",
        "archive_image_member": "",
        "creator_publisher": "test",
        "country": "Malaysia",
        "source_group": source_group,
        "rights_status": "rights_unverified",
        "stated_license": "",
        "bbox_xywh_normalized": "0.5 0.5 0.25 0.25",
        "target_class_evidence": "test evidence",
        "pre_reject_reason": "",
        "notes": "",
    }


def test_bbox_and_context_crop_stay_inside_original(collector: ModuleType) -> None:
    bbox = collector.parse_bbox("0.95 0.20 0.10 0.15", 1920, 1080)
    crop = collector.context_crop_box(bbox, 1920, 1080)

    assert 0 <= bbox[0] < bbox[2] <= 1920
    assert 0 <= bbox[1] < bbox[3] <= 1080
    assert 0 <= crop[0] <= bbox[0]
    assert 0 <= crop[1] <= bbox[1]
    assert bbox[2] <= crop[2] <= 1920
    assert bbox[3] <= crop[3] <= 1080


def test_collection_hard_cap_is_enforced(collector: ModuleType) -> None:
    leads = [valid_lead(f"candidate-{index}", f"scene-{index}") for index in range(51)]

    with pytest.raises(ValueError, match="download count 51 exceeds 50"):
        collector.validate_leads(leads)


def test_two_per_scene_cap_is_enforced(collector: ModuleType) -> None:
    leads = [valid_lead(f"candidate-{index}") for index in range(3)]

    with pytest.raises(ValueError, match="source groups exceed cap 2"):
        collector.validate_leads(leads)


def test_perceptual_hash_is_stable(collector: ModuleType) -> None:
    image = collector.Image.new("RGB", (128, 128), "white")
    collector.ImageDraw.Draw(image).ellipse((20, 20, 108, 108), outline="red", width=12)

    assert collector.perceptual_hash(image) == collector.perceptual_hash(image.copy())


def test_exclusion_manifest_loads_prior_downloaded_crops(
    collector: ModuleType, tmp_path: Path
) -> None:
    manifest = tmp_path / "prior.csv"
    manifest.write_text(
        "candidate_id,collection_status,crop_sha256,perceptual_hash\n"
        "prior-001,downloaded_and_decoded,abc123,0123456789abcdef\n"
        "prior-002,pre_rejected_not_downloaded,,\n",
        encoding="utf-8",
    )

    references = collector.excluded_batch_references([manifest])

    assert [(reference.candidate_id, reference.sha256) for reference in references] == [
        ("prior-001", "abc123")
    ]
