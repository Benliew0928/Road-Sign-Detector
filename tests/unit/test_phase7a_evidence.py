"""Safety checks for the new audit; all pixel fixtures are synthetic."""
import importlib.util
from pathlib import Path

import pytest
from PIL import Image

SPEC = importlib.util.spec_from_file_location('phase7a', Path(__file__).parents[2] / 'scripts/prepare_phase7a_evidence.py')
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def record(pool, digest, group, label=None):
    return audit.normalized(pool, {'sha256': digest, 'id': digest[:8], 'group_id': group, 'split': 'train', 'label': label, 'training_eligible': True, 'image_path': 'data/train/sample.jpg'})


def test_exclusion_propagates_through_duplicate_group_aliases():
    rows = [record('candidate05', 'a'*64, 'site-A'), record('road_ledger', 'a'*64, 'alias-A'), record('candidate05', 'b'*64, 'alias-A')]
    result = audit.join_inventory(rows, set(), {'site-A'})
    assert len(result) == 2
    assert all(r['blocked_by_known_evaluation_metadata'] and not r['pixel_access_eligible'] for r in result)


def test_candidate03_corrected_overlay_wins_without_erasing_old_label():
    result = audit.join_inventory([record('classifier_v3', 'a'*64, 'a', 'roadway_diverges'), record('candidate03', 'a'*64, 'b', 'divided_road_begins')], set(), set())[0]
    assert result['label'] == 'divided_road_begins'
    assert result['label_conflict']
    assert {r['label'] for r in result['memberships']} == {'roadway_diverges', 'divided_road_begins'}
    assert not result['training_allowed']


@pytest.mark.parametrize('path', ['C:/unrelated/image.jpg', 'C:/MiniProject/../other/image.jpg', 'C:/MiniProject/data/test/image.jpg', 'C:/MiniProject/data/images/development/image.jpg', 'C:/MiniProject/data/SEALED/image.jpg', 'C:/MiniProject/data/validation/image.jpg'])
def test_reject_unsafe_pixel_paths(path):
    with pytest.raises(ValueError):
        audit.safe_pixel_path(path)


def test_older_training_metadata_does_not_grant_pixel_access():
    row = record('older_detector', 'a'*64, 'site')
    assert not row['pixel_access_eligible']


def test_render_synthetic_fixture_and_duplicate_attempt_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    monkeypatch.setattr(audit, 'OUT', tmp_path)
    path = tmp_path / 'train' / 'fixture.png'
    path.parent.mkdir()
    Image.new('RGB', (40, 30), 'yellow').save(path)
    row = record('candidate05', audit.sha(path), 'fixture')
    row.update(image_path=str(path), boxes=[[2, 3, 15, 20]])
    row = audit.join_inventory([row], set(), set())[0]
    monkeypatch.setattr(audit, 'preflight', lambda: [row])
    audit.render()
    result = audit.read_json(tmp_path / 'visual_audit_v1/RESULT.json')
    assert result['review_images'] == 1
    assert result['quarantined'] == 0
    assert not result['training_allowed']
    with pytest.raises(FileExistsError):
        audit.render()
