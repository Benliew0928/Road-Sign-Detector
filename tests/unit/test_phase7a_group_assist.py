"""Synthetic geometry and admission guards; no real source pixels or inference."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('phase7a_group_assist', SCRIPTS / 'run_phase7a_group_assist.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def test_pair_shortlist_is_unique_deterministic_and_has_no_self_pairs():
    rows = [{'dhash64': f'{n:016x}'} for n in [0, 1, 3, 255, 65535]]
    pairs = mod.candidate_pairs(rows, neighbors=2)
    assert pairs == sorted(set(pairs))
    assert all(a < b for a, b in pairs)
    assert (0, 1) in pairs
    assert pairs == mod.candidate_pairs(rows, neighbors=2)


def test_blank_images_do_not_form_geometry_matches():
    blank = mod.features(Image.new('RGB', (300, 200), 'white'))
    assert mod.compare(blank, blank) is None


def test_translated_synthetic_texture_forms_review_only_proposal():
    import cv2
    rng = np.random.default_rng(2513)
    source = rng.integers(0, 256, (360, 480, 3), dtype=np.uint8)
    moved = cv2.warpAffine(source, np.float32([[1, 0, 15], [0, 1, 12]]), (480, 360))
    result = mod.compare(mod.features(Image.fromarray(source)), mod.features(Image.fromarray(moved)))
    assert result is not None
    assert result['homography_inliers'] >= 12
    assert result['status'] == 'visual_pair_proposal_only_requires_review'


def test_independent_synthetic_textures_do_not_match():
    rng = np.random.default_rng(77)
    a, b = [mod.features(Image.fromarray(rng.integers(0, 256, (300, 400, 3), dtype=np.uint8))) for _ in range(2)]
    assert mod.compare(a, b) is None


def test_preflight_refuses_existing_attempt_before_reading_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'OUTPUT', tmp_path)
    with pytest.raises(FileExistsError, match='no overwrite/retry'):
        mod.preflight()


def test_preflight_refuses_changed_pinned_input(tmp_path, monkeypatch):
    monkeypatch.setattr(mod, 'OUTPUT', tmp_path / 'not_created')
    monkeypatch.setattr(mod, 'PACKAGE', tmp_path)
    monkeypatch.setattr(mod, 'PINS', {'input.json': '0' * 64})
    (tmp_path / 'input.json').write_text('{}')
    with pytest.raises(ValueError, match='Changed input'):
        mod.preflight()
    assert not mod.OUTPUT.exists()
