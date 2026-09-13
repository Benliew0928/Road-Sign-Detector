"""Guard checks only; no real pixels, model load, or inference."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('phase7a_assist', Path(__file__).parents[2] / 'scripts/run_phase7a_annotation_assist.py')
assist = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(assist)


@pytest.mark.parametrize('path', ['C:/other/a.jpg', 'C:/MiniProject/../other/a.jpg', 'C:/MiniProject/data/test/a.jpg', 'C:/MiniProject/data/development/a.jpg', 'C:/MiniProject/data/sealed/a.jpg'])
def test_reject_nontraining_paths(path):
    row = {'pool': 'road_ledger', 'pixel_access_eligible': True, 'training_allowed': False,
           'new_independent_evaluation_eligible': False, 'image_path': path}
    with pytest.raises(ValueError):
        assist.eligible_path(row)


def test_output_collision_refuses_before_loading_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(assist, 'OUTPUT', tmp_path)
    with pytest.raises(FileExistsError):
        assist.preflight()


def test_modified_pinned_input_refuses_before_model_or_pixels(tmp_path, monkeypatch):
    (tmp_path / 'ledger.csv').write_text('changed', encoding='utf-8')
    monkeypatch.setattr(assist, 'PACKAGE', tmp_path)
    monkeypatch.setattr(assist, 'OUTPUT', tmp_path / 'new_run')
    monkeypatch.setattr(assist, 'PINS', {'ledger.csv': '0' * 64})
    with pytest.raises(ValueError, match='Changed review input'):
        assist.preflight()


@pytest.mark.parametrize('box', [[0, 0, float('nan'), 10], [0, 0, float('inf'), 10], [10, 0, 0, 10], [-1, 0, 10, 10], [0, 0, 101, 10]])
def test_discard_invalid_proposals(box):
    assert not assist.valid_box(box, 100, 100)


def test_keep_valid_edge_box():
    assert assist.valid_box([0, 0, 100, 100], 100, 100)
