from roadsign_assist.paths import PROJECT_ROOT, project_path


def test_project_path_resolves_relative_paths() -> None:
    assert project_path("params.yaml") == PROJECT_ROOT / "params.yaml"
