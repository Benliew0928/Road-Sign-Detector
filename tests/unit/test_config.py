from roadsign_assist.config import load_application_settings, load_yaml


def test_application_settings_are_valid() -> None:
    settings = load_application_settings()
    assert settings.seed == 2513
    assert settings.runtime.max_frame_queue >= 1


def test_experimental_profile_keeps_classical_detection_fallback_enabled() -> None:
    settings = load_yaml("configs/inference/experimental.yaml")
    assert settings["detector"]["fallback_to_baseline"] is True
