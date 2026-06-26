from roadsign_assist.config import load_application_settings


def test_application_settings_are_valid() -> None:
    settings = load_application_settings()
    assert settings.seed == 2513
    assert settings.runtime.max_frame_queue >= 1
