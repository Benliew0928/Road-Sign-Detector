from unittest.mock import Mock
from roadsign_api import dependencies


def test_close_up_defaults_to_deployed_model_session(monkeypatch):
    monkeypatch.delenv("ROADSIGN_CLOSE_UP_CONFIG", raising=False)
    engine = Mock()
    monkeypatch.setattr(dependencies, "get_engine", lambda: engine)
    dependencies.get_close_up_engine.cache_clear()
    try:
        assert dependencies.get_close_up_engine() is engine.new_session.return_value
        engine.new_session.assert_called_once_with()
    finally:
        dependencies.get_close_up_engine.cache_clear()


def test_close_up_explicit_override_is_preserved(monkeypatch):
    monkeypatch.setenv("ROADSIGN_CLOSE_UP_CONFIG", "custom.yaml")
    factory = Mock()
    monkeypatch.setattr(dependencies, "InferenceEngine", factory)
    dependencies.get_close_up_engine.cache_clear()
    try:
        dependencies.get_close_up_engine()
        factory.assert_called_once_with("custom.yaml")
    finally:
        dependencies.get_close_up_engine.cache_clear()
