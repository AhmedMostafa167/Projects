from importlib import reload

import src.config as config_module


def test_default_config():
    assert config_module.settings.base_model_id == "answerdotai/ModernBERT-base"
    assert config_module.settings.learning_rate == 5e-5
    assert config_module.settings.num_epochs == 4


def test_label_smoothing_is_set(monkeypatch):
    monkeypatch.setenv("LABEL_SMOOTHING", "0.05")
    reload(config_module)
    assert config_module.settings.label_smoothing == 0.05
    monkeypatch.delenv("LABEL_SMOOTHING")
    reload(config_module)
