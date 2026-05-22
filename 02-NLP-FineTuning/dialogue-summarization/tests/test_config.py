from importlib import reload

import src.config as config_module


def test_default_lora_config():
    assert config_module.settings.lora_r == 16
    assert config_module.settings.lora_alpha == 32
    assert config_module.settings.base_model_id == "meta-llama/Llama-3.2-3B-Instruct"


def test_target_modules_parses_csv(monkeypatch):
    monkeypatch.setenv("LORA_TARGET_MODULES", "q_proj, k_proj , v_proj ")
    reload(config_module)
    assert config_module.settings.lora_target_modules_list == ["q_proj", "k_proj", "v_proj"]
    monkeypatch.delenv("LORA_TARGET_MODULES")
    reload(config_module)
