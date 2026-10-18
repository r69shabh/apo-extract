"""AppConfig.load records its source path so exports/runs can find the config later."""

from apo_extract.config import AppConfig


def test_load_records_config_path():
    cfg = AppConfig.load("configs/swimming.yaml")
    assert cfg.config_path == "configs/swimming.yaml"
    assert cfg.run_id == "swimming-srpbo-v1"
