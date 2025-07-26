import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from OtrioAI.config import load_config, Config


def test_load_config_defaults(tmp_path):
    cfg_path = tmp_path / "conf.yaml"
    cfg_path.write_text("")
    cfg = load_config(str(cfg_path))
    assert isinstance(cfg, Config)
    assert cfg.buffer_capacity > 0
