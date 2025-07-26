import os
import sys
import json

os.environ["MPLBACKEND"] = "Agg"

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.logger import log_metrics, plot_metrics


def test_log_metrics_and_plot(tmp_path):
    log_path = tmp_path / "metrics.log"
    img_path = tmp_path / "metrics.png"

    log_metrics(str(log_path), step=1, loss=0.5, win_rate=0.1)
    log_metrics(str(log_path), step=2, loss=0.4, win_rate=0.2)

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2
    data = [json.loads(l) for l in lines]
    assert data[0] == {"step": 1, "loss": 0.5, "win_rate": 0.1}
    assert data[1] == {"step": 2, "loss": 0.4, "win_rate": 0.2}

    plot_metrics(str(log_path), str(img_path))
    assert img_path.exists() and img_path.stat().st_size > 0
