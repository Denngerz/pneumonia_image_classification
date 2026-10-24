"""Every registered model returns (batch, 2) for 224×224 input. Random weights only — no downloads."""

from pathlib import Path

import pytest
import torch

from pneumonia.models import MODEL_NAMES, BaselineCNN, count_parameters, create_model
from pneumonia.utils import load_config, load_yaml

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "models"


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_model_outputs_two_classes(name):
    model = create_model(name, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 224, 224))
    assert out.shape == (2, 2)


def test_baseline_is_small():
    assert 300_000 < count_parameters(BaselineCNN()) < 500_000


def test_unknown_model_name():
    with pytest.raises(ValueError, match="Unknown model"):
        create_model("alexnet")


def test_every_model_has_a_config_that_only_sets_the_name():
    config_files = sorted(CONFIGS_DIR.glob("*.yaml"))
    assert {p.stem for p in config_files} == set(MODEL_NAMES)
    for path in config_files:
        # Fair comparison: a model config may not override shared settings.
        assert load_yaml(path) == {"model": {"name": path.stem}}
        assert load_config(path)["model"]["name"] == path.stem
