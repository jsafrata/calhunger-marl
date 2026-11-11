from __future__ import annotations

import importlib.util

import numpy as np
import pytest


@pytest.fixture
def fake_policy_module():
    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch is not installed")

    import torch
    from torch import nn
    from ray.rllib.core.columns import Columns

    class FakePolicyModule(nn.Module):
        actor_type = "mlp"

        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(torch.zeros(()))

        def forward_inference(self, batch):
            mask = batch[Columns.OBS]["action_mask"]
            logits = torch.full_like(mask, -1e30)
            logits = torch.where(mask > 0, torch.zeros_like(logits), logits)
            return {Columns.ACTION_DIST_INPUTS: logits}

    return FakePolicyModule()
