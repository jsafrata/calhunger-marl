"""Run checkpoint_policy_crossplay against fine-tuned checkpoints.

The fine-tuned checkpoints' saved Algorithm config references FINETUNE_ENV_NAME, so any
loader that reconstructs the Algorithm needs that env registered first. This wrapper
registers it, then execs the peer-crossplay harness unchanged with the same CLI.

Usage: python hpc/finetune_vs_ladder/eval_crossplay_wrapper.py <crossplay args...>
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ray.tune.registry import register_env  # noqa: E402

from hpc.finetune_vs_ladder.scripted_opponent_env import (  # noqa: E402
    FINETUNE_ENV_NAME,
    ScriptedOpponentEnv,
)

register_env(FINETUNE_ENV_NAME, lambda cfg: ScriptedOpponentEnv(cfg))

HARNESS = Path(__file__).resolve().parents[1] / "best_config_peer_crossplay" / "checkpoint_policy_crossplay.py"
sys.argv = [str(HARNESS)] + sys.argv[1:]
runpy.run_path(str(HARNESS), run_name="__main__")
