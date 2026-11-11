"""Baseline agents for Nash Hunger."""
from functools import partial

from .base import Agent
from .random_agent import RandomAgent
from .noop_agent import NoopAgent
from .greedy_agent import GreedyAgent
from .old_greedy_agent import OldGreedyAgent
from .buffer_agent import BufferAgent
from .cadenced_scripted import CadencedScripted
from .scripted_v1 import ScriptedV1Agent
from .scripted_v2 import ScriptedV2Agent
from .scripted_v3 import ScriptedV3Agent
from .scripted_v4 import ScriptedV4Agent
from .scripted_v5 import ScriptedV5Agent

# The scripted baselines are always played through the cadence wrapper — firing
# rate is a property of the live game's bot loop, not of any one strategy, so
# all three are compared at the same cadence. "fast" matches what the eval
# harnesses (eval/pilot.py, eval/deep_eval.py, ...) already use.
_scripted = lambda inner: partial(  # noqa: E731
    CadencedScripted, cadence="fast", inner_factory=inner
)

# Convenience registry for the CLI runners
AGENT_REGISTRY = {
    "random": RandomAgent,
    "noop": NoopAgent,
    "greedy": GreedyAgent,
    "old_greedy": OldGreedyAgent,
    "buffer": BufferAgent,
    "scripted_v1": _scripted(ScriptedV1Agent),
    "scripted_v2": _scripted(ScriptedV2Agent),
    "scripted_v3": _scripted(ScriptedV3Agent),
    "scripted_v4": _scripted(ScriptedV4Agent),
    "scripted_v5": _scripted(ScriptedV5Agent),
}
