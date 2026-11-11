"""CalHunger LLM arena: mixed LLM / trained-policy / scripted lineups.

Extends the paper's cross-play harness (``workshops/eval/
checkpoint_policy_crossplay.py``) with a third seat kind: an LLM queried
over a strict JSON wire protocol (adapted from the preliminary
``jsafrata/calhunger-llm`` arena), so one seat-balanced, seed-paired
evaluation can seat language models against trained PPO checkpoints and
the scripted heuristic ladder in the same games.
"""
from ._paths import bootstrap

bootstrap()
