"""sys.path bootstrap shared by every llm_arena module and script.

The arena composes three code bases that are deliberately not installed
as packages:

- ``research/`` — the frozen game + RL stack (imported read-only, the
  same PYTHONPATH convention as ``workshops/eval``);
- ``workshops/eval/`` — the cross-play harness whose seat runners,
  rotation, and caching conventions we reuse (imported as the top-level
  module ``checkpoint_policy_crossplay``, the way those scripts import
  each other);
- ``workshops/`` — so ``llm_arena`` itself is importable from tests and
  scripts.

``bootstrap()`` is idempotent and safe to call from every entry point.
"""
from __future__ import annotations

import sys
from pathlib import Path

WORKSHOPS_ROOT = Path(__file__).resolve().parents[1]
RESEARCH_ROOT = WORKSHOPS_ROOT.parent / "research"
EVAL_DIR = WORKSHOPS_ROOT / "eval"
ARENA_ROOT = WORKSHOPS_ROOT / "llm_arena"


def bootstrap() -> None:
    # Insert in reverse priority: research must win name collisions
    # (``env``, ``agents``, ``eval``) exactly as in workshops/eval runs.
    for path in (WORKSHOPS_ROOT, EVAL_DIR, RESEARCH_ROOT):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
