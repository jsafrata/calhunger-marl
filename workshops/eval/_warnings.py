"""Selective warning silencing for the training/eval scripts.

We deliberately do NOT do a blanket `warnings.filterwarnings('ignore')`.
That would hide real problems (torch runtime warnings, numpy overflow,
etc.). Instead we silence only the specific Ray deprecation warnings
that the training team has already reviewed and accepted for this
codebase.
"""
from __future__ import annotations

import warnings


def silence_ray_deprecations() -> None:
    """Silence the four known Ray 2.53 deprecation warnings that clutter
    every training-driver run. Everything else stays visible."""
    # RayDeprecationWarning: UnifiedLogger / JsonLogger / CSVLogger /
    # TBXLogger interfaces will be removed in Ray 2.7. RLlib still uses
    # them internally; nothing on our end to change.
    for msg in (
        r"`UnifiedLogger`.*",
        r"The `JsonLogger.*",
        r"The `CSVLogger.*",
        r"The `TBXLogger.*",
        r"`RLModule\(config=.*",
        # FutureWarning about `RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO`.
        r".*RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO.*",
    ):
        warnings.filterwarnings("ignore", message=msg)
