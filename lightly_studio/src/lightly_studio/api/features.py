"""Global list of the active features."""

from __future__ import annotations

ONBOARDING_FEATURE = "onboarding"


def _get_active_features() -> list[str]:
    """Build the list of features the running app has switched on."""
    return [ONBOARDING_FEATURE]


lightly_studio_active_features: list[str] = _get_active_features()
