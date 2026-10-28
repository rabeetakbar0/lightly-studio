from lightly_studio.api import features


def test_get_active_features() -> None:
    assert features._get_active_features() == [features.ONBOARDING_FEATURE]
