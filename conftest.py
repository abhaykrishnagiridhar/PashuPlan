import pytest

from pashuplan.data_gen import generate_farm


TRAIN_SEEDS = range(100, 108)


@pytest.fixture(scope="session")
def farm():
    return generate_farm(seed=42)


@pytest.fixture(scope="session")
def train_farms():
    """Randomized-timeline farms the models learn from. The hero farm (seed 42) is never in here."""
    return [generate_farm(seed=s, randomize_events=True) for s in TRAIN_SEEDS]


@pytest.fixture(scope="session")
def attribution(train_farms):
    from pashuplan.attribution import LossAttribution
    return LossAttribution.fit(train_farms)


@pytest.fixture(scope="session")
def models(attribution, risk_model):
    from pashuplan.models import Models
    return Models(attribution=attribution, risk=risk_model)


@pytest.fixture(scope="session")
def risk_model(train_farms):
    from pashuplan.risk_model import MastitisRiskModel
    return MastitisRiskModel.fit(train_farms)
