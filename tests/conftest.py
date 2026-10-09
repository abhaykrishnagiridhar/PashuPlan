import pytest

from pashuplan.data_gen import generate_farm


@pytest.fixture(scope="session")
def farm():
    return generate_farm(seed=42)
