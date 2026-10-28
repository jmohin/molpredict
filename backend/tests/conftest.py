import pytest

from molpredict_backend.predictors import load_predictors


@pytest.fixture(scope="session")
def predictors():
    """Loading ADMET-AI's ensembles takes seconds, so every test shares one set."""
    return load_predictors()
