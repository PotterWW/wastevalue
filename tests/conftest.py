import os

os.environ["FAKE_MODEL"] = "1"
os.environ.pop("UNCERTAIN_THRESHOLD", None)

import pytest
from fastapi.testclient import TestClient

from server.main import app


@pytest.fixture(scope="session")
def client():
    return TestClient(app)
