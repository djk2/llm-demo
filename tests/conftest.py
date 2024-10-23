import pytest
from fastapi.testclient import TestClient

from llm_demo.app import app


@pytest.fixture
def client():
    yield TestClient(app)
