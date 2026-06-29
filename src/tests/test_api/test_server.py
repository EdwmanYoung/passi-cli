"""Tests for passi.api.server FastAPI stub."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from passi.api.server import create_app


@pytest.fixture
def client() -> TestClient:
    """Create a TestClient for the FastAPI app."""
    app = create_app()
    return TestClient(app)


def test_health_endpoint(client: TestClient) -> None:
    """/health returns ok status."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_endpoint(client: TestClient) -> None:
    """/ returns reserved-for-future-use message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "PassiAgent API" in data["message"]
