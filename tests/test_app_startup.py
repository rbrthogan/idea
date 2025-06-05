"""Test that the FastAPI application starts successfully."""

import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add the project root to the path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.viewer import app


def test_app_startup():
    """Ensure the main FastAPI app can start and serve the root page."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
