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


def test_viewer_injects_model_aware_thinking_config():
    client = TestClient(app)
    response = client.get("/")
    html = response.text

    assert "window.IDEA_THINKING_MODEL_CONFIG" in html
    assert '"gemini-3.1-pro-preview"' in html
    assert '"default_level": "low"' in html
    assert '"supports_thinking_budget": false' in html
