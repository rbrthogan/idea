"""
Test the rate page functionality including evolution loading and data flow
"""

import pytest
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
from contextlib import contextmanager
from fastapi.testclient import TestClient

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.viewer import app, require_auth
from idea.auth import UserInfo


@contextmanager
def authenticated_user():
    """Temporarily override auth for endpoints that require a logged-in user."""
    async def mock_require_auth():
        return UserInfo(uid="test-user", email="test@example.com", is_admin=False)

    app.dependency_overrides[require_auth] = mock_require_auth
    try:
        yield
    finally:
        app.dependency_overrides.pop(require_auth, None)


class TestRatePageAPI:
    """Test the API endpoints used by the rate page"""

    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)

    def test_rate_page_loads(self):
        """Test that the rate page loads without errors"""
        response = self.client.get("/rate")
        assert response.status_code == 200
        assert "Idea Rater" in response.text
        assert "evolutionSelect" in response.text

    def test_api_generations_empty_when_no_data(self):
        """Test that /api/generations returns empty array when no data is available

        Note: This endpoint now requires authentication. With no auth token,
        it should return 401/403. We test this behavior here.
        """
        # Without authentication, the endpoint should reject the request
        response = self.client.get("/api/generations")
        assert response.status_code in [401, 403]

    def test_api_generations_with_mock_auth(self):
        """Test that /api/generations returns proper data structure with auth

        Since the endpoint now requires authentication and uses user-scoped state,
        we need to mock both the auth dependency and the user state.
        """
        from idea.auth import UserInfo
        from idea.user_state import UserEvolutionState, user_states
        from asyncio import Queue

        mock_user = UserInfo(uid="test_user_123", email="test@example.com", is_admin=False)
        mock_state = UserEvolutionState()
        mock_state.engine = None  # No engine yet
        mock_state.latest_data = [
            [
                {"title": "Idea 1", "content": "Content 1"},
                {"title": "Idea 2", "content": "Content 2"}
            ],
            [
                {"title": "Idea 3", "content": "Content 3"},
                {"title": "Idea 4", "content": "Content 4"}
            ]
        ]

        async def mock_require_auth():
            return mock_user

        async def mock_get_state(user_id):
            return mock_state

        from idea.viewer import app, require_auth
        from idea.user_state import user_states as real_user_states

        # Override the auth dependency
        app.dependency_overrides[require_auth] = mock_require_auth

        # Patch user_states.get
        with patch.object(real_user_states, 'get', mock_get_state):
            response = self.client.get("/api/generations")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert len(data[0]) == 2
            assert len(data[1]) == 2
            assert data[0][0]["title"] == "Idea 1"
            assert data[0][0]["content"] == "Content 1"

        # Clean up dependency override
        app.dependency_overrides.clear()

    def test_api_evolutions_returns_list(self):
        """Test that /api/evolutions returns a list structure"""
        with authenticated_user(), patch("idea.viewer.db.list_evolutions", new_callable=AsyncMock) as mock_list:
            mock_list.return_value = []
            response = self.client.get("/api/evolutions")
            assert response.status_code == 200
            data = response.json()
            assert isinstance(data, list)

    def test_api_evolution_not_found(self):
        """Test that requesting non-existent evolution returns error"""
        with authenticated_user(), patch("idea.viewer.db.get_evolution", new_callable=AsyncMock) as mock_get:
            mock_get.return_value = None
            response = self.client.get("/api/evolution/nonexistent")
            assert response.status_code == 404

    def test_submit_rating_missing_evolution(self):
        """Test that submitting rating for non-existent evolution returns error"""
        rating_data = {
            "idea_a_id": "test_id_a",
            "idea_b_id": "test_id_b",
            "outcome": "A",
            "evolution_id": "nonexistent"
        }
        response = self.client.post("/api/submit-rating", json=rating_data)
        assert response.status_code == 404

    def test_auto_rate_progress_defaults_when_unauthenticated(self):
        response = self.client.get("/api/auto-rate/progress")
        assert response.status_code == 200
        data = response.json()
        assert data["is_running"] is False
        assert data["status"] == "idle"

    def test_auto_rate_progress_returns_user_state(self):
        from idea.viewer import get_current_user
        from idea.auth import UserInfo
        from idea.user_state import UserEvolutionState, user_states as real_user_states

        mock_user = UserInfo(uid="autorate_user", email="autorate@example.com", is_admin=False)
        mock_state = UserEvolutionState()
        mock_state.autorating_status = {
            "is_running": True,
            "status": "in_progress",
            "status_message": "Running Swiss round 1/3...",
            "progress": 42,
            "completed_matches": 7,
            "total_matches": 16,
            "version": 3,
        }

        async def mock_get_current_user():
            return mock_user

        async def mock_get_state(_user_id):
            return mock_state

        app.dependency_overrides[get_current_user] = mock_get_current_user
        try:
            with patch.object(real_user_states, "get", mock_get_state):
                response = self.client.get("/api/auto-rate/progress")
                assert response.status_code == 200
                data = response.json()
                assert data["is_running"] is True
                assert data["status"] == "in_progress"
                assert data["completed_matches"] == 7
                assert data["total_matches"] == 16
                assert data["version"] == 3
        finally:
            app.dependency_overrides.pop(get_current_user, None)


class TestEvolutionDataFlow:
    """Test the evolution data loading and processing flow"""

    def setup_method(self):
        """Set up test client and temporary data directory"""
        self.client = TestClient(app)
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_evolution_save_and_load_flow(self):
        """Test the complete flow of saving and loading evolution data"""
        # Create mock evolution data
        evolution_data = {
            "history": [
                [
                    {
                        "id": "idea_1",
                        "title": "Test Idea 1",
                        "content": "This is test content 1",
                        "elo": 1500,
                        "ratings": {"auto": 1500, "manual": 1500}
                    },
                    {
                        "id": "idea_2",
                        "title": "Test Idea 2",
                        "content": "This is test content 2",
                        "elo": 1520,
                        "ratings": {"auto": 1520, "manual": 1480}
                    }
                ],
                [
                    {
                        "id": "idea_3",
                        "title": "Test Idea 3",
                        "content": "This is test content 3",
                        "elo": 1540,
                        "ratings": {"auto": 1540, "manual": 1510}
                    }
                ]
            ]
        }

        # Patch the DATA_DIR to use our temporary directory
        with patch('idea.viewer.DATA_DIR', self.temp_path):
            # Save evolution data
            save_data = {
                "data": evolution_data,
                "filename": "test_evolution.json"
            }
            response = self.client.post("/api/save-evolution", json=save_data)
            assert response.status_code == 200

            # Verify file was created
            file_path = self.temp_path / "test_evolution.json"
            assert file_path.exists()

            with authenticated_user(), \
                patch("idea.viewer.db.list_evolutions", new_callable=AsyncMock) as mock_list, \
                patch("idea.viewer.db.get_evolution", new_callable=AsyncMock) as mock_get:
                mock_list.return_value = [
                    {
                        "id": "test_evolution",
                        "updated_at": "2026-01-01T00:00:00Z",
                        "name": "test_evolution",
                    }
                ]
                mock_get.return_value = evolution_data

                # Test loading evolutions list
                response = self.client.get("/api/evolutions")
                assert response.status_code == 200
                evolutions = response.json()
                assert len(evolutions) >= 1
                assert any(e["id"] == "test_evolution" for e in evolutions)

                # Test loading specific evolution
                response = self.client.get("/api/evolution/test_evolution")
                assert response.status_code == 200
                loaded_data = response.json()
                assert "data" in loaded_data
                assert "history" in loaded_data["data"]
                assert len(loaded_data["data"]["history"]) == 2

    def test_evolution_data_structure_validation(self):
        """Test that evolution data has the expected structure for the rate page"""
        evolution_data = {
            "history": [
                [
                    {
                        "id": "idea_1",
                        "title": "Test Idea 1",
                        "content": "This is test content 1",
                        "elo": 1500
                    }
                ]
            ]
        }

        with patch('idea.viewer.DATA_DIR', self.temp_path):
            # Save evolution data
            save_data = {
                "data": evolution_data,
                "filename": "test_evolution_structure.json"
            }
            response = self.client.post("/api/save-evolution", json=save_data)
            assert response.status_code == 200

            with authenticated_user(), patch("idea.viewer.db.get_evolution", new_callable=AsyncMock) as mock_get:
                mock_get.return_value = evolution_data

                # Load and verify structure
                response = self.client.get("/api/evolution/test_evolution_structure")
                assert response.status_code == 200
                loaded_data = response.json()

                # Verify essential fields for rate page
                assert "id" in loaded_data
                assert "timestamp" in loaded_data
                assert "data" in loaded_data
                assert "history" in loaded_data["data"]

                # Verify ideas structure
                ideas = loaded_data["data"]["history"][0]
                assert len(ideas) == 1
                idea = ideas[0]
                assert "id" in idea
                assert "title" in idea
                assert "content" in idea


class TestRatePageEdgeCases:
    """Test edge cases and error conditions for the rate page"""

    def setup_method(self):
        """Set up test client"""
        self.client = TestClient(app)

    def test_empty_evolution_handling(self):
        """Test handling of evolution with no ideas"""
        evolution_data = {"history": []}

        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)

        try:
            with patch('idea.viewer.DATA_DIR', temp_path):
                # Save empty evolution
                save_data = {
                    "data": evolution_data,
                    "filename": "empty_evolution.json"
                }
                response = self.client.post("/api/save-evolution", json=save_data)
                assert response.status_code == 200

                with authenticated_user(), patch("idea.viewer.db.get_evolution", new_callable=AsyncMock) as mock_get:
                    mock_get.return_value = evolution_data
                    # Load empty evolution
                    response = self.client.get("/api/evolution/empty_evolution")
                    assert response.status_code == 200
                    loaded_data = response.json()
                    assert loaded_data["data"]["history"] == []
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_malformed_evolution_data(self):
        """Test handling of malformed evolution data"""
        # Test with missing history
        evolution_data = {"not_history": []}

        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)

        try:
            with patch('idea.viewer.DATA_DIR', temp_path):
                save_data = {
                    "data": evolution_data,
                    "filename": "malformed_evolution.json"
                }
                response = self.client.post("/api/save-evolution", json=save_data)
                assert response.status_code == 200

                with authenticated_user(), patch("idea.viewer.db.get_evolution", new_callable=AsyncMock) as mock_get:
                    mock_get.return_value = evolution_data
                    # Load malformed evolution - should not crash
                    response = self.client.get("/api/evolution/malformed_evolution")
                    assert response.status_code == 200
                    loaded_data = response.json()
                    assert "data" in loaded_data
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
