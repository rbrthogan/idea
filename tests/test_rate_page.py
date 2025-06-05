"""
Test the rate page functionality including evolution loading and data flow
"""

import pytest
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.viewer import app


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
        """Test that /api/generations returns empty array when no data is available"""
        with patch('idea.viewer.engine', None), \
             patch('idea.viewer.latest_evolution_data', None):
            response = self.client.get("/api/generations")
            assert response.status_code == 200
            data = response.json()
            assert data == []

    def test_api_generations_with_mock_data(self):
        """Test that /api/generations returns proper data structure"""
        mock_evolution_data = [
            [
                {"title": "Idea 1", "content": "Content 1"},
                {"title": "Idea 2", "content": "Content 2"}
            ],
            [
                {"title": "Idea 3", "content": "Content 3"},
                {"title": "Idea 4", "content": "Content 4"}
            ]
        ]

        with patch('idea.viewer.latest_evolution_data', mock_evolution_data):
            response = self.client.get("/api/generations")
            assert response.status_code == 200
            data = response.json()
            assert len(data) == 2
            assert len(data[0]) == 2
            assert len(data[1]) == 2
            assert data[0][0]["title"] == "Idea 1"
            assert data[0][0]["content"] == "Content 1"

    def test_api_evolutions_returns_list(self):
        """Test that /api/evolutions returns a list structure"""
        response = self.client.get("/api/evolutions")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_api_evolution_not_found(self):
        """Test that requesting non-existent evolution returns error"""
        response = self.client.get("/api/evolution/nonexistent")
        # The API currently returns 500 for missing files, not 404
        assert response.status_code == 500

    def test_submit_rating_missing_evolution(self):
        """Test that submitting rating for non-existent evolution returns error"""
        rating_data = {
            "idea_a_id": "test_id_a",
            "idea_b_id": "test_id_b",
            "outcome": "A",
            "evolution_id": "nonexistent"
        }
        response = self.client.post("/api/submit-rating", json=rating_data)
        # The API currently returns 500 for missing files, not 404
        assert response.status_code == 500


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

                # Load malformed evolution - should not crash
                response = self.client.get("/api/evolution/malformed_evolution")
                assert response.status_code == 200
                loaded_data = response.json()
                assert "data" in loaded_data
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)