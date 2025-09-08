# Test API routes
import base64
import io
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..database import Base, get_db
from ..main import app
from ..models import Prediction


@pytest.fixture
def test_db():
    """Create a temporary test database."""
    # Create temporary database file
    db_fd, db_path_str = tempfile.mkstemp()
    db_path = Path(db_path_str)
    test_database_url = f"sqlite:///{db_path}"

    # Create engine and session
    engine = create_engine(test_database_url, connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Create tables
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        try:
            db = TestingSessionLocal()
            yield db
        finally:
            db.close()

    yield TestingSessionLocal, override_get_db

    # Cleanup
    os.close(db_fd)  # Still need os.close for file descriptor
    db_path.unlink()  # Use pathlib for file deletion


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def test_image_b64():
    """Create test base64 image."""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    return base64.b64encode(img_bytes).decode("utf-8")


class TestPredictEndpoint:
    """Test /predict endpoint functionality."""

    def test_predict_endpoint_success(self, client, test_db, test_image_b64):
        """Test successful prediction request."""
        # Override database dependency
        _, override_get_db = test_db
        app.dependency_overrides[get_db] = override_get_db

        # Mock the external API calls
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_response = MagicMock()
            mock_response.json.return_value = {
                "prediction": "phishing",
                "distance": 0.85,
                "target": "amazon",
            }
            mock_response.status_code = 200
            mock_post.return_value = mock_response

            # Make request
            response = client.post(
                "/predict", json={"image": test_image_b64, "url": "https://test.com"}
            )

            assert response.status_code == 200
            data = response.json()
            assert "results" in data

        # Clean up override
        app.dependency_overrides = {}

    def test_predict_endpoint_saves_to_database(self, client, test_db, test_image_b64):
        """Test that predictions are saved to database."""
        db_session_class, override_get_db = test_db
        app.dependency_overrides[get_db] = override_get_db

        # Mock the MODELS environment variable and reload routes
        with patch.dict(os.environ, {"MODELS": "visualphish"}):
            # Mock external API
            with patch("httpx.AsyncClient.post") as mock_post:
                mock_response = MagicMock()
                mock_response.json.return_value = {"prediction": "benign"}
                mock_response.status_code = 200
                mock_post.return_value = mock_response

                # Need to patch the URLS directly since MODELS is loaded at module import
                with patch("src.api.routes.URLS", ["http://visualphish:8888/predict"]):
                    # Make request
                    response = client.post(
                        "/predict",
                        json={"image": test_image_b64, "url": "https://example.com"},
                    )

                    assert response.status_code == 200

                    # Check database
                    db = db_session_class()
                    predictions = db.query(Prediction).all()
                    assert len(predictions) > 0

                    # Verify prediction data
                    saved_prediction = predictions[0]
                    assert saved_prediction.url == "https://example.com"
                    assert saved_prediction.img_hash is not None

                    db.close()

        app.dependency_overrides = {}

    def test_predict_endpoint_handles_partial_failure(
        self, client, test_db, test_image_b64
    ):
        """Test handling when some models fail."""
        _, override_get_db = test_db
        app.dependency_overrides[get_db] = override_get_db

        # Mock one successful, one failed response
        with patch("httpx.AsyncClient.post") as mock_post:

            def side_effect(*args, **kwargs):
                if "visualphish" in str(kwargs.get("url", "")):
                    mock_response = MagicMock()
                    mock_response.json.return_value = {"prediction": "phishing"}
                    mock_response.status_code = 200
                    return mock_response
                else:
                    # Simulate failure
                    raise Exception("Service unavailable")

            mock_post.side_effect = side_effect

            response = client.post(
                "/predict",
                json={"image": test_image_b64, "url": "https://partial-fail.com"},
            )

            # Should still return 200 with partial results
            assert response.status_code == 200
            data = response.json()
            assert "results" in data

        app.dependency_overrides = {}


class TestNewEndpoints:
    """Test new prediction history endpoints."""

    def test_get_predictions_endpoint(self, client, test_db):
        """Test GET /predictions endpoint."""
        db_session_class, override_get_db = test_db
        app.dependency_overrides[get_db] = override_get_db

        # Add test data
        db = db_session_class()
        test_predictions = [
            Prediction(img_hash="hash1", method="test1", url="url1", class_=1),
            Prediction(img_hash="hash2", method="test2", url="url2", class_=0),
        ]
        for pred in test_predictions:
            db.add(pred)
        db.commit()
        db.close()

        # Test endpoint
        response = client.get("/predictions")
        assert response.status_code == 200

        data = response.json()
        assert "predictions" in data
        assert "total" in data
        assert data["total"] == 2
        assert len(data["predictions"]) == 2

        app.dependency_overrides = {}

    def test_get_prediction_by_id_endpoint(self, client, test_db):
        """Test GET /prediction/{id} endpoint."""
        db_session_class, override_get_db = test_db
        app.dependency_overrides[get_db] = override_get_db

        # Add test prediction
        db = db_session_class()
        prediction = Prediction(
            img_hash="specific_hash",
            method="test_method",
            url="specific_url",
            class_=1,
            target="test_target",
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        test_id = prediction.id
        db.close()

        # Test endpoint
        response = client.get(f"/prediction/{test_id}")
        assert response.status_code == 200

        data = response.json()
        assert data["id"] == test_id
        assert data["img_hash"] == "specific_hash"
        assert data["method"] == "test_method"
        assert data["url"] == "specific_url"
        assert data["class_"] == 1
        assert data["target"] == "test_target"

        app.dependency_overrides = {}

    def test_get_prediction_not_found(self, client, test_db):
        """Test GET /prediction/{id} with non-existent ID."""
        _, override_get_db = test_db
        app.dependency_overrides[get_db] = override_get_db

        response = client.get("/prediction/99999")
        assert response.status_code == 404

        data = response.json()
        assert "detail" in data

        app.dependency_overrides = {}
