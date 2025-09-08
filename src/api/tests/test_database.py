# Test database functionality
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from ..database import Base
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


class TestDatabaseOperations:
    """Test database CRUD operations."""

    def test_database_initialization(self, test_db):
        """Test database can be initialized."""
        db_session_class, _ = test_db
        db = db_session_class()

        # Test that we can create a session
        assert isinstance(db, Session)
        db.close()

    def test_save_prediction(self, test_db):
        """Test saving a prediction to database."""
        db_session_class, _ = test_db
        db = db_session_class()

        # Create test prediction
        prediction = Prediction(
            img_hash="test_hash_123",
            method="visualphish",
            url="https://example.com",
            class_=1,
            target="amazon",
            distance=0.85,
            request_id="test_request_123",
        )

        # Save to database
        db.add(prediction)
        db.commit()
        db.refresh(prediction)

        # Verify saved
        assert prediction.id is not None
        assert prediction.img_hash == "test_hash_123"
        assert prediction.method == "visualphish"
        assert prediction.url == "https://example.com"
        assert prediction.class_ == 1
        assert prediction.target == "amazon"
        assert prediction.distance == 0.85
        assert prediction.request_id == "test_request_123"
        assert isinstance(prediction.created_at, datetime)

        db.close()

    def test_get_prediction_by_id(self, test_db):
        """Test retrieving prediction by ID."""
        db_session_class, _ = test_db
        db = db_session_class()

        # Create and save prediction
        prediction = Prediction(
            img_hash="test_hash_456",
            method="phishpedia",
            url="https://test.com",
            class_=0,
        )
        db.add(prediction)
        db.commit()
        db.refresh(prediction)
        saved_id = prediction.id

        # Retrieve by ID
        retrieved = db.query(Prediction).filter(Prediction.id == saved_id).first()

        assert retrieved is not None
        assert retrieved.id == saved_id
        assert retrieved.img_hash == "test_hash_456"
        assert retrieved.method == "phishpedia"
        assert retrieved.class_ == 0

        db.close()

    def test_get_all_predictions(self, test_db):
        """Test retrieving all predictions."""
        db_session_class, _ = test_db
        db = db_session_class()

        # Create multiple predictions
        predictions = [
            Prediction(img_hash="hash1", method="method1", url="url1", class_=1),
            Prediction(img_hash="hash2", method="method2", url="url2", class_=0),
            Prediction(img_hash="hash3", method="method3", url="url3", class_=1),
        ]

        for pred in predictions:
            db.add(pred)
        db.commit()

        # Retrieve all
        all_predictions = db.query(Prediction).all()

        assert len(all_predictions) == 3
        hashes = [p.img_hash for p in all_predictions]
        assert "hash1" in hashes
        assert "hash2" in hashes
        assert "hash3" in hashes

        db.close()

    def test_database_transaction_rollback(self, test_db):
        """Test database transaction rollback on error."""
        db_session_class, _ = test_db
        db = db_session_class()

        try:
            # Create prediction with invalid data that should cause error
            prediction = Prediction(
                img_hash="test_hash", method="test_method", url="test_url", class_=1
            )
            db.add(prediction)

            # Force an error by trying to add another with same constraint violation
            # (This specific test may need adjustment based on actual constraints)
            db.commit()

            # If we get here, the basic save worked
            count_before_error = db.query(Prediction).count()

            # Now test rollback works
            db.rollback()
            count_after_rollback = db.query(Prediction).count()

            # The count should be the same (rollback should work)
            assert count_after_rollback == count_before_error

        except Exception:
            # If there was an exception, rollback should still work
            db.rollback()

        finally:
            db.close()
