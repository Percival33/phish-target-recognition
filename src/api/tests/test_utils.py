# Test utilities module
import base64
import io

import numpy as np
import pytest
from PIL import Image

from ..utils import compute_image_hash, decode_base64_image, parse_prediction_response


class TestComputeImageHash:
    """Test image hashing functionality."""

    def test_compute_image_hash_valid_base64(self):
        """Test computing hash with valid base64 image."""
        # Create a simple test image
        img = Image.new("RGB", (10, 10), color="red")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        b64_string = base64.b64encode(img_bytes).decode("utf-8")

        # Test hash computation
        hash_result = compute_image_hash(b64_string)
        assert isinstance(hash_result, str)
        assert len(hash_result) > 0

    def test_compute_image_hash_with_data_url_prefix(self):
        """Test computing hash with data URL prefix."""
        # Create test image
        img = Image.new("RGB", (10, 10), color="blue")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        b64_string = base64.b64encode(img_bytes).decode("utf-8")
        data_url = f"data:image/png;base64,{b64_string}"

        # Test hash computation
        hash_result = compute_image_hash(data_url)
        assert isinstance(hash_result, str)
        assert len(hash_result) > 0

    def test_compute_image_hash_invalid_base64(self):
        """Test computing hash with invalid base64 string."""
        with pytest.raises(ValueError, match="Failed to compute image hash"):
            compute_image_hash("invalid_base64_string")


class TestDecodeBase64Image:
    """Test base64 image decoding."""

    def test_decode_base64_image_valid(self):
        """Test decoding valid base64 image."""
        # Create test image
        img = Image.new("RGB", (5, 5), color="green")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_bytes = buffer.getvalue()
        b64_string = base64.b64encode(img_bytes).decode("utf-8")

        # Test decoding
        result = decode_base64_image(b64_string)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 5  # height
        assert result.shape[1] == 5  # width

    def test_decode_base64_image_invalid(self):
        """Test decoding invalid base64 string."""
        with pytest.raises(ValueError, match="Failed to decode base64 image"):
            decode_base64_image("invalid_string")


class TestParsePredictionResponse:
    """Test prediction response parsing."""

    def test_parse_visualphish_response(self):
        """Test parsing VisualPhish response."""
        response = {"prediction": "phishing", "distance": 0.85, "target": "amazon"}

        result = parse_prediction_response(response, "visualphish")

        assert result["method"] == "visualphish"
        assert result["class_"] == 1
        assert result["distance"] == 0.85
        assert result["target"] == "amazon"

    def test_parse_phishpedia_response(self):
        """Test parsing Phishpedia response."""
        response = {"result": "phish", "confidence": 0.92, "brand": "google"}

        result = parse_prediction_response(response, "phishpedia")

        assert result["method"] == "phishpedia"
        assert result["class_"] == 1
        assert result["distance"] == 0.92
        assert result["target"] == "google"

    def test_parse_benign_response(self):
        """Test parsing benign response."""
        response = {"prediction": "benign"}

        result = parse_prediction_response(response, "visualphish")

        assert result["method"] == "visualphish"
        assert result["class_"] == 0
        assert result["target"] is None
