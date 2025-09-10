# Utility functions for API
import base64
import io
from typing import Any, Dict

import numpy as np
from perception import hashers
from PIL import Image


def compute_image_hash(base64_image: str) -> str:
    """
    Compute perceptual hash of base64-encoded image.

    Args:
        base64_image: Base64-encoded image string

    Returns:
        Hexadecimal string representation of perceptual hash

    Raises:
        ValueError: If base64 string is invalid or image cannot be processed
    """
    try:
        # Remove data URL prefix if present
        if base64_image.startswith("data:image"):
            base64_image = base64_image.split(",")[1]

        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_image)

        # Open image with PIL
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Convert to numpy array
        image_array = np.array(image)

        # Compute perceptual hash using perception library
        hasher = hashers.PHash()
        phash_result = hasher.compute(image_array)

        if phash_result is None:
            raise ValueError("Hash computation returned None")

        return phash_result

    except Exception as e:
        raise ValueError(f"Failed to compute image hash: {str(e)}")


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    Decode base64 image string to numpy array.

    Args:
        base64_string: Base64-encoded image string

    Returns:
        Numpy array representation of the image

    Raises:
        ValueError: If base64 string is invalid
    """
    try:
        # Remove data URL prefix if present
        if base64_string.startswith("data:image"):
            base64_string = base64_string.split(",")[1]

        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)

        # Open image with PIL and convert to array
        image = Image.open(io.BytesIO(image_bytes))

        # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")

        return np.array(image)

    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")


def _parse_visualphish_response(response: Dict[Any, Any]) -> Dict[str, Any]:
    """Parse VisualPhish response format."""
    parsed = {}
    if "prediction" in response:
        parsed["class_"] = 1 if response["prediction"] == "phishing" else 0
    if "distance" in response:
        parsed["distance"] = float(response["distance"])
    if "target" in response:
        parsed["target"] = response["target"]
    return parsed


def _parse_phishpedia_response(response: Dict[Any, Any]) -> Dict[str, Any]:
    """Parse Phishpedia response format."""
    parsed = {}
    if "result" in response:
        parsed["class_"] = 1 if response["result"] == "phish" else 0
    if "confidence" in response:
        parsed["distance"] = float(response["confidence"])
    if "brand" in response:
        parsed["target"] = response["brand"]
    return parsed


def _parse_baseline_response(response: Dict[Any, Any]) -> Dict[str, Any]:
    """Parse Baseline response format."""
    parsed = {}
    if "class" in response:
        parsed["class_"] = int(response["class"])
    if "distance" in response:
        parsed["distance"] = float(response["distance"])
    if "target" in response:
        parsed["target"] = response["target"]
    return parsed


def parse_prediction_response(response: Dict[Any, Any], method: str) -> Dict[str, Any]:
    """
    Parse prediction response from different models into standardized format.

    Args:
        response: Raw response from prediction model
        method: Name of the prediction method/model

    Returns:
        Standardized prediction dictionary
    """
    # Default values
    parsed = {
        "method": method,
        "class_": 0,  # Default to benign
        "target": None,
        "distance": None,
    }

    # Method-specific parsers
    parsers = {
        "visualphish": _parse_visualphish_response,
        "phishpedia": _parse_phishpedia_response,
        "baseline": _parse_baseline_response,
    }

    # Get the appropriate parser and update the parsed response
    parser = parsers.get(method.lower())
    if parser:
        parsed.update(parser(response))

    return parsed
