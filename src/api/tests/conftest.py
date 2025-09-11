"""
Configuration for pytest.

This file adjusts the Python path to allow tests to run correctly
from the project root, ensuring that the `api` module can be found.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
