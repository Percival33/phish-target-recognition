#!/usr/bin/env python3
"""
Entry point script for running the FastAPI application in Docker.
This script uses the shared app factory with absolute imports.
"""

import os
import sys
from pathlib import Path

import uvicorn

if __name__ == "__main__":
    # Add current directory to Python path for absolute imports
    current_dir = Path(__file__).parent
    if str(current_dir) not in sys.path:
        sys.path.insert(0, str(current_dir))

    # Import the shared app factory
    from app_factory import create_app

    # Create app using absolute imports for Docker
    app = create_app(use_relative_imports=False)
    PORT = int(os.getenv("API_PORT", 8888))

    # Run the application
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
