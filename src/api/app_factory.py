"""
Factory function for creating the FastAPI application.
This module handles both relative and absolute imports depending on the context.
"""

import logging.config
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def create_app(use_relative_imports=True):
    """
    Create and configure the FastAPI application.

    Args:
        use_relative_imports: If True, use relative imports (for local development).
                            If False, use absolute imports (for Docker).
    """
    # Configure logging
    logging_conf_path = Path(__file__).parent / "logging.conf"
    if logging_conf_path.exists():
        logging.config.fileConfig(str(logging_conf_path))
    else:
        # Fallback logging configuration if file doesn't exist
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
            handlers=[logging.StreamHandler(), logging.FileHandler("api.log")],
        )

    logger = logging.getLogger(__name__)

    # Create FastAPI app
    app = FastAPI(
        title="Phishing Detection API",
        description="API for phishing detection using multiple models with persistent storage",
        version="1.0.0",
    )

    # Import modules based on context
    if use_relative_imports:
        from .database import init_database
        from .routes import router
    else:
        import database
        import routes

        init_database = database.init_database
        router = routes.router

    # Initialize database
    try:
        init_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise

    # Add validation error handler
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ):
        """Handle validation errors with detailed logging."""
        logger = logging.getLogger(__name__)

        # Get the raw request body for debugging
        try:
            body = await request.body()
            content_type = request.headers.get("content-type", "unknown")
            logger.error(
                f"Validation error - URL: {request.url}, "
                f"Method: {request.method}, "
                f"Content-Type: {content_type}, "
                f"Body length: {len(body)}, "
                f"Body preview: {body[:200]}, "
                f"Errors: {exc.errors()}"
            )
        except Exception as e:
            logger.error(f"Error getting request details during validation error: {e}")

        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    app.include_router(router)
    return app
