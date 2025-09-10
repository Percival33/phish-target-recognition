# ==============================================================================
# BUILD STAGE: Install dependencies and build environment
# ==============================================================================
FROM python:3.9.21-bookworm AS builder

# Install build dependencies and system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libglib2.0-dev \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:0.5.15 /uv /uvx /bin/

WORKDIR /build

# Copy dependency files
COPY uv.lock pyproject.toml ./

# Install dependencies with all development packages
# This creates a complete Python environment with all packages
RUN uv sync --frozen --no-cache

# Copy application source code
COPY main.py routes.py database.py models.py schemas.py utils.py __init__.py run.py app_factory.py ./
COPY logging.conf ./

# ==============================================================================
# PRODUCTION STAGE: Minimal runtime environment
# ==============================================================================
FROM python:3.9.21-slim-bookworm AS production

# Install only essential runtime dependencies for OpenCV and libraries
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1-mesa-glx \
    libgtk-3-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && rm -rf /tmp/* /var/tmp/*

# Create non-root user for security
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy the entire Python environment from builder stage
# This includes all installed packages and dependencies
COPY --from=builder /build/.venv /app/.venv
COPY --from=builder /build/*.py /app/
COPY --from=builder /build/logging.conf /app/

# Copy uv binary from builder for running the application
COPY --from=builder /bin/uv /bin/uv

# Create directories for data and logs with proper permissions
RUN mkdir -p /app/data /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Set environment variables for production
ENV PYTHONUNBUFFERED=1 \
    API_PORT=8888 \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app"

# Expose the port
EXPOSE 8888

# Health check using the non-root user
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8888/models || exit 1

# Run the application using the entry point script
CMD ["uv", "run", "python", "run.py"]
