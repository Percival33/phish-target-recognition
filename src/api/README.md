# Phishing Detection API

A FastAPI-based service for phishing detection using multiple machine learning models with persistent storage and comprehensive logging.

## Features

- **Multi-model prediction aggregation**: Integrates multiple phishing detection models
- **Persistent storage**: SQLite database for storing prediction results
- **RESTful API**: Clean REST endpoints for accessing historical predictions
- **Comprehensive logging**: File and console logging with structured error tracking
- **Request tracing**: UUID-based request tracking across services
- **Error resilience**: Partial model failures don't break entire requests
- **Image hashing**: Perceptual hash computation for duplicate detection
- **Test coverage**: 83% coverage with comprehensive test suite

## API Endpoints

### POST `/predict`
Submit an image for phishing analysis across all configured models.

**Request Body:**
```json
{
    "image": "base64-encoded-image-data",
    "url": "https://example.com"
}
```

**Response:**
```json
{
    "results": [
        {
            "prediction": "phishing",
            "distance": 0.85,
            "target": "amazon"
        }
    ],
    "request_id": "uuid-string"
}
```

### GET `/predictions`
Retrieve all stored predictions with pagination support.

**Response:**
```json
{
    "predictions": [
        {
            "id": 1,
            "img_hash": "abc123...",
            "method": "visualphish",
            "url": "https://example.com",
            "class_": 1,
            "target": "amazon",
            "distance": 0.85,
            "created_at": "2025-01-26T10:00:00Z",
            "request_id": "uuid-string"
        }
    ],
    "total": 1
}
```

### GET `/prediction/{id}`
Get a specific prediction by ID.

**Response:**
```json
{
    "id": 1,
    "img_hash": "abc123...",
    "method": "visualphish",
    "url": "https://example.com",
    "class_": 1,
    "target": "amazon",
    "distance": 0.85,
    "created_at": "2025-01-26T10:00:00Z",
    "request_id": "uuid-string"
}
```

### GET `/models`
List all configured prediction models.

**Response:**
```json
{
    "models": ["visualphish", "phishpedia"]
}
```

## Database

- **SQLite database** at `database.db`
- **Automatic migration** on startup
- **Indexed** for performance on `img_hash` and `created_at`
- **Prediction storage** with full request metadata

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODELS` | Comma-separated list of models | `""` |
| `VP_PORT` | VisualPhish service port | `8888` |
| `PP_PORT` | Phishpedia service port | `8888` |
| `API_PORT` | API service port | `8888` |

### Example
```bash
export MODELS="visualphish,phishpedia"
export VP_PORT=8001
export PP_PORT=8002
export API_PORT=8888
```

## Development

### Installation
```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=.

# Lint code
uv run ruff check .
uv run ruff format .
```

### Running Locally
```bash
# Start the API server
uv run uvicorn main:app --host 0.0.0.0 --port 8888

# Or run directly
uv run python main.py
```

## Docker Deployment

### Multi-Stage Build Architecture

The Dockerfile uses a [multi-stage build](https://docs.docker.com/build/building/multi-stage/) to minimize production image size:

- **Build Stage** (`builder`): Full Python environment with build tools, compilers, and dev dependencies
- **Production Stage** (`production`): Minimal slim image with only runtime dependencies and application code

This approach reduces the final image size by ~60-70% while maintaining full functionality.

### Build Options

#### Production Build (Default - Optimized Size)
```bash
# Build optimized production image
docker build -t phishing-api .

# Or explicitly target production stage
docker build --target production -t phishing-api .
```

#### Development Build (Full Build Tools)
```bash
# Build with all development tools (larger image, useful for debugging)
docker build --target builder -t phishing-api:dev .
```

### Run Container

#### Basic Production Deployment
```bash
# Run optimized production container
docker run -p 8888:8888 phishing-api

# With environment variables
docker run -p 8888:8888 \
  -e MODELS="visualphish,phishpedia" \
  -e VP_PORT=8001 \
  -e PP_PORT=8002 \
  phishing-api
```

#### Development with Debugging
```bash
# Run development container with shell access
docker run -it --entrypoint /bin/bash phishing-api:dev
```

#### Persistent Storage
```bash
# With volume mounts for persistent data and logs
docker run -p 8888:8888 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  phishing-api
```

### Security Features

- **Non-root execution**: Application runs as `appuser` (UID 1001) for enhanced security
- **Minimal attack surface**: Production image contains only essential runtime libraries
- **Clean environment**: No build tools, package managers, or dev dependencies in production

### Health Check
The container includes a health check that verifies the `/models` endpoint:
```bash
docker ps  # Check health status
```

### Image Size Comparison
| Build Type | Approximate Size | Use Case |
|------------|------------------|----------|
| Single-stage (legacy) | ~1.2GB | Not recommended |
| Multi-stage production | ~400-500MB | Production deployment |
| Multi-stage builder | ~1.5GB | Development/debugging |

## Logging

- **File logging**: `api.log` (or configured path)
- **Console logging**: Structured output to stdout
- **Log levels**: DEBUG, INFO, WARNING, ERROR
- **Request tracking**: UUID-based tracing
- **Error context**: Full stack traces with `exc_info=True`

### Log Configuration
The service uses `logging.conf` for configuration. Key features:
- Separate file and console handlers
- Configurable log levels
- Structured format with timestamps

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   FastAPI App   │    │   Prediction     │    │   External      │
│                 │    │   Database       │    │   ML Services   │
│  ┌─────────────┐│    │                  │    │                 │
│  │  Endpoints  ││    │  ┌─────────────┐ │    │  ┌─────────────┐│
│  │             ││────│  │ SQLite DB   │ │    │  │ VisualPhish ││
│  │ - /predict  ││    │  │             │ │    │  │             ││
│  │ - /predictions││  │  │ Predictions │ │    │  │ Phishpedia  ││
│  │ - /models   ││    │  │ Table       │ │    │  │             ││
│  └─────────────┘│    │  └─────────────┘ │    │  └─────────────┘│
│                 │    │                  │    │                 │
│  ┌─────────────┐│    │  ┌─────────────┐ │    │                 │
│  │ Utils       ││    │  │ Indexes     │ │    │                 │
│  │             ││    │  │             │ │    │                 │
│  │ - Hash      ││    │  │ - img_hash  │ │    │                 │
│  │ - Parse     ││    │  │ - created_at│ │    │                 │
│  │ - Validate  ││    │  └─────────────┘ │    │                 │
│  └─────────────┘│    └──────────────────┘    └─────────────────┘
└─────────────────┘
```

## Testing

The project includes comprehensive tests with 83% coverage:

- **Unit tests**: Utils, database operations
- **Integration tests**: API endpoints, error handling
- **Database tests**: CRUD operations, transactions
- **Mock tests**: External service failures

### Test Categories
- `test_utils.py`: Image hashing, response parsing (8 tests)
- `test_database.py`: Database operations (5 tests)
- `test_routes.py`: API endpoints, error cases (6 tests)

## Production Considerations

### Performance
- **Connection pooling**: Consider implementing for high loads
- **Caching**: Add Redis for frequent predictions
- **Rate limiting**: Implement request throttling
- **Load balancing**: Use multiple instances behind a proxy

### Security
- **Input validation**: Comprehensive base64 and URL validation
- **Error handling**: No sensitive data in error responses
- **Request logging**: Full audit trail of predictions
- **CORS**: Configure for production domains

### Monitoring
- **Health checks**: Built-in Docker health check
- **Metrics**: Consider adding Prometheus metrics
- **Alerting**: Monitor prediction success rates
- **Database size**: Monitor SQLite database growth

## License

This project is part of a research initiative for phishing detection systems.
