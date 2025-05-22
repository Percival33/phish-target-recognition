# Baseline Phishing Detection Service

A baseline phishing detection service that uses perceptual hashing (via the perception library) and FAISS vector similarity search to detect potential phishing attempts.

## Features

- Fast perceptual hash computation using the perception library
- Efficient similarity search using FAISS vector database
- Configurable distance threshold for phishing detection
- Target mapping support for brand name normalization
- FastAPI-based REST API interface

## Configuration

The service can be configured using environment variables or command line arguments:

- `PORT` - Service port (default: 8888)
- `DISTANCE_THRESHOLD` - Similarity threshold for phishing detection (default: 1.0)

## File Structure

- `BaselineServing.py` - Main service implementation
- `BaselineEmbedder.py` - Core embedding and search functionality
- `config/target_mappings.json` - Brand name normalization mappings
- `index/faiss_index.idx` - Pre-built FAISS index (mounted at runtime)

## API

### POST /predict

Endpoint for phishing detection predictions.

Request body:
```json
{
    "url": "string",
    "image": "base64 encoded image"
}
```

Response:
```json
{
    "url": "string",
    "class": "int (0 or 1)",
    "target": "string (brand name)",
    "confidence": "float",
    "baseline_distance": "float"
}
```

## Docker

The service is containerized and can be run using Docker:

```bash
docker build -t baseline-service .
docker run -p 8888:8888 \
    -v ./index:/code/index \
    -v ./config:/code/config \
    -e DISTANCE_THRESHOLD=1.0 \
    baseline-service
```
