import asyncio
import base64
import logging
import os
import uuid

import httpx
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from database import get_db
from models import Prediction
from schemas import (
    PredictionListResponse,
    PredictionResponse,
    PredictRequest,
)
from utils import compute_image_hash, parse_prediction_response

logger = logging.getLogger(__name__)

MODELS = os.getenv("MODELS", "").split(",") if os.getenv("MODELS") else []
SERVICE_PORTS = {
    "visualphish": os.getenv("VP_PORT", 8888),
    "phishpedia": os.getenv("PP_PORT", 8888),
}
URLS = [f"http://{model}:{SERVICE_PORTS.get(model, 8888)}/predict" for model in MODELS]

router = APIRouter()


async def fetch_data(
    client: httpx.AsyncClient,
    url: str,
    image: str,
    url_param: str,
    db: Session,
    request_id: str,
):
    try:
        # Create JSON payload with exact structure: image first, then url
        json_data = {"image": image, "url": url_param}

        print(f"Fetching {url} with URL param: {url_param}")
        print(f"Image data type: {type(image)}, length: {len(image)}")

        response = await client.post(url, json=json_data)

        # Debug response
        if response.status_code != 200:
            logger.error(
                f"Error response from {url}: {response.status_code} - {response.text[:1000]}"
            )

        response.raise_for_status()
        result = response.json()
        logger.info(f"Successfully received response from {url}")

        # Extract method name from URL
        method = url.split("//")[1].split(":")[0] if "//" in url else "unknown"

        # Add method to the result for the frontend
        if isinstance(result, dict):
            result["method"] = method

        # Save prediction to database
        try:
            img_hash = compute_image_hash(image)
            parsed_result = parse_prediction_response(result, method)

            prediction = Prediction(
                img_hash=img_hash,
                method=parsed_result["method"],
                url=url_param,
                class_=parsed_result["class_"],
                target=parsed_result.get("target"),
                distance=parsed_result.get("distance"),
                request_id=request_id,
            )

            db.add(prediction)
            db.commit()
            logger.info(
                f"Saved prediction to database: {prediction.id} for method {method}"
            )

        except Exception as e:
            logger.error(f"Error saving prediction to database: {e}", exc_info=True)
            # Don't fail the request if database save fails
            try:
                db.rollback()
            except Exception as rollback_error:
                logger.error(f"Error rolling back transaction: {rollback_error}")

        return result

    except (httpx.HTTPError, KeyError, ValueError) as e:
        logger.error(f"Error fetching from {url}: {str(e)}", exc_info=True)
        # Return None for failed requests instead of raising exception
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching from {url}: {str(e)}", exc_info=True)
        return None


@router.get("/models")
async def models_list():
    return {"models": MODELS}


@router.post("/predict")
async def predict(request: PredictRequest, db: Session = Depends(get_db)):
    try:
        # Validate base64 string by attempting to decode it
        try:
            # Test if this is valid base64
            base64.b64decode(request.image)
        except Exception as e:
            logger.warning(f"Invalid base64 encoding: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {e}")

        # Generate request ID for grouping predictions
        request_id = str(uuid.uuid4())
        logger.info(f"Starting prediction request {request_id} for URL: {request.url}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                fetch_data(
                    client, model_url, request.image, request.url, db, request_id
                )
                for model_url in URLS
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out None results (failed requests)
        successful_results = [r for r in results if r is not None]
        failed_count = len(results) - len(successful_results)

        if failed_count > 0:
            logger.warning(
                f"Request {request_id}: {failed_count} out of {len(results)} models failed"
            )

        logger.info(
            f"Completed prediction request {request_id} with {len(successful_results)} successful results"
        )
        return {"results": successful_results, "request_id": request_id}

    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error in predict endpoint: {e}")


@router.get("/predictions", response_model=PredictionListResponse)
async def get_predictions(db: Session = Depends(get_db)):
    """Get all predictions."""
    try:
        predictions = db.query(Prediction).order_by(Prediction.created_at.desc()).all()
        return PredictionListResponse(
            predictions=[PredictionResponse.model_validate(p) for p in predictions],
            total=len(predictions),
        )
    except Exception as e:
        logger.error(f"Error getting predictions: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error getting predictions: {e}")


@router.get("/predictions/{request_id}", response_model=PredictionListResponse)
async def get_predictions_by_request_id(request_id: str, db: Session = Depends(get_db)):
    """Get all predictions for a specific request ID."""
    try:
        predictions = (
            db.query(Prediction)
            .filter(Prediction.request_id == request_id)
            .order_by(Prediction.created_at.desc())
            .all()
        )
        if not predictions:
            raise HTTPException(
                status_code=404, detail="Predictions not found for this request ID"
            )

        return PredictionListResponse(
            predictions=[PredictionResponse.model_validate(p) for p in predictions],
            total=len(predictions),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error getting predictions for request_id {request_id}: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Error getting predictions: {e}")
