from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import os
import base64
import numpy as np
import cv2
from typing import Optional


class PredictRequest(BaseModel):
    url: str
    image: str


class PredictResponse(BaseModel):
    """Unified response model for all model serving classes"""

    url: str = Field(description="The URL that was analyzed")
    class_: int = Field(
        alias="class", description="Classification result: 0 for benign, 1 for phishing"
    )
    target: str = Field(description="Predicted target/brand name")
    confidence: Optional[float] = Field(
        None, description="Confidence score (used by Phishpedia)"
    )
    distance: Optional[float] = Field(
        None, description="Distance metric (used by VisualPhish and Baseline)"
    )

    class Config:
        populate_by_name = True
        json_encoders = {float: lambda v: v if v != float("inf") else None}


class ModelServing(ABC):
    def __init__(self, port=None) -> None:
        self.port = port if port is not None else int(os.getenv("PORT", 8888))
        self.app = FastAPI()

        # Set up lifespan context
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self.on_startup()
            yield
            await self.on_shutdown()

        self.app.router.lifespan_context = lifespan

        # Register routes - this lets subclasses define their own implementations
        self.register_routes()

    def register_routes(self) -> None:
        """Register routes for the FastAPI application"""

        @self.app.post("/predict", response_model=PredictResponse)
        async def predict_route(request_data: PredictRequest) -> PredictResponse:
            try:
                url = request_data.url
                image_str = request_data.image

                print(f"Received request with URL: {url}")
                print(
                    f"Received base64 image string, length: {len(image_str) if image_str else 0}"
                )

                # Try to decode the base64 string
                try:
                    # Decode base64 image
                    image_data = base64.b64decode(image_str)
                    print(
                        f"Successfully decoded base64 to binary, length: {len(image_data)}"
                    )

                    # Convert binary image to cv2 format
                    nparr = np.frombuffer(image_data, np.uint8)
                    image_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if image_cv2 is None:
                        print("Failed to decode image to cv2 format")
                        raise HTTPException(
                            status_code=400,
                            detail="Failed to decode image to OpenCV format",
                        )

                    print(
                        f"Successfully converted to cv2 image with shape: {image_cv2.shape}"
                    )
                except Exception as e:
                    # Handle potential decoding or conversion errors
                    print(f"Error decoding base64 image: {str(e)}")
                    raise HTTPException(
                        status_code=400, detail=f"Invalid base64 image data: {str(e)}"
                    )
            except Exception as e:
                print(f"Error processing request: {str(e)}")
                raise HTTPException(
                    status_code=400, detail=f"Invalid request data: {str(e)}"
                )

            prediction_data = {
                "url": url,
                "image": image_cv2,
            }
            try:
                result = await self.predict(prediction_data)
                # The result should already be a PredictResponse from the subclass implementation
                return result
            except Exception as e:
                print(f"Error in prediction: {str(e)}")
                raise HTTPException(
                    status_code=500, detail=f"Prediction failed: {str(e)}"
                )

    async def on_startup(self) -> None:
        """Startup logic (e.g., loading resources)"""
        print("Starting up...")

    async def on_shutdown(self) -> None:
        """Shutdown logic (e.g., cleaning up resources)"""
        print("Shutting down...")

    def run(self) -> None:
        """Run the FastAPI application"""
        import uvicorn

        uvicorn.run(self.app, host="0.0.0.0", port=self.port)

    @abstractmethod
    async def predict(self, data: dict) -> PredictResponse:
        """Abstract method that subclasses must implement"""
        pass
