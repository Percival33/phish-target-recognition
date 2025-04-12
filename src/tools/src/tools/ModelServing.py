from abc import ABC, abstractmethod
from fastapi import FastAPI, Form, UploadFile, File, Request
from contextlib import asynccontextmanager
import os
import base64
import numpy as np
import cv2


class ModelServing(ABC):
    def __init__(self, port=None):
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

    def register_routes(self):
        """Register routes for the FastAPI application"""

        @self.app.post("/predict")
        async def predict_route(request: Request):
            try:
                # Parse JSON request body
                data = await request.json()
                url = data.get("url")
                image_str = data.get("image")
                
                if not url or not image_str:
                    return {"error": "Missing required fields: url and/or image"}
                
                print(f"Received request with URL: {url}")
                print(f"Received base64 image string, length: {len(image_str) if image_str else 0}")
                
                # Try to decode the base64 string
                try:
                    # Decode base64 image
                    image_data = base64.b64decode(image_str)
                    print(f"Successfully decoded base64 to binary, length: {len(image_data)}")
                    
                    # Convert binary image to cv2 format
                    nparr = np.frombuffer(image_data, np.uint8)
                    image_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image_cv2 is None:
                        print("Failed to decode image to cv2 format")
                        return {"error": "Failed to decode image to OpenCV format"}
                    
                    print(f"Successfully converted to cv2 image with shape: {image_cv2.shape}")
                except Exception as e:
                    # Handle potential decoding or conversion errors
                    print(f"Error decoding base64 image: {str(e)}")
                    return {"error": f"Invalid base64 image data: {str(e)}"}
            except Exception as e:
                print(f"Error processing request: {str(e)}")
                return {"error": f"Invalid request data: {str(e)}"}

            prediction_data = {
                "url": url,
                "image": image_cv2,
            }
            try:
                result = await self.predict(prediction_data)
                return result
            except Exception as e:
                print(f"Error in prediction: {str(e)}")
                return {"error": f"Prediction failed: {str(e)}"}

    async def on_startup(self):
        """Startup logic (e.g., loading resources)"""
        print("Starting up...")

    async def on_shutdown(self):
        """Shutdown logic (e.g., cleaning up resources)"""
        print("Shutting down...")

    def run(self):
        """Run the FastAPI application"""
        import uvicorn

        uvicorn.run(self.app, host="0.0.0.0", port=self.port)


    @abstractmethod
    async def predict(self, data: dict):
        """Abstract method that subclasses must implement"""
        pass
