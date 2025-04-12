from abc import ABC, abstractmethod
from fastapi import FastAPI, Request
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
            # Get JSON data from request
            try:
                data = await request.json()
                url = data.get("url")
                base64_image = data.get("image")
                
                if not url or not base64_image:
                    return {"error": "Missing required fields: url and image"}
                
                # Decode base64 image
                try:
                    image_data = base64.b64decode(base64_image)
                    # Convert binary image to cv2 format
                    nparr = np.frombuffer(image_data, np.uint8)
                    image_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if image_cv2 is None:
                        return {"error": "Failed to decode image"}
                except Exception as e:
                    # Handle potential decoding or conversion errors
                    print(f"Error decoding base64 image: {e}")
                    return {"error": "Invalid base64 image data"}
            except Exception as e:
                print(f"Error processing request: {e}")
                return {"error": "Invalid request data"}

            prediction_data = {
                "url": url,
                "image": image_cv2,
            }
            return await self.predict(prediction_data)

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

    def convert_image(self, image_content):
        try:
            im_bytes = base64.b64decode(image_content)
            im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
            return cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"Error converting image: {e}")
            raise


    @abstractmethod
    async def predict(self, data: dict):
        """Abstract method that subclasses must implement"""
        pass
