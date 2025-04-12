from abc import ABC, abstractmethod
from fastapi import FastAPI, UploadFile, File, Form
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
        async def predict_route(image_base64: str = Form(...), url: str = Form(...)):
            # Convert the base64 encoded image to cv2 format
            try:
                image_cv2 = self.convert_image(image_base64)
            except Exception as e:
                # Handle potential decoding or conversion errors
                # You might want to log the error and return a specific HTTP status code
                print(f"Error converting image: {e}")
                # Example: raise HTTPException(status_code=400, detail="Invalid image data")
                return {"error": "Invalid image data"} # Or re-raise, or return appropriate response

            data = {
                "url": url,
                "image": image_cv2,
            }
            return await self.predict(data)

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
