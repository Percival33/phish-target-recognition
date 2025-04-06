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
        async def predict_route(image: UploadFile = File(...), url: str = Form(...)):
            # Convert the uploaded file to a dict with content and metadata
            image_content = await image.read()
            # TODO: convert image_content from base64
            data = {
                "url": url,
                "image_content": image_content,
                "image_filename": image.filename,
                "image_content_type": image.content_type,
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
        im_bytes = base64.b64decode(image_content)
        im_arr = np.frombuffer(im_bytes, dtype=np.uint8)
        return cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)

    @abstractmethod
    async def predict(self, data: dict):
        """Abstract method that subclasses must implement"""
        pass
