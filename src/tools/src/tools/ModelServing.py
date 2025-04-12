from fastapi import HTTPException
from pydantic import BaseModel
import base64
import cv2
import numpy as np


class PredictRequest(BaseModel):
    url: str
    image: str


def register_routes(self):
    """Register routes for the FastAPI application"""

    @self.app.post("/predict")
    async def predict_route(request_data: PredictRequest):
        try:
            url = request_data.url
            image_str = request_data.image

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
                    raise HTTPException(status_code=400, detail="Failed to decode image to OpenCV format")

                print(f"Successfully converted to cv2 image with shape: {image_cv2.shape}")
            except Exception as e:
                # Handle potential decoding or conversion errors
                print(f"Error decoding base64 image: {str(e)}")
                raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Invalid request data: {str(e)}")

        prediction_data = {
            "url": url,
            "image": image_cv2,
        }
        try:
            result = await self.predict(prediction_data)
            return result
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
