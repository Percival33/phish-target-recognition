from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import httpx
import asyncio
import os
import base64

MODELS = os.getenv("MODELS", "").split(",") if os.getenv("MODELS") else []
SERVICE_PORTS = {
    "visualphish": os.getenv("VP_PORT", 8888),
    "phishpedia": os.getenv("PP_PORT", 8888),
}
URLS = [f"http://{model}:{SERVICE_PORTS.get(model, 8888)}/predict" for model in MODELS]

router = APIRouter()


async def fetch_data(
    client: httpx.AsyncClient, url: str, image_data: bytes, url_param: str
):
    try:
        # Send the image as a binary file
        files = {
            "image": image_data,
        }
        data = {
            "url": url_param,
        }

        print(f"Fetching {url} with URL param: {url_param} and binary image data...")
        response = await client.post(url, data=data, files=files)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, KeyError, ValueError) as e:
        print(f"Error fetching {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching {url}: {str(e)}")


@router.get("/models")
async def models_list():
    return {"models": MODELS}


@router.post("/predict")
async def predict(image: UploadFile = File(...), url: str = Form(...)):
    try:
        image_data = await image.read()
    except Exception as e:
        print(f"Error reading image: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading image: {e}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [
            fetch_data(client, model_url, image_data, url)
            for model_url in URLS
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
