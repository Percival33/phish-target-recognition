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
    client: httpx.AsyncClient, url: str, image_data, url_param
):
    try:
        # Encode image data to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Send as form data
        data = {
            "url": url_param,
            "image_base64": image_base64,
        }

        print(f"Fetching {url} with URL param: {url_param} and image_base64 (first 50 chars): {image_base64[:50]}...")
        # Use data parameter instead of files
        response = await client.post(url, data=data)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, KeyError, ValueError, base64.binascii.Error) as e:
        print(f"Error fetching {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching {url}: {str(e)}")


@router.get("/models")
async def models_list():
    return {"models": MODELS}


@router.post("/predict")
async def predict(image: UploadFile = File(...), url: str = Form(...)):
    image_data = await image.read()

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [
            fetch_data(client, model_url, image_data, url)
            for model_url in URLS
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
