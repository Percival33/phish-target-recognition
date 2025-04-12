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
    client: httpx.AsyncClient, url: str, image: str, url_param: str
):
    try:
        data = {
            "url": url_param,
            "image": image
        }

        print(f"Fetching {url} with URL param: {url_param} and base64 image data of length {len(image)}")
        response = await client.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, KeyError, ValueError) as e:
        print(f"Error fetching {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching {url}: {str(e)}")


@router.get("/models")
async def models_list():
    return {"models": MODELS}


@router.post("/predict")
async def predict(image: str = Form(...), url: str = Form(...)):
    try:
        # Decode base64 string to binary
        image_data = base64.b64decode(image)
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        raise HTTPException(status_code=500, detail=f"Error decoding base64 image: {e}")

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [
            fetch_data(client, model_url, image, url)
            for model_url in URLS
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    return results
