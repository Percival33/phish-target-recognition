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
        form_data = {
            "url": url_param,
            "image": image
        }

        print(f"Fetching {url} with URL param: {url_param}")
        print(f"Image data type: {type(image)}, length: {len(image)}")
        
        # Send as form data
        response = await client.post(url, data=form_data)
        
        # Debug response
        if response.status_code != 200:
            print(f"Error response from {url}: {response.status_code} - {response.text}")
            
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
        # Validate base64 string by attempting to decode it
        try:
            # Test if this is valid base64
            base64.b64decode(image)
        except Exception as e:
            print(f"Invalid base64 encoding: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid base64 encoding: {e}")
            
        async with httpx.AsyncClient(timeout=30.0) as client:
            tasks = [
                fetch_data(client, model_url, image, url)
                for model_url in URLS
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    except Exception as e:
        print(f"Error in predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Error in predict endpoint: {e}")
