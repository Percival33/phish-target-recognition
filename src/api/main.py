from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import httpx
import asyncio
import uvicorn
import os

MODELS = []
PORT = int(os.getenv("API_PORT", 8888))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODELS
    MODELS = os.getenv("MODELS", "").split(",") if os.getenv("MODELS") else []
    print(f"Initializing with models: {MODELS}")
    yield
    print("Cleaning up...")


app = FastAPI(lifespan=lifespan)


async def fetch_data(
    client: httpx.AsyncClient, url: str, image_data, image_filename, url_param
) -> dict:
    try:
        # Prepare the multipart form data
        files = {"image": (image_filename, image_data, "application/octet-stream")}
        data = {"url": url_param}

        print(f"Fetching {url} with URL param: {url_param}")

        # Send as multipart form data
        response = await client.post(url, files=files, data=data)
        response.raise_for_status()
        return response.json()
    except (httpx.HTTPError, KeyError, ValueError) as e:
        print(f"Error fetching {url}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching {url}: {str(e)}")


@app.get("/models")
async def models_list():
    return {"models": MODELS}


def get_service(service_name):
    port_map = {
        "visualphish": os.getenv("VP_PORT", 8888),
        "phishpedia": os.getenv("PP_PORT", 8888),
    }
    return f"{service_name}:{port_map.get(service_name, 8888)}"


@app.post("/predict")
async def predict(image: UploadFile = File(...), url: str = Form(...)):
    # Read image data
    image_data = await image.read()

    # Create URLs for each model service
    urls = [f"http://{get_service(model)}/predict" for model in MODELS]

    async with httpx.AsyncClient(timeout=30.0) as client:
        tasks = [
            fetch_data(client, model_url, image_data, image.filename, url)
            for model_url in urls
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
    return results


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
