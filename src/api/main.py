import uvicorn
from fastapi import FastAPI
from lifespan import lifespan
from routes import router
import os

PORT = int(os.getenv("API_PORT", 8888))

app = FastAPI(lifespan=lifespan)

app.include_router(router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
