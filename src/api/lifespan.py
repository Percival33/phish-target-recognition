from contextlib import asynccontextmanager
from fastapi import FastAPI
from config import MODELS


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Initializing with models: {MODELS}")
    yield
    print("Cleaning up...")
