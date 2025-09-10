import os

import uvicorn

from app_factory import create_app

# Create app instance for local development and testing
app = create_app(use_relative_imports=True)

if __name__ == "__main__":
    PORT = int(os.getenv("API_PORT", 8888))
    uvicorn.run(app, host="0.0.0.0", port=PORT)
