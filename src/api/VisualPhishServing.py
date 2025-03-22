import time

from .ModelServing import ModelServing


class VisualPhishServing(ModelServing):
    def __init__(self):
        super().__init__()

    async def predict(self, data: dict):
        """Implementation of the predict method for VisualPhish"""
        time.sleep(2)
        return {"prediction": "visualphish"}


serving = VisualPhishServing()
app = serving.app

if __name__ == "__main__":
    serving.run()
# uv run fastapi run VisualPhishServing.py
