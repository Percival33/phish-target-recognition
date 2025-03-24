from ..ModelServing import ModelServing


class ExampleServing(ModelServing):
    def __init__(self):
        super().__init__()

    async def predict(self, data: dict):
        """Implementation of the predict method"""
        return {"prediction": "example"}


serving = ExampleServing()
app = serving.app

if __name__ == "__main__":
    serving.run()
# uv run fastapi run ExampleServing.py
