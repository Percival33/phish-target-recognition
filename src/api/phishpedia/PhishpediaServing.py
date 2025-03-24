from ..ModelServing import ModelServing


class PhishpediaServing(ModelServing):
    def __init__(self):
        super().__init__()

    async def predict(self, data: dict):
        """Implementation of the predict method for Phishpedia"""
        return {"prediction": "phishpedia"}


serving = PhishpediaServing()
app = serving.app

if __name__ == "__main__":
    serving.run()
# uv run fastapi run PhishpediaServing.py
