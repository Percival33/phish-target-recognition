from argparse import ArgumentParser
from pathlib import Path
import logging
import os

from tools.ModelServing import ModelServing, PredictResponse
from tools.config import setup_logging

from BaselineEmbedder import BaselineEmbedder

setup_logging()
logger = logging.getLogger(__name__)


class BaselineServing(ModelServing):
    def __init__(self, args=None):
        super().__init__()
        self.args = args
        self.embedder = None
        self.target_mapping = None

    async def predict(self, data: dict):
        """Implementation of the predict method for BaselineServing"""
        img = data.get("image", None)
        url = data.get("url", None)

        # Search for similar images directly using the image array
        results = self.embedder.search_by_image(
            image=img, k=1, threshold=self.args.threshold
        )

        # Handle error cases
        if "error" in results:
            return PredictResponse(
                url=str(url), class_=0, target="unknown", distance=float("inf")
            )

        match = results["matches"][0]
        baseline_class = results.get("baseline_class", 0)

        return PredictResponse(
            url=str(url),
            class_=int(baseline_class),
            target=str(match.get("true_target", "unknown")),
            distance=float(match.get("distance", float("inf"))),
        )

    async def on_startup(self):
        """Startup logic - load FAISS index and target mappings"""
        logger.info("Starting up BaselineServing...")

        index_path = Path("/code/index/faiss_index.idx")
        metadata_path = index_path.with_suffix(".csv")

        self.embedder = BaselineEmbedder(
            index_path=index_path, metadata_path=metadata_path
        )

        if not self.embedder.index:
            raise RuntimeError("Failed to initialize embedder with index")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--threshold", type=float, default=float(os.getenv("DISTANCE_THRESHOLD", "1.0"))
    )
    args = parser.parse_args()

    serving = BaselineServing(args)
    serving.run()
