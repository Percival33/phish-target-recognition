from argparse import ArgumentParser
from pathlib import Path
import json
import logging
import os

import cv2
from tools.ModelServing import ModelServing
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

        # Save image temporarily for processing
        temp_path = Path("/tmp/temp_image.jpg")
        cv2.imwrite(str(temp_path), img)

        # Search for similar images
        results = self.embedder.search_similar(
            query_paths=[temp_path], k=1, threshold=self.args.threshold
        )

        # Clean up temp file
        temp_path.unlink()

        if results.empty:
            return {
                "url": str(url),
                "class": 0,
                "target": "unknown",
                "confidence": 1.0,
                "baseline_distance": float("inf"),
            }

        # Get first result
        result = results.iloc[0]

        return {
            "url": str(url),
            "class": int(result.get("baseline_class", 0)),
            "target": str(result.get("baseline_target", "unknown")),
            "confidence": 1.0,
            "baseline_distance": float(result.get("baseline_distance", float("inf"))),
        }

    async def on_startup(self):
        """Startup logic - load FAISS index and target mappings"""
        logger.info("Starting up BaselineServing...")

        # Load target mappings if available
        mapping_path = Path("/code/config/target_mappings.json")
        if mapping_path.exists():
            try:
                with open(mapping_path) as f:
                    self.target_mapping = json.load(f)
                logger.info("Loaded target mappings")
            except Exception as e:
                logger.error(f"Failed to load target mappings: {e}")
                self.target_mapping = None

        # Initialize embedder with FAISS index
        self.embedder = BaselineEmbedder(
            target_mapping=self.target_mapping,
            index_path=Path("/code/index/faiss_index.idx"),
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--threshold", type=float, default=float(os.getenv("DISTANCE_THRESHOLD", "1.0"))
    )
    args = parser.parse_args()

    serving = BaselineServing(args)
    serving.run()
