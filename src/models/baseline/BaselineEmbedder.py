import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import faiss
import pandas as pd
from perception import hashers
from tqdm import tqdm

from tools.config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)


class BaselineEmbedder:
    def __init__(
        self,
        target_mapping: Optional[Dict[str, str]] = None,
        hasher: Optional[hashers.PerceptualHash] = None,
        index_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        self.hasher = hasher if hasher is not None else hashers.PHash()
        self.index = None
        self.image_metadata: List[Dict[str, Any]] = []
        self.target_mapping = target_mapping

        # Try to load index if path is provided and file exists
        if index_path is not None and index_path.exists():
            try:
                self.index = faiss.read_index(str(index_path))
                logger.info(
                    f"Loaded FAISS index from {index_path} with {self.index.ntotal} vectors"
                )
                # Try to load metadata if path is provided
                if metadata_path is not None and metadata_path.exists():
                    self.load_metadata_csv(metadata_path)
            except Exception as e:
                logger.error(f"Failed to load FAISS index from {index_path}: {e}")

    def load_metadata_csv(self, metadata_path: Path) -> bool:
        """Load metadata from CSV file.

        Args:
            metadata_path: Path to metadata CSV file

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            df = pd.read_csv(metadata_path)
            if len(df) != self.index.ntotal:
                logger.error(
                    f"Metadata count ({len(df)}) does not match index count ({self.index.ntotal})"
                )
                return False

            self.image_metadata = df.to_dict("records")
            logger.info(f"Loaded metadata for {len(self.image_metadata)} images")
            return True
        except Exception as e:
            logger.error(f"Failed to load metadata from {metadata_path}: {e}")
            return False

    def save_metadata_csv(self, metadata_path: Path, overwrite: bool = False) -> bool:
        """Save metadata to CSV file.

        Args:
            metadata_path: Path to save metadata CSV
            overwrite: Whether to overwrite existing file

        Returns:
            bool: True if successful, False otherwise
        """
        if metadata_path.exists() and not overwrite:
            logger.warning(
                f"Metadata file {metadata_path} already exists. Use overwrite=True to replace."
            )
            return False

        try:
            df = pd.DataFrame(self.image_metadata)
            df.to_csv(metadata_path, index=False)
            logger.info(
                f"Saved metadata for {len(self.image_metadata)} images to {metadata_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save metadata to {metadata_path}: {e}")
            return False

    def save_index(self, index_path: Path, overwrite: bool = False) -> bool:
        if self.index is None:
            logger.warning("No index to save")
            return False

        if index_path.exists() and not overwrite:
            logger.warning(
                f"Index file {index_path} already exists. Use overwrite=True to replace."
            )
            return False

        try:
            faiss.write_index(self.index, str(index_path))
            logger.info(
                f"Saved FAISS index to {index_path} with {self.index.ntotal} vectors"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save FAISS index to {index_path}: {e}")
            return False

    def compute_embeddings(
        self, image_paths: List[Path], is_phish: bool = False, batch_size: int = 32
    ) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
        """Compute perceptual hashes for a list of images and store their metadata.

        Args:
            image_paths: List of paths to images to process
            batch_size: Number of images to process in each batch

        Returns:
            Tuple containing:
            - FAISS index with stored embeddings
            - List of metadata dictionaries for each processed image
        """
        if not image_paths:
            logger.error("No image paths provided")
            return self.index, []

        # Reset metadata for fresh computation
        self.image_metadata = []
        batch_embeddings = []

        # Process images in batches
        for batch_start in tqdm(
            range(0, len(image_paths), batch_size), desc="Processing images"
        ):
            batch_paths = image_paths[batch_start : batch_start + batch_size]

            for img_path in batch_paths:
                try:
                    phash = self.hasher.compute(str(img_path))
                    phash_array = np.array(phash, dtype=np.float32).reshape(1, -1)

                    true_target = str(img_path.parent.name).split("+")[0]

                    metadata = {
                        "file": img_path.name,
                        "phash": phash,
                        "true_class": is_phish,
                        "true_target": true_target,
                    }

                    self.image_metadata.append(metadata)
                    batch_embeddings.append(phash_array)

                except Exception as e:
                    logger.error(f"Failed to process image {img_path}: {str(e)}")
                    continue

            # Create or update FAISS index with batch
            if batch_embeddings:
                if self.index is None:
                    embedding_dim = batch_embeddings[0].shape[1]
                    self.index = faiss.IndexFlatL2(embedding_dim)

                embeddings_array = np.vstack(batch_embeddings)
                self.index.add(embeddings_array)
                batch_embeddings = []  # Clear batch after adding to index

        if not self.image_metadata:
            logger.warning("No images were successfully processed")

        return self.index, self.image_metadata

    def search_similar(
        self,
        query_paths: List[Path],
        k: int = 5,
        threshold: Optional[float] = None,
        output_path: Optional[Path] = None,
        batch_size: int = 32,
        true_targets: Optional[List[str]] = None,
        true_classes: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Search for similar images in the index and optionally save results to CSV.

        Args:
            query_paths: List of paths to query images
            k: Number of similar images to return per query
            threshold: Optional distance threshold for binary classification
            output_path: Optional path to save results CSV
            batch_size: Number of images to process in each batch to control memory usage
            true_targets: Optional list of true target labels for query images. Must be same length as query_paths.
            true_classes: Optional list of true class labels for query images. Must be same length as query_paths.

        Returns:
            DataFrame with results in the format:
            file,baseline_class,baseline_distance,baseline_target,true_class,true_target
        """
        if self.index is None:
            logger.error("No index loaded")
            return pd.DataFrame()

        if true_targets is not None and len(true_targets) != len(query_paths):
            logger.error("Number of true_targets must match number of query paths")
            return pd.DataFrame()

        if true_classes is not None and len(true_classes) != len(query_paths):
            logger.error("Number of true_classes must match number of query paths")
            return pd.DataFrame()

        results = []

        # Process queries in batches
        for batch_start in tqdm(
            range(0, len(query_paths), batch_size), desc="Processing queries"
        ):
            batch_end = min(batch_start + batch_size, len(query_paths))
            batch_paths = query_paths[batch_start:batch_end]
            batch_embeddings = []

            # Compute embeddings for the batch
            for query_path in batch_paths:
                try:
                    query_hash = self.hasher.compute(str(query_path))
                    query_array = np.array(query_hash, dtype=np.float32).reshape(1, -1)
                    batch_embeddings.append(query_array)
                except Exception as e:
                    logger.error(f"Failed to process query {query_path}: {str(e)}")
                    continue

            if not batch_embeddings:
                continue

            # Stack batch embeddings and perform batch search
            batch_embeddings_array = np.vstack(batch_embeddings)
            distances, indices = self.index.search(batch_embeddings_array, k)

            # Process batch results
            for i, (query_path, distances, indices) in enumerate(
                zip(batch_paths, distances, indices)
            ):
                try:
                    closest_distance = float(distances[0])
                    closest_idx = int(indices[0])
                    closest_metadata = self.image_metadata[closest_idx]

                    result = {
                        "file": query_path.name,
                        "baseline_class": closest_metadata["true_class"],
                        "baseline_distance": closest_distance,
                        "baseline_target": closest_metadata["true_target"],
                        "true_class": true_classes[batch_start + i]
                        if true_classes is not None
                        else None,
                        "true_target": true_targets[batch_start + i]
                        if true_targets is not None
                        else None,
                    }
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process result for {query_path}: {str(e)}")
                    continue

        # Create DataFrame with results
        df = pd.DataFrame(results)

        # Save to CSV if output path provided
        if output_path is not None:
            df.to_csv(output_path, index=False)
            logger.info(f"Saved results to {output_path}")

        return df

    def search_by_image(
        self, image: np.ndarray, k: int = 1, threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """Search for similar images in the index using a raw image array.

        Args:
            image: OpenCV image array (numpy.ndarray)
            k: Number of similar images to return (default: 1)
            threshold: Optional distance threshold for binary classification

        Returns:
            Dictionary containing:
            - distances: List of distance scores for top k matches
            - matches: List of dictionaries containing metadata for each match
            - baseline_class: Binary classification if threshold provided
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning(
                "FAISS index is not initialized or is empty. Cannot perform search."
            )
            return {"error": "Index not initialized or empty"}

        if not isinstance(image, np.ndarray):
            logger.error("Input must be a numpy array (OpenCV image)")
            return {"error": "Invalid input type"}

        try:
            # Compute perceptual hash for input image
            query_hash = self.hasher.compute_array(image)
            query_array = np.array(query_hash, dtype=np.float32).reshape(1, -1)

            # Perform similarity search
            distances, indices = self.index.search(query_array, k)

            # Process results
            matches = []
            for idx, distance in zip(indices[0], distances[0]):
                match_metadata = self.image_metadata[idx].copy()
                match_metadata["distance"] = float(distance)
                matches.append(match_metadata)

            result = {"distances": distances[0].tolist(), "matches": matches}

            # Add binary classification if threshold provided
            if threshold is not None:
                result["baseline_class"] = 1 if distances[0][0] < threshold else 0

            return result

        except Exception as e:
            logger.error(f"Failed to process image: {str(e)}")
            return {"error": str(e)}
