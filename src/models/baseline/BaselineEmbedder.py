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
            except Exception as e:
                logger.error(f"Failed to load FAISS index from {index_path}: {e}")

    def save_index(self, index_path: Path) -> bool:
        if self.index is None:
            logger.warning("No index to save")
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
        true_classes: Optional[List[bool]] = None,
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
            DataFrame with columns:
            - file: query image filename
            - baseline_class: binary classification (if threshold provided)
            - baseline_distance: distance to closest match
            - baseline_target: target from closest match
            - true_class: from query metadata if available
            - true_target: from query metadata if available
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning(
                "FAISS index is not initialized or is empty. Cannot perform search."
            )
            return pd.DataFrame()

        if true_targets is not None and len(true_targets) != len(query_paths):
            logger.error(
                f"Length mismatch: true_targets ({len(true_targets)}) != query_paths ({len(query_paths)})"
            )
            return pd.DataFrame()

        if true_classes is not None and len(true_classes) != len(query_paths):
            logger.error(
                f"Length mismatch: true_classes ({len(true_classes)}) != query_paths ({len(query_paths)})"
            )
            return pd.DataFrame()

        results = []

        # Process in batches to control memory usage
        for batch_start in range(0, len(query_paths), batch_size):
            batch_end = min(batch_start + batch_size, len(query_paths))
            batch_paths = query_paths[batch_start:batch_end]

            # Pre-compute embeddings for current batch
            query_embeddings = []
            query_names = []
            batch_true_targets = []
            batch_true_classes = []

            logger.info(
                f"Computing embeddings for batch {batch_start // batch_size + 1}/{(len(query_paths) - 1) // batch_size + 1}..."
            )
            for i, query_path in enumerate(
                tqdm(batch_paths, desc="Computing embeddings")
            ):
                try:
                    query_hash = self.hasher.compute(str(query_path))
                    query_array = np.array(query_hash, dtype=np.float32).reshape(1, -1)
                    query_embeddings.append(query_array)
                    query_names.append(query_path.name)
                    batch_true_targets.append(
                        true_targets[batch_start + i] if true_targets else None
                    )
                    batch_true_classes.append(
                        true_classes[batch_start + i] if true_classes else None
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to process query image {query_path}: {str(e)}"
                    )
                    continue

            if not query_embeddings:
                logger.warning(
                    f"No query embeddings could be computed for batch {batch_start // batch_size + 1}"
                )
                continue

            # Stack embeddings for current batch
            query_matrix = np.vstack(query_embeddings)

            # Perform batch search
            logger.info(
                f"Performing batch similarity search for batch {batch_start // batch_size + 1}..."
            )
            distances, indices = self.index.search(query_matrix, k)

            # Process batch results
            for query_name, true_target, true_class, distance_row, index_row in zip(
                query_names, batch_true_targets, batch_true_classes, distances, indices
            ):
                closest_match = self.image_metadata[index_row[0]]
                closest_distance = float(distance_row[0])

                result = {
                    "file": query_name,
                    "baseline_distance": closest_distance,
                    "baseline_target": closest_match["true_target"],
                    "true_class": true_class,
                    "true_target": true_target,
                }

                if threshold is not None:
                    result["baseline_class"] = 1 if closest_distance < threshold else 0

                results.append(result)

        if not results:
            logger.warning("No results could be generated for any batch")
            return pd.DataFrame()

        # Create DataFrame with specified column order
        df = pd.DataFrame(results)
        columns = [
            "file",
            "baseline_class",
            "baseline_distance",
            "baseline_target",
            "true_class",
            "true_target",
        ]
        df = df.reindex(columns=columns)

        # Save to CSV if output path provided
        if output_path is not None:
            if output_path.exists():
                logger.warning(
                    f"Output file {output_path} already exists. Use --overwrite flag to replace."
                )
            else:
                df.to_csv(output_path, index=False)
                logger.info(f"Results saved to {output_path}")

        return df
