import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import faiss
from PIL import Image
from perception import hashers
from tqdm import tqdm

from tools.config import setup_logging

setup_logging()

# It's good practice for each module to have its own logger.
logger = logging.getLogger(__name__)


class BaselineEmbedder:
    def __init__(self, target_mapping: Optional[Dict[str, str]] = None):
        self.hasher = hashers.PHash()
        self.index = None
        self.image_metadata: List[Dict[str, Any]] = []
        self.target_mapping = target_mapping

    def compute_embeddings(
        self, image_paths: List[Path], batch_size: int = 32
    ) -> List[Dict[str, Any]]:
        """Compute perceptual hashes for a list of images and store their metadata."""
        # Clear previous state if any, or ensure this is instance-specific
        self.image_metadata = []

        all_embeddings_for_faiss = []

        for batch_start_idx in tqdm(
            range(0, len(image_paths), batch_size), desc="Computing perceptual hashes"
        ):
            batch_paths = image_paths[batch_start_idx : batch_start_idx + batch_size]

            current_batch_metadata = []
            current_batch_embeddings_for_faiss = []

            for img_path in batch_paths:
                try:
                    phash = self.hasher.compute(str(img_path))
                    phash_array = np.array(phash, dtype=np.float32).reshape(1, -1)

                    image_file_name = img_path.name

                    # Determine true_class from parent directory name (assumption)
                    current_true_class_raw = None
                    try:
                        # Ensure parent directory is meaningful (e.g., not '.' or the root of image_paths)
                        # This simple check assumes direct parent is the class folder.
                        parent_name = img_path.parent.name
                        if parent_name and parent_name != ".":  # Basic check
                            current_true_class_raw = str(parent_name)
                    except AttributeError:
                        current_true_class_raw = None

                    # Initialize target fields
                    final_baseline_class = None
                    final_baseline_target = None
                    final_true_target = None

                    if current_true_class_raw and self.target_mapping:
                        mapped_value = self.target_mapping.get(current_true_class_raw)
                        if mapped_value is not None:
                            final_true_target = mapped_value
                            final_baseline_class = mapped_value
                            final_baseline_target = mapped_value

                    item_metadata = {
                        "file": image_file_name,
                        "phash": phash,  # Storing the original phash object
                        "baseline_class": final_baseline_class,
                        "baseline_distance": None,
                        "baseline_target": final_baseline_target,
                        "true_class": current_true_class_raw,
                        "true_target": final_true_target,
                    }

                    current_batch_metadata.append(item_metadata)
                    current_batch_embeddings_for_faiss.append(phash_array)

                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    continue

            self.image_metadata.extend(current_batch_metadata)
            all_embeddings_for_faiss.extend(current_batch_embeddings_for_faiss)

        if not all_embeddings_for_faiss:
            logger.warning(
                "No embeddings were computed. FAISS index will not be created/updated."
            )
            return self.image_metadata

        # Create FAISS index if not exists
        if self.index is None:
            # Check if embeddings list is not empty before accessing shape
            if all_embeddings_for_faiss:
                embedding_dim = all_embeddings_for_faiss[0].shape[1]
                self.index = faiss.IndexFlatL2(embedding_dim)
            else:  # Should ideally not happen if check above is done, but as safeguard
                logger.error("Cannot create FAISS index: no embeddings available.")
                return self.image_metadata  # Or raise error

        # Add embeddings to FAISS index
        if self.index is not None and all_embeddings_for_faiss:
            embeddings_array = np.vstack(all_embeddings_for_faiss)
            self.index.add(embeddings_array)

        return self.image_metadata

    def search_similar(
        self, query_image: Image.Image, k: int = 5
    ) -> Tuple[List[str], List[float], List[str]]:
        """Search for similar images in the index.

        Args:
            query_image: Query image to find similar images for
            k: Number of similar images to return

        Returns:
            Tuple containing:
            - List of file names of similar images
            - List of distances to similar images
            - List of baseline targets for similar images from the indexed data
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning(
                "FAISS index is not initialized or is empty. Cannot perform search."
            )
            return [], [], []

        # Compute query embedding
        query_hash = self.hasher.compute(query_image)
        query_array = np.array(query_hash, dtype=np.float32).reshape(1, -1)

        # Ensure k is not greater than the number of items in the index
        effective_k = min(k, self.index.ntotal)
        if effective_k == 0:  # Should be caught by ntotal check above but good practice
            return [], [], []

        # Search in FAISS index
        distances, indices = self.index.search(query_array, effective_k)

        # Get metadata for matched indices
        found_metadata_list = [self.image_metadata[idx] for idx in indices[0]]

        similar_files = [meta["file"] for meta in found_metadata_list]
        output_distances = distances[0].tolist()
        # Retrieve the baseline_target stored during indexing for these similar images
        output_targets = [meta["baseline_target"] for meta in found_metadata_list]

        return similar_files, output_distances, output_targets
