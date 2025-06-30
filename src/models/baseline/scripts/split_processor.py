#!/usr/bin/env python3
"""Simple split processor for baseline evaluation."""

import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from tools.metrics import calculate_metrics

logger = logging.getLogger(__name__)


class SplitProcessor:
    """Simple processor for a single data split."""

    def __init__(self, batch_size: int = 512, top_k: int = 1, threshold: float = None):
        self.batch_size = batch_size
        self.top_k = top_k
        self.threshold = threshold
        self.baseline_dir = Path(__file__).parent.parent

    def process_split(
        self, split_paths: Dict[str, Path], temp_dir: Path
    ) -> Dict[str, Any]:
        """Process a single split through the pipeline."""
        start_time = time.time()

        # Step 1: Create index with benign samples
        index_path = temp_dir / "index.faiss"
        metadata_path = temp_dir / "metadata.json"
        self._run_load(
            split_paths["train_benign"], index_path, metadata_path, is_phish=False
        )

        # Step 2: Extend index with phish samples
        self._run_load(
            split_paths["train_phish"],
            index_path,
            metadata_path,
            is_phish=True,
            append=True,
        )

        # Step 3: Query validation set
        results_path = temp_dir / "results.csv"
        self._run_query(split_paths["val"], index_path, metadata_path, results_path)

        # Step 4: Calculate metrics
        class_metrics, target_metrics = self._calculate_metrics(results_path)

        # Step 5: Collect info
        info = self._collect_info(split_paths, start_time)

        return {
            "class_metrics": class_metrics,
            "target_metrics": target_metrics,
            "info": info,
        }

    def _run_load(
        self,
        images_dir: Path,
        index_path: Path,
        metadata_path: Path,
        is_phish: bool,
        append: bool = False,
    ):
        """Run load.py script."""
        cmd = [
            sys.executable,
            str(self.baseline_dir / "load.py"),
            "--images",
            str(images_dir),
            "--index",
            str(index_path),
            "--batch-size",
            str(self.batch_size),
        ]

        if is_phish:
            cmd.append("--is-phish")
        if append:
            cmd.append("--append")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Load failed: {result.stderr}")

    def _run_query(
        self, images_dir: Path, index_path: Path, metadata_path: Path, output_path: Path
    ):
        """Run query.py script."""
        cmd = [
            sys.executable,
            str(self.baseline_dir / "query.py"),
            "--images",
            str(images_dir),
            "--index",
            str(index_path),
            "--output",
            str(output_path),
            "--top-k",
            str(self.top_k),
            "--batch-size",
            str(self.batch_size),
        ]

        if self.threshold:
            cmd.extend(["--threshold", str(self.threshold)])

        # Simple phish detection based on path
        if "phish" in str(images_dir).lower():
            cmd.append("--is-phish")

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Query failed: {result.stderr}")

    def _calculate_metrics(self, csv_path: Path) -> tuple:
        """Calculate metrics from results CSV."""
        df = pd.read_csv(csv_path)

        # Basic validation
        required_cols = [
            "baseline_class",
            "baseline_target",
            "true_class",
            "true_target",
        ]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"Missing required columns in {csv_path}")

        # Fill missing values
        df = df.fillna(
            {
                "baseline_target": "unknown",
                "true_target": "unknown",
                "baseline_class": 0,
                "true_class": 0,
            }
        )

        return calculate_metrics(
            cls_true=df["true_class"],
            cls_pred=df["baseline_class"],
            targets_true=df["true_target"],
            targets_pred=df["baseline_target"],
        )

    def _collect_info(
        self, split_paths: Dict[str, Path], start_time: float
    ) -> Dict[str, Any]:
        """Collect basic split information."""
        return {
            "train_benign_count": len(list(split_paths["train_benign"].glob("*"))),
            "train_phish_count": len(list(split_paths["train_phish"].glob("*"))),
            "val_count": len(list(split_paths["val"].glob("*"))),
            "processing_time": time.time() - start_time,
        }
