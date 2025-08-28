#!/usr/bin/env python3
"""Simple results aggregator for baseline evaluation."""

import json
import statistics
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List


class ResultsAggregator:
    """Simple aggregator for split results."""

    def __init__(self, dataset_name: str, config: Dict[str, Any]):
        self.dataset_name = dataset_name
        self.config = config

    def aggregate_results(
        self, split_results: Dict[str, Dict], failed_splits: List[str]
    ) -> Dict[str, Any]:
        """Aggregate results from all splits."""
        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "dataset": self.dataset_name,
                "total_splits": len(split_results) + len(failed_splits),
                "successful_splits": len(split_results),
                "failed_splits": failed_splits,
                "config": self.config,
            },
            "summary": self._calculate_summary(split_results),
            **split_results,
        }

    def _calculate_summary(self, results: Dict[str, Dict]) -> Dict[str, Dict]:
        """Calculate summary statistics."""
        # Collect values for each metric
        class_metrics = {}
        target_metrics = {}
        info_metrics = {}

        for split_result in results.values():
            # Class metrics
            for metric, value in split_result["class_metrics"].items():
                if metric not in class_metrics:
                    class_metrics[metric] = []
                class_metrics[metric].append(value)

            # Target metrics
            for metric, value in split_result["target_metrics"].items():
                if metric not in target_metrics:
                    target_metrics[metric] = []
                target_metrics[metric].append(value)

            # Info metrics
            for metric, value in split_result["info"].items():
                if isinstance(value, (int, float)):
                    if metric not in info_metrics:
                        info_metrics[metric] = []
                    info_metrics[metric].append(value)

        return {
            "class_metrics": {
                metric: self._calc_stats(values)
                for metric, values in class_metrics.items()
            },
            "target_metrics": {
                metric: self._calc_stats(values)
                for metric, values in target_metrics.items()
            },
            "info": {
                metric: self._calc_stats(values)
                for metric, values in info_metrics.items()
            },
        }

    def _calc_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate basic statistics."""
        if not values:
            return {"mean": 0, "std": 0, "min": 0, "max": 0, "count": 0}

        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
        }

    def save_results(self, results: Dict[str, Any], output_path: Path):
        """Save results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

    def print_summary(self, results: Dict[str, Any]):
        """Print simple summary."""
        metadata = results["metadata"]
        summary = results["summary"]

        print(f"\n{'=' * 60}")
        print(f"BASELINE EVALUATION RESULTS - {metadata['dataset']}")
        print(f"{'=' * 60}")
        print(
            f"Successful splits: {metadata['successful_splits']}/{metadata['total_splits']}"
        )

        if metadata["failed_splits"]:
            print(f"Failed splits: {metadata['failed_splits']}")

        print("\nClass Metrics:")
        for metric, stats in summary["class_metrics"].items():
            print(f"  {metric:15}: {stats['mean']:.3f} ± {stats['std']:.3f}")

        print("\nTarget Metrics:")
        for metric, stats in summary["target_metrics"].items():
            print(f"  {metric:15}: {stats['mean']:.3f} ± {stats['std']:.3f}")

        print(f"{'=' * 60}\n")
