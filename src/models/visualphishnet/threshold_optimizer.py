#!/usr/bin/env python3
"""
VisualPhish Threshold Optimizer

A standalone script to optimize threshold for VisualPhish using Equal Error Rate (EER).
"""

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from tools.config import setup_logging

from eval_new import load_targetemb, process_dataset
from Evaluate import Evaluate
from ModelHelper import ModelHelper


def _get_scipy_stats():
    """Lazy import of scipy.stats to avoid dependency when not plotting."""
    try:
        from scipy import stats

        return stats
    except ImportError:
        raise ImportError("scipy is required for plotting. Install with: pip install scipy")


def parse_arguments():
    """Parse command line arguments."""
    parser = ArgumentParser(description="VisualPhish Threshold Optimizer")

    parser.add_argument("--emb-dir", type=Path, required=True, help="Directory containing target embeddings")
    parser.add_argument("--val-phish-dir", type=Path, required=True, help="Validation phishing dataset directory")
    parser.add_argument("--val-benign-dir", type=Path, required=True, help="Validation benign dataset directory")
    parser.add_argument("--model-name", type=str, default="model2", help="Model name (default: model2)")
    parser.add_argument("--margin", type=float, default=2.2)
    parser.add_argument("--reshape-size", default=[224, 224, 3])
    parser.add_argument("--batch-size", type=int, default=512, help="Processing batch size (default: 512)")
    parser.add_argument("--mean", type=float, required=True, help="Mean value for statistical threshold range")
    parser.add_argument("--std", type=float, required=True, help="Standard deviation for statistical threshold range")
    parser.add_argument("--max", type=float, required=True, help="Maximum value for absolute threshold range")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for results")
    parser.add_argument("--plot", action="store_true", help="Generate plots and enable scipy")

    return parser.parse_args()


def generate_threshold_ranges(mean, std, max_val, logger):
    """Generate combined threshold ranges from statistical and absolute ranges."""
    logger.info(f"Generating threshold ranges with mean={mean}, std={std}, max={max_val}")

    stat_start = int(mean - std)
    stat_end = int(mean + std)
    stat_range = np.arange(stat_start, stat_end + 1, 1)

    abs_range = np.arange(0, int(max_val) + 1, 10)

    combined_thresholds = np.unique(np.concatenate([stat_range, abs_range])).astype(int)
    combined_thresholds = np.sort(combined_thresholds)

    logger.info(f"Statistical range: [{stat_start:.1f}, {stat_end:.1f}] with {len(stat_range)} points")
    logger.info(f"Absolute range: [0, {max_val:.1f}] with {len(abs_range)} points")
    logger.info(f"Combined range: {len(combined_thresholds)} unique thresholds")

    return combined_thresholds


def setup_environment(args):
    """Setup logging and create output directory."""
    setup_logging()
    logger = logging.getLogger(__name__)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    thresholds = generate_threshold_ranges(args.mean, args.std, args.max, logger)

    return logger, thresholds


def load_model_and_targets(args, logger):
    """Load the trained model and target embeddings."""
    logger.info("Loading model and target embeddings...")

    model_helper = ModelHelper()
    model = model_helper.load_model(args.emb_dir, args.model_name, args.margin)
    embedder = model.layers[3]  # Extract embedding layer

    targetlist_emb, target_labels, target_filenames = load_targetemb(
        args.emb_dir / "whitelist_emb.npy",
        args.emb_dir / "whitelist_labels.npy",
        args.emb_dir / "whitelist_file_names.npy",
    )

    logger.info(f"Loaded model with embedder shape: {embedder.output_shape}")
    logger.info(f"Loaded {len(targetlist_emb)} target embeddings")

    return embedder, targetlist_emb, target_labels, target_filenames


def load_validation_datasets(args, embedder, logger):
    """Process validation datasets and compute embeddings."""
    logger.info("Processing validation datasets...")

    logger.info("Processing validation phishing dataset...")
    phish_count, val_phish_emb, val_phish_labels, val_phish_files = process_dataset(
        args.val_phish_dir, args.reshape_size, embedder, batch_size=args.batch_size, logger=logger
    )

    logger.info("Processing validation benign dataset...")
    benign_count, val_benign_emb, val_benign_labels, val_benign_files = process_dataset(
        args.val_benign_dir, args.reshape_size, embedder, batch_size=args.batch_size, logger=logger
    )

    logger.info(f"Processed {phish_count} phishing images")
    logger.info(f"Processed {benign_count} benign images")

    return (val_phish_emb, val_phish_labels, val_phish_files, val_benign_emb, val_benign_labels, val_benign_files)


def compute_distances(val_phish_emb, val_benign_emb, targetlist_emb, batch_size, logger):
    """Compute minimum distances from validation samples to target embeddings."""
    logger.info("Computing pairwise distances...")

    val_combined_emb = np.vstack([val_benign_emb, val_phish_emb])
    logger.info(f"Combined validation embeddings shape: {val_combined_emb.shape}")

    pairwise_distances = Evaluate.compute_all_distances_batched(val_combined_emb, targetlist_emb, batch_size=batch_size)
    min_distances = np.min(pairwise_distances, axis=1)

    num_benign = val_benign_emb.shape[0]
    benign_min_distances = min_distances[:num_benign]
    phish_min_distances = min_distances[num_benign:]

    logger.info(f"Computed distances - Benign: {len(benign_min_distances)}, Phishing: {len(phish_min_distances)}")

    return benign_min_distances, phish_min_distances


def save_embeddings_data(output_dir, embeddings_data, config_hash, logger):
    """Save computed embeddings to disk with configuration validation."""
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(exist_ok=True)

    config_file = cache_dir / "config.json"
    with open(config_file, "w") as f:
        json.dump({"config_hash": config_hash}, f)

    cache_file = cache_dir / "embeddings_data.npz"
    np.savez_compressed(
        cache_file,
        targetlist_emb=embeddings_data["targetlist_emb"],
        target_labels=embeddings_data["target_labels"],
        target_filenames=embeddings_data["target_filenames"],
        val_phish_emb=embeddings_data["val_phish_emb"],
        val_phish_labels=embeddings_data["val_phish_labels"],
        val_phish_files=embeddings_data["val_phish_files"],
        val_benign_emb=embeddings_data["val_benign_emb"],
        val_benign_labels=embeddings_data["val_benign_labels"],
        val_benign_files=embeddings_data["val_benign_files"],
    )
    logger.info(f"Saved embeddings data to {cache_file}")


def load_embeddings_data(output_dir, current_config_hash, logger):
    """Load previously computed embeddings if they match current configuration."""
    cache_dir = output_dir / "cache"
    cache_file = cache_dir / "embeddings_data.npz"
    config_file = cache_dir / "config.json"

    if not cache_file.exists() or not config_file.exists():
        logger.info("Cache files not found, will compute embeddings from scratch")
        return None

    try:
        with open(config_file, "r") as f:
            cached_config = json.load(f)

        if cached_config.get("config_hash") != current_config_hash:
            logger.warning("Configuration changed, invalidating cache")
            return None

        npz_data = np.load(cache_file, allow_pickle=True)
        embeddings_data = {
            "targetlist_emb": npz_data["targetlist_emb"],
            "target_labels": npz_data["target_labels"],
            "target_filenames": npz_data["target_filenames"],
            "val_phish_emb": npz_data["val_phish_emb"],
            "val_phish_labels": npz_data["val_phish_labels"],
            "val_phish_files": npz_data["val_phish_files"],
            "val_benign_emb": npz_data["val_benign_emb"],
            "val_benign_labels": npz_data["val_benign_labels"],
            "val_benign_files": npz_data["val_benign_files"],
        }

        logger.info("Using cached embeddings")
        return embeddings_data

    except Exception as e:
        logger.error(f"Cache load failed: {e}")
        return None


def get_config_hash(args):
    """Generate simple hash of key configuration parameters."""
    config_str = (
        f"{args.model_name}_{args.margin}_{args.val_phish_dir.name}_{args.val_benign_dir.name}_{args.emb_dir.name}"
    )
    return str(hash(config_str))


def analyze_distributions(benign_min_distances, phish_min_distances, logger):
    """Analyze distance distributions using Gaussian PDF fitting."""
    stats = _get_scipy_stats()
    logger.info("Analyzing distance distributions with Gaussian PDF fitting...")

    benign_mu, benign_sigma = stats.norm.fit(benign_min_distances)
    phish_mu, phish_sigma = stats.norm.fit(phish_min_distances)

    logger.info(f"Benign distribution: μ={benign_mu:.3f}, σ={benign_sigma:.3f}")
    logger.info(f"Phishing distribution: μ={phish_mu:.3f}, σ={phish_sigma:.3f}")

    return {"benign_mu": benign_mu, "benign_sigma": benign_sigma, "phish_mu": phish_mu, "phish_sigma": phish_sigma}


def find_eer_threshold(benign_min_distances, phish_min_distances, thresholds, logger):
    """Find the top 3 Equal Error Rate thresholds where FPR ≈ FNR."""
    logger.info(f"Finding EER thresholds across {len(thresholds)} threshold values...")

    results = []

    for threshold in thresholds:
        # Calculate False Positive Rate (FPR)
        # FPR: proportion of benign samples with distance ≤ threshold (incorrectly classified as phishing)
        false_positives = np.sum(benign_min_distances <= threshold)
        fpr = false_positives / len(benign_min_distances)

        # Calculate False Negative Rate (FNR)
        # FNR: proportion of phishing samples with distance > threshold (incorrectly classified as benign)
        false_negatives = np.sum(phish_min_distances > threshold)
        fnr = false_negatives / len(phish_min_distances)

        # Calculate other metrics for completeness
        true_positives = np.sum(phish_min_distances <= threshold)
        true_negatives = np.sum(benign_min_distances > threshold)

        tpr = true_positives / len(phish_min_distances)  # Recall/Sensitivity
        tnr = true_negatives / len(benign_min_distances)  # Specificity

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        accuracy = (true_positives + true_negatives) / (len(benign_min_distances) + len(phish_min_distances))
        f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0

        # Find threshold where FPR ≈ FNR (EER point)
        eer_diff = abs(fpr - fnr)
        eer_value = (fpr + fnr) / 2

        metrics = {
            "threshold": threshold,
            "fpr": fpr,
            "fnr": fnr,
            "tpr": tpr,
            "tnr": tnr,
            "precision": precision,
            "recall": tpr,
            "accuracy": accuracy,
            "f1": f1,
            "eer_diff": eer_diff,
            "eer_value": eer_value,
        }

        results.append(metrics)

    results_sorted = sorted(results, key=lambda x: x["eer_diff"])
    top_3_thresholds = results_sorted[:3]

    logger.info("Top 3 EER thresholds found:")
    for i, metrics in enumerate(top_3_thresholds, 1):
        logger.info(
            f"  #{i}: Threshold={metrics['threshold']:.3f}, "
            f"EER={metrics['eer_value']:.4f} "
            f"(FPR={metrics['fpr']:.4f}, FNR={metrics['fnr']:.4f})"
        )

    return top_3_thresholds, results


def generate_plots(
    benign_min_distances, phish_min_distances, distribution_params, optimal_threshold, all_results, output_dir, logger
):
    """Generate statistical plots if --plot flag is enabled."""
    stats = _get_scipy_stats()
    logger.info("Generating plots...")

    plt.figure(figsize=(12, 8))
    bins = np.linspace(0, max(np.max(benign_min_distances), np.max(phish_min_distances)), 50)
    plt.hist(
        benign_min_distances, bins=bins, density=True, alpha=0.6, label="Benign Sites", color="blue", edgecolor="black"
    )
    plt.hist(
        phish_min_distances, bins=bins, density=True, alpha=0.6, label="Phishing Sites", color="red", edgecolor="black"
    )

    x_range = np.linspace(0, max(bins), 200)

    benign_pdf = stats.norm.pdf(x_range, distribution_params["benign_mu"], distribution_params["benign_sigma"])
    phish_pdf = stats.norm.pdf(x_range, distribution_params["phish_mu"], distribution_params["phish_sigma"])

    plt.plot(
        x_range,
        benign_pdf,
        "b-",
        linewidth=2,
        label=f"Benign Fit (μ={distribution_params['benign_mu']:.2f}, σ={distribution_params['benign_sigma']:.2f})",
    )
    plt.plot(
        x_range,
        phish_pdf,
        "r-",
        linewidth=2,
        label=f"Phishing Fit (μ={distribution_params['phish_mu']:.2f}, σ={distribution_params['phish_sigma']:.2f})",
    )

    plt.axvline(
        x=optimal_threshold,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Optimal EER Threshold ({optimal_threshold:.2f})",
    )

    plt.xlabel("Distance to Closest Target")
    plt.ylabel("Density")
    plt.title("Distance Distribution with Gaussian Fits and Optimal EER Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    density_plot_path = output_dir / "density_histogram.png"
    plt.tight_layout()
    plt.savefig(density_plot_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    logger.info(f"Saved density histogram to {density_plot_path}")

    plt.figure(figsize=(12, 8))
    thresholds = [r["threshold"] for r in all_results]
    fprs = [r["fpr"] for r in all_results]
    fnrs = [r["fnr"] for r in all_results]

    plt.plot(thresholds, fprs, "b-", linewidth=2, label="False Positive Rate (FPR)")
    plt.plot(thresholds, fnrs, "r-", linewidth=2, label="False Negative Rate (FNR)")

    optimal_result = next(r for r in all_results if r["threshold"] == optimal_threshold)
    plt.plot(
        optimal_threshold,
        optimal_result["fpr"],
        "go",
        markersize=10,
        label=f"EER Point (FPR={optimal_result['fpr']:.3f}, FNR={optimal_result['fnr']:.3f})",
    )

    plt.xlabel("Distance Threshold")
    plt.ylabel("Error Rate")
    plt.title("Equal Error Rate Analysis: FPR and FNR vs Threshold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    eer_plot_path = output_dir / "eer_curve.png"
    plt.tight_layout()
    plt.savefig(eer_plot_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()
    logger.info(f"Saved EER curve to {eer_plot_path}")

    return density_plot_path, eer_plot_path


def convert_numpy_types(obj):
    """Convert NumPy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_results(optimal_metrics, all_results, output_dir, logger):
    """Save results in enhanced JSON format matching plan specification."""
    results = {
        "optimal_threshold": optimal_metrics["threshold"],
        "equal_error_rate": optimal_metrics["eer_value"],
        "validation_metrics": {
            "accuracy": optimal_metrics["accuracy"],
            "precision": optimal_metrics["precision"],
            "recall": optimal_metrics["recall"],
            "f1": optimal_metrics["f1"],
            "fpr": optimal_metrics["fpr"],
            "fnr": optimal_metrics["fnr"],
        },
    }

    # Convert NumPy types to native Python types for JSON serialization
    results = convert_numpy_types(results)

    optimal_threshold_file = output_dir / "optimal_threshold.json"
    with open(optimal_threshold_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {optimal_threshold_file}")

    threshold_sweep_file = output_dir / "threshold_sweep.csv"
    pd.DataFrame(all_results).to_csv(threshold_sweep_file, index=False)
    logger.info(f"Saved sweep data to {threshold_sweep_file}")

    return optimal_threshold_file, threshold_sweep_file


def main():
    args = parse_arguments()
    logger, thresholds = setup_environment(args)

    run = wandb.init(
        project="VisualPhish-Threshold", name="eer", config=args, tags=["validation", "threshold-optimization"]
    )

    try:
        logger.info("Checking for cached embeddings...")
        config_hash = get_config_hash(args)
        embeddings_data = load_embeddings_data(args.output_dir, config_hash, logger)

        if embeddings_data is not None:
            logger.info("Using cached embeddings")
            targetlist_emb = embeddings_data["targetlist_emb"]
            target_labels = embeddings_data["target_labels"]
            target_filenames = embeddings_data["target_filenames"]
            val_phish_emb = embeddings_data["val_phish_emb"]
            val_phish_labels = embeddings_data["val_phish_labels"]
            val_phish_files = embeddings_data["val_phish_files"]
            val_benign_emb = embeddings_data["val_benign_emb"]
            val_benign_labels = embeddings_data["val_benign_labels"]
            val_benign_files = embeddings_data["val_benign_files"]
        else:
            logger.info("Computing embeddings from scratch...")
            embedder, targetlist_emb, target_labels, target_filenames = load_model_and_targets(args, logger)
            (val_phish_emb, val_phish_labels, val_phish_files, val_benign_emb, val_benign_labels, val_benign_files) = (
                load_validation_datasets(args, embedder, logger)
            )

            embeddings_data = {
                "targetlist_emb": targetlist_emb,
                "target_labels": target_labels,
                "target_filenames": target_filenames,
                "val_phish_emb": val_phish_emb,
                "val_phish_labels": val_phish_labels,
                "val_phish_files": val_phish_files,
                "val_benign_emb": val_benign_emb,
                "val_benign_labels": val_benign_labels,
                "val_benign_files": val_benign_files,
            }
            save_embeddings_data(args.output_dir, embeddings_data, config_hash, logger)

        logger.info("Computing distances...")
        benign_min_distances, phish_min_distances = compute_distances(
            val_phish_emb, val_benign_emb, targetlist_emb, args.batch_size, logger
        )

        top_3_thresholds, all_results = find_eer_threshold(
            benign_min_distances, phish_min_distances, thresholds, logger
        )

        optimal_metrics = top_3_thresholds[0]

        distribution_params = None
        density_plot_path, eer_plot_path = None, None
        if args.plot:
            distribution_params = analyze_distributions(benign_min_distances, phish_min_distances, logger)
            density_plot_path, eer_plot_path = generate_plots(
                benign_min_distances,
                phish_min_distances,
                distribution_params,
                optimal_metrics["threshold"],
                all_results,
                args.output_dir,
                logger,
            )

        optimal_threshold_file, threshold_sweep_file = save_results(
            optimal_metrics, all_results, args.output_dir, logger
        )

        for i, metrics in enumerate(top_3_thresholds):
            wandb.log(
                {
                    f"top_{i + 1}_threshold": metrics["threshold"],
                    f"top_{i + 1}_eer": metrics["eer_value"],
                    f"top_{i + 1}_fpr": metrics["fpr"],
                    f"top_{i + 1}_fnr": metrics["fnr"],
                    f"top_{i + 1}_f1": metrics["f1"],
                    f"top_{i + 1}_accuracy": metrics["accuracy"],
                }
            )

        wandb.log(
            {
                "optimal_threshold": optimal_metrics["threshold"],
                "equal_error_rate": optimal_metrics["eer_value"],
                "optimal_accuracy": optimal_metrics["accuracy"],
                "optimal_f1": optimal_metrics["f1"],
            }
        )

        if distribution_params:
            wandb.log(distribution_params)

        run.log_artifact(
            wandb.Artifact("optimal_threshold", type="result", description="Optimal threshold configuration"),
            str(optimal_threshold_file),
        )

        run.log_artifact(
            wandb.Artifact("threshold_sweep", type="data", description="Complete threshold sweep results"),
            str(threshold_sweep_file),
        )

        if args.plot and density_plot_path and eer_plot_path:
            run.log_artifact(
                wandb.Artifact(
                    "density_histogram", type="plot", description="Distance distribution with Gaussian fits"
                ),
                str(density_plot_path),
            )

            run.log_artifact(
                wandb.Artifact("eer_curve", type="plot", description="EER analysis curve"), str(eer_plot_path)
            )

    except Exception as e:
        logger.error(f"Error during execution: {e}")
        raise
    finally:
        run.finish()


if __name__ == "__main__":
    main()
