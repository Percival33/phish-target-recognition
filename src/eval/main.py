"""
Main orchestration module for the eval package.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd
from tools.config import setup_logging
from config_loader import load_config, ConfigError, Config
from data_processor import process_all_data, DataProcessingError
from analysis import (
    create_performance_dataframe,
    calculate_rankings,
    perform_friedman_test,
    create_summary_statistics,
)
from reporting import (
    create_formatted_results_table,
    save_table_as_csv,
    save_table_as_image,
    create_output_directory,
)
from plot_generator import create_and_save_cd_diagram


def _process_data(config: Config) -> tuple:
    """Process data and create analysis DataFrames."""
    scores_matrix = process_all_data(config)
    scores_df = create_performance_dataframe(scores_matrix, config)
    rankings_df = calculate_rankings(scores_df, config)
    return scores_matrix, scores_df, rankings_df


def _save_outputs(config: Config, formatted_table: pd.DataFrame) -> Path:
    """Save all output files and return output directory."""
    output_dir = Path(config.output_config.results_directory)
    create_output_directory(output_dir)

    save_table_as_csv(
        formatted_table, output_dir / config.output_config.summary_csv_filename
    )
    save_table_as_image(
        formatted_table,
        output_dir / config.output_config.ranked_table_image_filename,
        config,
    )

    return output_dir


def _create_plots(scores_matrix, rankings_df, config: Config, output_dir: Path) -> None:
    """Create and save plots."""
    cd_plot_path = output_dir / config.output_config.critical_difference_plot_filename
    create_and_save_cd_diagram(scores_matrix, config, cd_plot_path)


def _print_summary(
    config: Config,
    summary_stats: pd.DataFrame,
    friedman_stat: float,
    friedman_p: float,
    output_dir: Path,
) -> None:
    """Print evaluation summary."""
    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")

    print(f"Metric: {config.metric_config.display_name}")
    print(f"Datasets: {', '.join(config.dataset_names)}")
    print(f"Algorithms: {', '.join(config.algorithm_names)}")

    print("\nFriedman Test Results:")
    print(f"  Statistic: {friedman_stat:.4f}")
    print(f"  P-value: {friedman_p:.6f}")
    print(f"  Significant: {'Yes' if friedman_p < 0.05 else 'No'} (α = 0.05)")

    print("\nAlgorithm Rankings (Mean Rank):")
    for algorithm, mean_rank in summary_stats["Mean_Rank"].items():
        print(f"  {algorithm}: {mean_rank:.2f}")

    print(f"\nOutput files saved to: {output_dir}")
    print(f"{'=' * 60}")


def run_full_evaluation(config_file_path: str) -> None:
    """Run the complete evaluation pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        config = load_config(config_file_path)
        scores_matrix, scores_df, rankings_df = _process_data(config)

        friedman_stat, friedman_p = perform_friedman_test(rankings_df)
        summary_stats = create_summary_statistics(scores_df, rankings_df)
        formatted_table = create_formatted_results_table(scores_df, rankings_df, config)

        output_dir = _save_outputs(config, formatted_table)
        _create_plots(scores_matrix, rankings_df, config, output_dir)
        _print_summary(config, summary_stats, friedman_stat, friedman_p, output_dir)

        logger.info(
            f"Evaluation completed successfully. Results saved to: {output_dir}"
        )

    except (ConfigError, DataProcessingError) as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run model performance evaluation pipeline with critical difference analysis."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration JSON file (default: config.json)",
    )

    args = parser.parse_args()
    run_full_evaluation(args.config)
