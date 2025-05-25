"""
Statistical analysis module for ranking and Friedman test.
"""

import pandas as pd
import numpy as np
from scipy.stats import friedmanchisquare
from typing import Tuple

from config_loader import Config


def create_performance_dataframe(
    scores_matrix: np.ndarray, config: Config
) -> pd.DataFrame:
    return pd.DataFrame(
        scores_matrix, index=config.dataset_names, columns=config.algorithm_names
    )


def calculate_rankings(scores_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    higher_is_better = config.metric_config.higher_scores_are_better
    return scores_df.rank(axis=1, ascending=not higher_is_better, method="min")


def perform_friedman_test(rankings_df: pd.DataFrame) -> Tuple[float, float]:
    return friedmanchisquare(*rankings_df.T.values)


def create_summary_statistics(
    scores_df: pd.DataFrame, rankings_df: pd.DataFrame
) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "Mean_Score": scores_df.mean(),
            "Std_Score": scores_df.std(),
            "Mean_Rank": rankings_df.mean(),
            "Sum_Ranks": rankings_df.sum(),
        }
    )
    return summary.sort_values("Mean_Rank")
