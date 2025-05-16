import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve,
)


def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)


def plot_confusion_matrix(df, output_path=None):
    """Plot confusion matrix."""
    # Extract true and predicted classes
    y_true = df["true_class"].values
    y_pred = df["pp_class"].values

    # Get unique class labels
    labels = np.unique(np.concatenate([y_true, y_pred]))

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if output_path:
        plt.savefig(f"{output_path}/confusion_matrix.png", bbox_inches="tight")

    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels))


def plot_confidence_distribution(df, output_path=None):
    """Plot confidence score distribution by class."""
    plt.figure(figsize=(12, 6))

    # Convert true_class to string for better visualization
    df["true_class_str"] = df["true_class"].astype(str)

    sns.histplot(
        data=df,
        x="pp_conf",
        hue="true_class_str",
        bins=30,
        kde=True,
        element="step",
        common_norm=False,
    )

    plt.title("Confidence Score Distribution by True Class")
    plt.xlabel("Prediction Confidence")
    plt.ylabel("Count")

    if output_path:
        plt.savefig(f"{output_path}/confidence_distribution.png", bbox_inches="tight")

    plt.show()


def plot_roc_curve(df, output_path=None):
    """Plot ROC curve if binary classification."""
    # Check if binary classification
    unique_classes = df["true_class"].nunique()

    if unique_classes == 2:
        y_true = df["true_class"].values
        # Use pp_class instead of pp_conf for predictions
        y_pred = df["pp_class"].values

        # Convert string labels to binary if needed
        if isinstance(y_true[0], str) or isinstance(y_pred[0], str):
            # Map unique values to 0, 1 (ensure consistency between true and pred)
            unique_values = np.unique(np.concatenate([y_true, y_pred]))
            value_map = {val: i for i, val in enumerate(unique_values)}
            y_true_bin = np.array([value_map[val] for val in y_true])
            y_pred_bin = np.array([value_map[val] for val in y_pred])
            fpr, tpr, _ = roc_curve(y_true_bin, y_pred_bin)
        else:
            fpr, tpr, _ = roc_curve(y_true, y_pred)

        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 8))
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=2,
            label=f"ROC curve (area = {roc_auc:.2f})",
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")

        if output_path:
            plt.savefig(f"{output_path}/roc_curve.png", bbox_inches="tight")

        plt.show()
    else:
        print("ROC curve is only applicable for binary classification tasks.")


def plot_precision_recall_curve(df, output_path=None):
    """Plot precision-recall curve if binary classification."""
    # Check if binary classification
    unique_classes = df["true_class"].nunique()

    if unique_classes == 2:
        y_true = df["true_class"].values
        # Use pp_class instead of pp_conf for predictions
        y_pred = df["pp_class"].values

        # Convert string labels to binary if needed
        if isinstance(y_true[0], str) or isinstance(y_pred[0], str):
            # Map unique values to 0, 1 (ensure consistency between true and pred)
            unique_values = np.unique(np.concatenate([y_true, y_pred]))
            value_map = {val: i for i, val in enumerate(unique_values)}
            y_true_bin = np.array([value_map[val] for val in y_true])
            y_pred_bin = np.array([value_map[val] for val in y_pred])
            precision, recall, _ = precision_recall_curve(y_true_bin, y_pred_bin)
        else:
            precision, recall, _ = precision_recall_curve(y_true, y_pred)

        pr_auc = auc(recall, precision)

        plt.figure(figsize=(8, 8))
        plt.plot(
            recall,
            precision,
            color="blue",
            lw=2,
            label=f"PR curve (area = {pr_auc:.2f})",
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend(loc="lower left")

        if output_path:
            plt.savefig(
                f"{output_path}/precision_recall_curve.png", bbox_inches="tight"
            )

        plt.show()
    else:
        print(
            "Precision-Recall curve is only applicable for binary classification tasks."
        )


def plot_error_analysis(df, output_path=None):
    """Plot misclassified examples with confidence."""
    # Create a boolean mask for misclassified examples
    misclassified = df["true_class"] != df["pp_class"]

    if misclassified.sum() > 0:
        misclassified_df = df[misclassified].copy()

        plt.figure(figsize=(10, 6))
        plt.scatter(
            range(len(misclassified_df)),
            misclassified_df["pp_conf"],
            c=misclassified_df["true_class"],
            cmap="viridis",
            alpha=0.6,
        )

        plt.title("Confidence Scores for Misclassified Examples")
        plt.xlabel("Example Index")
        plt.ylabel("Prediction Confidence")
        plt.colorbar(label="True Class")

        if output_path:
            plt.savefig(
                f"{output_path}/misclassified_confidence.png", bbox_inches="tight"
            )

        plt.show()

        # Print some statistics about misclassified examples
        print("\nMisclassification Analysis:")
        print(f"Total examples: {len(df)}")
        print(
            f"Misclassified examples: {misclassified.sum()} ({misclassified.sum() / len(df):.2%})"
        )

        # Group by true class and predicted class
        error_types = misclassified_df.groupby(["true_class", "pp_class"]).size()
        print("\nError Types:")
        print(error_types)
    else:
        print("No misclassified examples found!")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize classification results from CSV"
    )
    parser.add_argument(
        "file_path", type=str, help="Path to CSV file with classification results"
    )
    parser.add_argument(
        "--output", type=str, help="Output directory for saving plots", default=None
    )

    args = parser.parse_args()

    # Load data
    df = load_data(args.file_path)

    # Fill NaN values in pp_class with 'benign'
    df["pp_class"] = df["pp_class"].fillna("benign")

    # Display summary info
    print(f"Loaded dataset with {len(df)} rows")
    print("\nData summary:")
    print(df.head())
    print("\nClass distribution:")
    print(df["true_class"].value_counts())

    # Plot confusion matrix
    plot_confusion_matrix(df, args.output)

    # Plot confidence distribution
    plot_confidence_distribution(df, args.output)

    # Plot ROC curve (if binary classification)
    # plot_roc_curve(df, args.output)

    # Plot precision-recall curve (if binary classification)
    plot_precision_recall_curve(df, args.output)

    # Plot error analysis
    plot_error_analysis(df, args.output)


if __name__ == "__main__":
    main()
