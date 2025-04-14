import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve


def load_data(file_path):
    """Load data from CSV file."""
    return pd.read_csv(file_path)


def plot_confusion_matrix(df, output_path=None):
    """Plot confusion matrix."""
    # Extract true and predicted classes
    y_true = df['true_class'].values
    y_pred = df['vp_class'].values

    # Get unique class labels
    labels = np.unique(np.concatenate([y_true, y_pred]))

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    if output_path:
        plt.savefig(f"{output_path}/confusion_matrix.png", bbox_inches='tight')

    plt.show()

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, labels=labels))


def plot_distance_distribution(df, output_path=None):
    """Plot distance score distribution by class."""
    plt.figure(figsize=(12, 6))

    # Convert true_class to string for better visualization
    df['true_class_str'] = df['true_class'].astype(str)

    sns.histplot(data=df, x='vp_distance', hue='true_class_str',
                 bins=30, kde=True, element='step', common_norm=False)

    plt.title('Distance Score Distribution by True Class')
    plt.xlabel('Prediction Distance')
    plt.ylabel('Count')

    if output_path:
        plt.savefig(f"{output_path}/distance_distribution.png", bbox_inches='tight')

    plt.show()


def plot_roc_curve(df, output_path=None):
    """Plot ROC curve if binary classification."""
    # Check if binary classification
    unique_classes = df['true_class'].nunique()

    if unique_classes == 2:
        y_true = df['true_class'].values
        # Use vp_class for predictions
        y_pred = df['vp_class'].values

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
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')

        if output_path:
            plt.savefig(f"{output_path}/roc_curve.png", bbox_inches='tight')

        plt.show()
    else:
        print("ROC curve is only applicable for binary classification tasks.")


def plot_precision_recall_curve(df, output_path=None):
    """Plot precision-recall curve if binary classification."""
    # Check if binary classification
    unique_classes = df['true_class'].nunique()

    if unique_classes == 2:
        y_true = df['true_class'].values
        # Use vp_class for predictions
        y_pred = df['vp_class'].values

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
        plt.plot(recall, precision, color='blue', lw=2,
                 label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')

        if output_path:
            plt.savefig(f"{output_path}/precision_recall_curve.png", bbox_inches='tight')

        plt.show()
    else:
        print("Precision-Recall curve is only applicable for binary classification tasks.")


def plot_error_analysis(df, output_path=None):
    """Plot misclassified examples with distance."""
    # Create a boolean mask for misclassified examples
    misclassified = df['true_class'] != df['vp_class']

    if misclassified.sum() > 0:
        misclassified_df = df[misclassified].copy()

        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(misclassified_df)), misclassified_df['vp_distance'],
                    c=misclassified_df['true_class'], cmap='viridis', alpha=0.6)

        plt.title('Distance Scores for Misclassified Examples')
        plt.xlabel('Example Index')
        plt.ylabel('Prediction Distance')
        plt.colorbar(label='True Class')

        if output_path:
            plt.savefig(f"{output_path}/misclassified_distance.png", bbox_inches='tight')

        plt.show()

        # Print some statistics about misclassified examples
        print(f"\nMisclassification Analysis:")
        print(f"Total examples: {len(df)}")
        print(f"Misclassified examples: {misclassified.sum()} ({misclassified.sum() / len(df):.2%})")

        # Group by true class and predicted class
        error_types = misclassified_df.groupby(['true_class', 'vp_class']).size()
        print("\nError Types:")
        print(error_types)
    else:
        print("No misclassified examples found!")


def plot_target_analysis(df, output_path=None):
    """Plot target prediction analysis."""
    # Create a boolean mask for examples where target prediction is available
    has_target = ~df['vp_target'].isna() & ~df['true_target'].isna()

    if has_target.sum() > 0:
        target_df = df[has_target].copy()

        # Check if targets match
        target_df['target_match'] = target_df['vp_target'] == target_df['true_target']

        plt.figure(figsize=(10, 6))

        # Group by class and target match
        group_data = target_df.groupby(['true_class', 'target_match']).size().unstack(fill_value=0)

        # Plot stacked bar chart
        group_data.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.title('Target Prediction Accuracy by Class')
        plt.xlabel('True Class')
        plt.ylabel('Count')
        plt.legend(title='Target Match')

        if output_path:
            plt.savefig(f"{output_path}/target_analysis.png", bbox_inches='tight')

        plt.show()

        # Print some statistics about target predictions
        print(f"\nTarget Prediction Analysis:")
        print(f"Total examples with target info: {has_target.sum()}")
        print(f"Target match rate: {target_df['target_match'].mean():.2%}")

        # Group by true class
        by_class = target_df.groupby('true_class')['target_match'].agg(['mean', 'count'])
        print("\nTarget Match Rate by Class:")
        print(by_class)
    else:
        print("No target prediction information found!")


def plot_distance_threshold_analysis(df, output_path=None):
    """Plot how different distance thresholds affect classification metrics."""

    # We'll use distance as a threshold - lower distance means higher confidence
    thresholds = np.linspace(df['vp_distance'].min(), df['vp_distance'].max(), 100)

    # Arrays to store metrics
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []

    y_true = df['true_class'].values

    # Convert to binary if needed
    if df['true_class'].nunique() == 2:
        # Map classes to 0 and 1
        classes = sorted(df['true_class'].unique())
        class_map = {cls: i for i, cls in enumerate(classes)}
        y_true_bin = np.array([class_map[cls] for cls in y_true])

        for threshold in thresholds:
            # Predict positive class if distance is below threshold
            y_pred = (df['vp_distance'] <= threshold).astype(int)

            # Calculate metrics
            tp = ((y_pred == 1) & (y_true_bin == 1)).sum()
            fp = ((y_pred == 1) & (y_true_bin == 0)).sum()
            tn = ((y_pred == 0) & (y_true_bin == 0)).sum()
            fn = ((y_pred == 0) & (y_true_bin == 1)).sum()

            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            accuracies.append(accuracy)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        # Plot metrics
        plt.figure(figsize=(12, 8))
        plt.plot(thresholds, accuracies, label='Accuracy')
        plt.plot(thresholds, precisions, label='Precision')
        plt.plot(thresholds, recalls, label='Recall')
        plt.plot(thresholds, f1_scores, label='F1 Score')

        plt.xlabel('Distance Threshold')
        plt.ylabel('Metric Value')
        plt.title('Classification Metrics vs. Distance Threshold')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)

        if output_path:
            plt.savefig(f"{output_path}/threshold_analysis.png", bbox_inches='tight')

        plt.show()
    else:
        print("Threshold analysis is only implemented for binary classification.")


def main():
    parser = argparse.ArgumentParser(description='Visualize classification results from CSV')
    parser.add_argument('file_path', type=str, help='Path to CSV file with classification results')
    parser.add_argument('--output', type=str, help='Output directory for saving plots', default=None)

    args = parser.parse_args()

    # Load data
    df = load_data(args.file_path)

    # Fill NaN values in vp_class with a default value if needed
    if df['vp_class'].isna().any():
        print("Filling NaN values in vp_class with default value (benign)")
        df['vp_class'] = df['vp_class'].fillna('benign')

    # Display summary info
    print(f"Loaded dataset with {len(df)} rows")
    print("\nData summary:")
    print(df.head())
    print("\nClass distribution:")
    print(df['true_class'].value_counts())

    # Plot confusion matrix
    plot_confusion_matrix(df, args.output)

    # Plot distance distribution (replaced confidence with distance)
    plot_distance_distribution(df, args.output)

    # Plot ROC curve (if binary classification)
    plot_roc_curve(df, args.output)

    # Plot precision-recall curve (if binary classification)
    plot_precision_recall_curve(df, args.output)

    # Plot error analysis
    plot_error_analysis(df, args.output)

    # New visualizations for the new dataset

    # Target analysis
    plot_target_analysis(df, args.output)

    # Distance threshold analysis
    plot_distance_threshold_analysis(df, args.output)


if __name__ == "__main__":
    main()