import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import pandas as pd
import seaborn as sns


def imshow(img_path: str, gray=False):
    img = cv2.imread(img_path)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap="gray") if gray or len(
        img.shape
    ) == 2 else plt.imshow(img)


def formatted_name(img_path):
    return "-".join(img_path.split("/")[-1].split())


def compute_accuracy(y_true, y_pred) -> tuple[float, float]:
    """
    Compute accuracy and balanced accuracy
    :param y_true:
    :param y_pred:
    :return: tuple of accuracy and balanced accuracy
    """
    accuracy = accuracy_score(y_true, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    print(f"acc:\t\t\t{accuracy * 100 :.5f}")
    print(f"balanced_acc:\t{balanced_accuracy * 100:.5f}")
    return accuracy, balanced_accuracy


def visualize(y_true, y_pred):
    """
    Visualize the distribution of predictions and ground truth
    :param y_true:
    :param y_pred:
    :return: dataframe containing the data
    """

    data = pd.DataFrame(
        {
            "Type": ["Prediction"] * len(y_pred) + ["Ground Truth"] * len(y_true),
            "Company": y_pred + y_true,
        }
    )

    plt.figure(figsize=(12, 6))
    sns.countplot(data=data, x="Company", hue="Type", palette="viridis")

    plt.title("Distribution of Predictions and Ground Truth", fontsize=16)
    plt.xlabel("Company", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Data Type", fontsize=12)
    plt.tight_layout()

    plt.show()

    return data
