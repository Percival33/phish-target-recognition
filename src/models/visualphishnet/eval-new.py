import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
from sklearn.metrics import roc_auc_score, roc_curve
from tools.config import LOGS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, setup_logging
from tqdm import tqdm

from DataHelper import read_image
from Evaluate import Evaluate
from ModelHelper import ModelHelper


# load targetlist embedding
def load_targetemb(emb_path, label_path, file_name_path):
    """
    load targetlist embedding
    :return:
    """
    targetlist_emb = np.load(emb_path)
    all_labels = np.load(label_path)
    all_file_names = np.load(file_name_path)
    return targetlist_emb, all_labels, all_file_names


def read_data(data_path, reshape_size):
    """
    read data
    :param data_path:
    :param reshape_size:
    :param chunk_range: Tuple
    :return:
    """
    all_imgs = []
    all_labels = []
    all_file_names = []

    if (data_path / "labels.txt").exists():
        with open(data_path / "labels.txt", "r") as f:
            for line in f:
                all_labels.append(line.strip())
    else:
        all_labels = ["benign"] * len(list(data_path.iterdir()))

    for file_path in tqdm(sorted(data_path.iterdir())):
        if file_path.suffix == ".txt":
            continue
        img = read_image(file_path, logger)

        if img is None:
            img = read_image(file_path, logger, format="jpeg")

        if img is None:
            logger.error(f"Failed to process {file_path}")
            exit(1)
        all_imgs.append(resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True))
        all_file_names.append(file_path.name)

    all_imgs = np.asarray(all_imgs)
    all_labels = np.asarray(all_labels)
    return all_imgs, all_labels, all_file_names


# L2 distance
def compute_distance_pair(layer1, layer2, targetlist_emb):
    diff = layer1 - layer2
    l2_diff = np.sum(diff**2) / targetlist_emb.shape[1]
    return l2_diff


def compute_all_distances(test_matrix, targetlist_emb):
    train_size = targetlist_emb.shape[0]
    pairwise_distance = np.zeros([test_matrix.shape[0], train_size])

    for i in tqdm(range(test_matrix.shape[0]), desc="Computing pairwise distances"):
        pair1 = test_matrix[i, :]
        for j in range(0, train_size):
            pair2 = targetlist_emb[j, :]
            l2_diff = compute_distance_pair(pair1, pair2, targetlist_emb)
            pairwise_distance[i, j] = l2_diff

    return pairwise_distance


def compute_all_distances_batched(test_matrix, targetlist_emb, batch_size=256):
    """
    Computes pairwise L2 distances between test_matrix and targetlist_emb in a batch-wise manner.
    """
    train_size = targetlist_emb.shape[0]
    test_size = test_matrix.shape[0]
    pairwise_distance = np.zeros((test_size, train_size))

    for batch_start in tqdm(range(0, test_size, batch_size), desc="Computing pairwise distances (batched)"):
        batch_end = min(batch_start + batch_size, test_size)
        batch = test_matrix[batch_start:batch_end]

        # Compute pairwise L2 distances efficiently
        diff = batch[:, np.newaxis, :] - targetlist_emb[np.newaxis, :, :]
        l2_diff = np.sum(diff**2, axis=2) / targetlist_emb.shape[1]

        pairwise_distance[batch_start:batch_end] = l2_diff

    return pairwise_distance


# Find names of examples with min distance
def find_names_min_distances(idx, values, all_file_names):
    names_min_distance = ""
    only_names = []
    distances = ""
    for i in range(idx.shape[0]):
        index_min_distance = idx[i]
        names_min_distance = names_min_distance + "Targetlist: " + all_file_names[index_min_distance] + ","
        only_names.append(all_file_names[index_min_distance])
        distances = distances + str(values[i]) + ","

    names_min_distance = names_min_distance[:-1]
    distances = distances[:-1]

    return names_min_distance, only_names, distances


def evaluate_threshold(
    pairwise_distance, data_emb, targetlist_emb, all_file_names, file_names, y, threshold, result_path
):
    """
    Evaluate model performance for a given threshold
    """
    # pairwise_distance = compute_all_distances_batched(data_emb, targetlist_emb)
    """
    0 - benign
    1 - phishing
    """
    y_true = np.array([0 if label == "benign" else 1 for label in y])

    y_pred = np.zeros([len(data_emb), 1])
    n = 1  # Top-1 match

    result_file = result_path / f"results_threshold_{threshold}.log"
    result_file.parent.mkdir(parents=True, exist_ok=True)

    if not result_file.exists():
        result_file.touch()

    for i in tqdm(range(data_emb.shape[0]), desc=f"Processing threshold {threshold}"):
        distances_to_target = pairwise_distance[i, :]
        idx, values = Evaluate.find_min_distances(np.ravel(distances_to_target), n)
        names_min_distance, only_names, min_distances = find_names_min_distances(idx, values, all_file_names)

        # distance lower than threshold ==> report as phishing
        if float(min_distances) <= threshold:
            y_pred[i] = 1

        with open(result_file, "a+", encoding="utf-8", errors="ignore") as f:
            f.write(f"{file_names[i]}\t{y_pred[i]}\t{str(min_distances)}\t{str(only_names[0])}\n")

    auc_score = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1, drop_intermediate=False)

    return auc_score, fpr, tpr


if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)

    """
    file_handler = logging.FileHandler('result.log')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(file_handler)
    """

    parser = ArgumentParser()
    parser.add_argument("--emb-dir", type=Path, default=PROCESSED_DATA_DIR / "VisualPhish")
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR / "VisualPhish")
    parser.add_argument("--margin", type=float, default=2.2)
    parser.add_argument("--saved-model-name", type=str, default="model2")
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--result-path", type=Path, default=LOGS_DIR / "VisualPhish")
    parser.add_argument("--reshape-size", default=[224, 224, 3])

    args = parser.parse_args()
    logger.info("Evaluating VisualPhishNet")

    # load targetlist and model
    targetlist_emb, all_labels, all_file_names = load_targetemb(
        args.emb_dir / "whitelist_emb.npy",
        args.emb_dir / "whitelist_labels.npy",
        args.emb_dir / "whitelist_file_names.npy",
    )
    modelHelper = ModelHelper()
    model = modelHelper.load_model(args.emb_dir, args.saved_model_name, args.margin).layers[3]

    logger.info("Loaded targetlist and model, number of protected target screenshots {}".format(len(targetlist_emb)))

    # read data
    # X_phish, y_phish, file_names_phish = read_data(args.data_dir / "newly_crawled_phishing", args.reshape_size)
    # logger.info("Finish reading data, number of data {}".format(len(X_phish)))
    #
    # X_benign, y_benign, file_names_benign = read_data(args.data_dir / "benign_test", args.reshape_size)
    # logger.info("Finish reading data, number of data {}".format(len(X_benign)))
    #
    # X = np.concatenate((X_benign, X_phish), axis=0)
    # y = np.concatenate((y_benign, y_phish), axis=0)
    # file_names = np.concatenate((file_names_benign, file_names_phish), axis=0)
    # # random shuffle all data as tuples (X, y, file_names)
    # assert len(X) == len(y) == len(file_names)
    # np.random.seed(42)
    # idx = np.random.permutation(len(X))
    # X = X[idx]
    # y = y[idx]
    # file_names = file_names[idx]
    # np.savez_compressed("val_data", X=X, y=y, file_names=file_names)

    val_data = np.load("val_data.npz")
    X = val_data["X"]
    y = val_data["y"]
    file_names = val_data["file_names"]
    logger.info("Finish reading data, number of data {}".format(len(X)))

    # TODO: save X, y, file_names as test_set

    # get embeddings from data
    # data_emb = model.predict(X, batch_size=32)
    # np.save("data_emb", data_emb)
    data_emb = np.load("data_emb.npy")
    # pairwise_distance = Evaluate.compute_all_distances(data_emb, targetlist_emb, np.zeros([1,512]))

    # pairwise_distance_batched = compute_all_distances_batched(data_emb, targetlist_emb)
    # pairwise_distance = compute_all_distances(data_emb, targetlist_emb)
    # assert np.array_equal(pairwise_distance, pairwise_distance_batched)

    # np.save("pairwise_distance", pairwise_distance)
    pairwise_distance = np.load("pairwise_distance.npy")
    logger.info("Finish getting embedding")

    # Test different thresholds
    thresholds = np.arange(3, 8, .25)
    # np.arange(4, 71, 2)
    results = []
    plt.figure(figsize=(10, 6))

    for threshold in thresholds:
        auc_score, fpr, tpr = evaluate_threshold(
            pairwise_distance, data_emb, targetlist_emb, all_file_names, file_names, y, threshold, args.result_path
        )
        results.append({"threshold": threshold, "auc_score": auc_score, "fpr": fpr, "tpr": tpr})
        logger.info(f"Threshold: {threshold}, AUC Score: {auc_score}")
        plt.step(fpr, tpr, where="post", label=f"Threshold={threshold}")
        # plt.plot(fpr, tpr, label=f'Threshold={threshold}')

    # Plot ROC curves
    x = np.linspace(0, 1, 100)
    plt.plot(x, x, "k--", label="Random Classifier (y=x)")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Different Thresholds")
    plt.legend()
    plt.savefig(args.result_path / "roc_curves.png")
    plt.close()

    final_result_path = args.result_path / "threshold_results.txt"
    final_result_path.parent.mkdir(parents=True, exist_ok=True)
    if not final_result_path.exists():
        final_result_path.touch()

    # Save results
    with open(final_result_path, "w") as f:
        for result in results:
            f.write(f"Threshold: {result['threshold']}, AUC Score: {result['auc_score']}\n")

    # y_true = np.zeros([len(X), 1])
    # y_score = np.zeros([len(X), 1])
    #
    # n = 1  # Top-1 match
    # print("Start ")
    # args.result_path.mkdir(parents=True, exist_ok=True)
    # if not args.result_path.exists():
    #     args.result_path.touch()
    #
    # # from 4 to 70 co dwa
    # for i in tqdm(range(data_emb.shape[0])):
    #     distances_to_target = pairwise_distance[i, :]
    #     idx, values = Evaluate.find_min_distances(np.ravel(distances_to_target), n)
    #     names_min_distance, only_names, min_distances = find_names_min_distances(idx, values, all_file_names)
    #     # distance lower than threshold ==> report as phishing
    #     if float(min_distances) <= args.threshold:
    #         phish = 1
    #         y_score[i] = 1
    #     # else it is benign
    #     else:
    #         phish = 0
    #
    #     with open(args.result_path, "a+", encoding="utf-8", errors="ignore") as f:
    #         f.write(f"{file_names[i]}\t{y_score[i]}\t{str(min_distances)}\t{str(only_names[0])}\n")
    #
    # # TODO: closest target => target
    # print(roc_auc_score(y_true, y_score))
    # fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=0)
    #
    # # Print ROC curve
    # plt.plot(fpr, tpr)
    # plt.show()
