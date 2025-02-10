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

    for file_path in tqdm(data_path.iterdir()):
        img = read_image(file_path, logger)

        if img is None:
            img = read_image(file_path, logger, format="jpeg")

        if img is None:
            logger.error(f"Failed to process {file_path}")
            exit(1)
        all_imgs.append(resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True))
        all_labels.append("benign")
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

    for i in tqdm(range(test_matrix.shape[0])):  # every instance in test_matrix
        pair1 = test_matrix[i, :]
        for j in range(0, train_size):
            pair2 = targetlist_emb[j, :]
            l2_diff = compute_distance_pair(pair1, pair2, targetlist_emb)
            pairwise_distance[i, j] = l2_diff

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
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR / "VisualPhish" / "benign_test")
    parser.add_argument("--margin", type=float, default=2.2)
    parser.add_argument("--saved-model-name", type=str, default="model2")
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--result-path", type=Path, default=LOGS_DIR / "VisualPhish" / "result.log")
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
    # X, y, file_names = read_data(args.data_dir, args.reshape_size)
    # logger.info("Finish reading data, number of data {}".format(len(X)))

    # np.save('val_imgs', X)
    # np.save('val_labels', y)
    # np.save('val_file_names', file_names)

    X = np.load("val_imgs.npy")
    y = np.load("val_labels.npy")
    file_names = np.load("val_file_names.npy")
    logger.info("Finish reading data, number of data {}".format(len(X)))

    # TODO: save X, y, file_names as test_set

    # get embeddings from data
    data_emb = model.predict(X, batch_size=32)
    # pairwise_distance = Evaluate.compute_all_distances(data_emb, targetlist_emb, np.zeros([1, 512]))
    # assert np.array_equal(pairwise_distance, compute_all_distances(data_emb, targetlist_emb))
    pairwise_distance = compute_all_distances(data_emb, targetlist_emb)
    logger.info("Finish getting embedding")

    """
    0 - benign
    1 - phishing
    """

    y_true = np.zeros([len(X), 1])
    y_score = np.zeros([len(X), 1])

    n = 1  # Top-1 match
    print("Start ")
    # args.result_path.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(data_emb.shape[0])):
        # url = open(file_names[i].replace('shot.png', 'info.txt'), encoding='utf-8', errors='ignore').read()
        # print(url)
        distances_to_target = pairwise_distance[i, :]
        idx, values = Evaluate.find_min_distances(np.ravel(distances_to_target), n)
        names_min_distance, only_names, min_distances = find_names_min_distances(idx, values, all_file_names)
        # distance lower than threshold ==> report as phishing
        if float(min_distances) <= args.threshold:
            phish = 1
            y_score[i] = 1
        # else it is benign
        else:
            phish = 0

        with open(args.result_path, "a+", encoding="utf-8", errors="ignore") as f:
            f.write(f"{file_names[i]}\t{y_score[i]}\t{str(min_distances)}\t{str(only_names[0])}\n")

    # TODO: closest target => target
    print(roc_auc_score(y_true, y_score))
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=0)

    # Print ROC curve
    plt.plot(fpr, tpr)
    plt.show()
