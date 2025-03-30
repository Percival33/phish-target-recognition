import gc
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


def read_data_batched(data_path, reshape_size, batch_size=32):
    """
    Read and process data in batches, yielding each batch
    """
    all_labels = []
    if (data_path / "labels.txt").exists():
        with open(data_path / "labels.txt", "r") as f:
            all_labels = [line.strip() for line in f]

    # Get list of all valid files
    image_files = [f for f in sorted(data_path.iterdir()) if f.suffix != ".txt"]
    if not all_labels:
        all_labels = ["benign"] * len(image_files)

    total_files = len(image_files)

    for start_idx in range(0, total_files, batch_size):
        end_idx = min(start_idx + batch_size, total_files)
        batch_imgs = []
        batch_files = []

        for file_path in image_files[start_idx:end_idx]:
            img = read_image(file_path, logger)
            if img is None:
                img = read_image(file_path, logger, format="jpeg")
            if img is None:
                logger.error(f"Failed to process {file_path}")
                continue

            batch_imgs.append(resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True))
            batch_files.append(file_path.name)

        if batch_imgs:
            yield (np.asarray(batch_imgs), np.asarray(all_labels[start_idx:end_idx]), np.asarray(batch_files))


def process_dataset(data_path, reshape_size, model, save_path=None, batch_size=256):
    """
    Process dataset in batches and compute embeddings
    Returns total count of processed images and accumulated embeddings
    """
    total_processed = 0
    embeddings_list = []
    labels_list = []
    filenames_list = []

    data_generator = read_data_batched(data_path, reshape_size, batch_size)

    for imgs, labels, files in tqdm(
        data_generator, desc=f"Processing {data_path.name}", total=len(list(data_path.iterdir())) // batch_size
    ):
        # Get embeddings for current batch
        batch_emb = model.predict(imgs, batch_size=batch_size, verbose=0)
        embeddings_list.append(batch_emb)

        if save_path:
            labels_list.append(labels)
            filenames_list.append(files)

        total_processed += len(imgs)

        # Free memory
        if total_processed % 4096 == 0:
            del imgs
            gc.collect()

    # Concatenate all embeddings
    all_embeddings = np.concatenate(embeddings_list, axis=0)

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        all_labels = np.concatenate(labels_list, axis=0)
        all_filenames = np.concatenate(filenames_list, axis=0)

        np.save(save_path / "embeddings.npy", all_embeddings)
        np.save(save_path / "labels.npy", all_labels)
        np.save(save_path / "filenames.npy", all_filenames)

        return total_processed, all_embeddings, all_labels, all_filenames

    return total_processed, all_embeddings, None, None


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
            y_pred[i] = abs(float(min_distances) - threshold)

        with open(result_file, "a+", encoding="utf-8", errors="ignore") as f:
            f.write(f"{file_names[i]}\t{y_pred[i]}\t{str(min_distances)}\t{str(only_names[0])}\n")

    auc_score = roc_auc_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_pred, pos_label=1, drop_intermediate=False)

    return auc_score, fpr, tpr


def process_and_evaluate(args, model, targetlist_emb, all_file_names, phish_folder, benign_folder, batch_size):
    """
    Process both datasets and compute pairwise distances
    Args:
        args: ArgumentParser arguments
        model: loaded model
        targetlist_emb: target list embeddings
        all_file_names: list of target file names
        phish_folder: path to phishing dataset folder
        benign_folder: path to benign dataset folder
    """
    # Process phishing dataset
    phish_count, phish_emb, phish_labels, phish_files = process_dataset(
        args.data_dir / phish_folder,
        args.reshape_size,
        model,
        save_path=args.save_folder / phish_folder.name if args.save_intermediate else None,
        batch_size=batch_size,
    )
    logger.info(f"Processed {phish_count} phishing images")

    # Process benign dataset
    benign_count, benign_emb, benign_labels, benign_files = process_dataset(
        args.data_dir / benign_folder,
        args.reshape_size,
        model,
        save_path=args.save_folder / benign_folder.name if args.save_intermediate else None,
        batch_size=batch_size,
    )
    logger.info(f"Processed {benign_count} benign images")
    args.save_folder.mkdir(parents=True, exist_ok=True)
    # Combine embeddings
    data_emb = np.concatenate([benign_emb, phish_emb], axis=0)
    np.save(args.save_folder / "all_embeddings.npy", data_emb)

    # Combine labels and filenames if they were saved
    if args.save_intermediate:
        y = np.concatenate([benign_labels, phish_labels], axis=0)
        file_names = np.concatenate([benign_files, phish_files], axis=0)
    else:
        # For evaluation, create file names that indicate source
        benign_files = np.array([f"benign/file_{i}" for i in range(benign_count)])
        phish_files = np.array([f"phish/file_{i}" for i in range(phish_count)])
        file_names = np.concatenate([benign_files, phish_files])
        y = np.array(["benign"] * benign_count + ["phish"] * phish_count)

    np.save(args.save_folder / "all_labels.npy", y)
    np.save(args.save_folder / "all_file_names.npy", file_names)

    # Compute pairwise distances
    pairwise_distance = Evaluate.compute_all_distances_batched(data_emb, targetlist_emb)
    np.save(args.save_folder / "pairwise_distances.npy", pairwise_distance)

    return data_emb, pairwise_distance, y, file_names


def calculate_roc_curve(pairwise_distance, data_emb, targetlist_emb, all_file_names, file_names, y, args):
    # Test different thresholds
    thresholds = np.arange(3, 20, 1)
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


if __name__ == "__main__":
    np.random.seed(42)

    setup_logging()
    logger = logging.getLogger(__name__)

    parser = ArgumentParser()
    parser.add_argument("--emb-dir", type=Path, default=PROCESSED_DATA_DIR / "VisualPhish")
    parser.add_argument("--data-dir", type=Path, default=RAW_DATA_DIR / "VisualPhish")
    parser.add_argument("--margin", type=float, default=2.2)
    parser.add_argument("--saved-model-name", type=str, default="model2")
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--result-path", type=Path, default=LOGS_DIR / "VisualPhish")
    parser.add_argument("--reshape-size", default=[224, 224, 3])
    parser.add_argument("--save-intermediate", action="store_true", help="Save intermediate results")
    parser.add_argument("--phish-folder", type=str, default="newly_crawled_phishing")
    parser.add_argument("--benign-folder", type=str, default="benign_test")
    parser.add_argument("--save-folder", type=Path, default=LOGS_DIR / "VisualPhish-Results")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--weights-only", action="store_true", help="Load only weights")

    args = parser.parse_args()
    logger.info("Evaluating VisualPhishNet")

    targetlist_emb, all_labels, all_file_names = load_targetemb(
        args.emb_dir / "whitelist_emb.npy",
        args.emb_dir / "whitelist_labels.npy",
        args.emb_dir / "whitelist_file_names.npy",
    )
    modelHelper = ModelHelper()
    model = None
    if args.weights_only:
        input_shape = [224, 224, 3]
        new_conv_params = [3, 3, 512]
        model = modelHelper.define_triplet_network(input_shape, new_conv_params)
        model.load_weights("/Users/mjarczewski/Repositories/inz/data/processed/VP-from-baseline-pp/my_model2.h5")
    else:
        model = modelHelper.load_model(args.emb_dir, args.saved_model_name, args.margin)
    model = model.layers[3]
    logger.info("Loaded targetlist and model, number of protected target screenshots {}".format(len(targetlist_emb)))

    data_emb, pairwise_distance, y, file_names = process_and_evaluate(
        args,
        model,
        targetlist_emb,
        all_file_names,
        phish_folder=args.phish_folder,
        benign_folder=args.benign_folder,
        batch_size=args.batch_size,
    )

    calculate_roc_curve(pairwise_distance, data_emb, targetlist_emb, all_file_names, file_names, y, args)
