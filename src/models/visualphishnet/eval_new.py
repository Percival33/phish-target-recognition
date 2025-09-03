import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import wandb
from skimage.transform import resize
from sklearn.metrics import precision_recall_curve
from tools.config import LOGS_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR, setup_logging
from tools.metrics import calculate_metrics
from tqdm import tqdm

from DataHelper import read_image
from Evaluate import Evaluate
from ModelHelper import ModelHelper


def load_targetemb(emb_path, label_path, file_name_path):
    """
    load targetlist embedding
    :return:
    """
    targetlist_emb = np.load(emb_path)
    all_labels = np.load(label_path)
    all_file_names = np.load(file_name_path)
    return targetlist_emb, all_labels, all_file_names


def read_data_batched(data_path, reshape_size, batch_size=32, logger=None):
    """
    Read and process data in batches, yielding each batch
    Data is expected to be organized as:
    data_path/
        target1/
            image1.png
            image2.png
        target2/
            image3.png
            ...
    """
    all_data = []

    for target_folder in sorted(data_path.iterdir()):
        if not target_folder.is_dir():
            continue

        target_name = target_folder.name

        for file_path in sorted(target_folder.iterdir()):
            if file_path.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp", ".gif"]:
                all_data.append(
                    {"path": file_path, "target": target_name, "filename": f"{target_name}/{file_path.name}"}
                )

    total_files = len(all_data)

    for start_idx in range(0, total_files, batch_size):
        end_idx = min(start_idx + batch_size, total_files)
        batch_imgs = []
        batch_labels = []
        batch_files = []

        for item in all_data[start_idx:end_idx]:
            img = read_image(item["path"], logger)
            if img is None:
                img = read_image(item["path"], logger, format="jpeg")
            if img is None:
                if logger:
                    logger.error(f"Failed to process {item['path']}")
                continue

            batch_imgs.append(resize(img, (reshape_size[0], reshape_size[1]), anti_aliasing=True))
            batch_labels.append(item["target"])
            batch_files.append(item["filename"])

        if batch_imgs:
            yield (np.asarray(batch_imgs), np.asarray(batch_labels), np.asarray(batch_files))


def process_dataset(data_path, reshape_size, model, save_path=None, batch_size=256, logger=None):
    """
    Process dataset in batches and compute embeddings
    Returns total count of processed images and accumulated embeddings
    """
    total_processed = 0
    embeddings_list = []
    labels_list = []
    filenames_list = []

    total_files = sum(
        1
        for folder in data_path.iterdir()
        if folder.is_dir()
        for file in folder.iterdir()
        if file.suffix.lower() in [".png", ".jpg", ".jpeg"]
    )

    data_generator = read_data_batched(data_path, reshape_size, batch_size, logger)

    for imgs, labels, files in tqdm(
        data_generator, desc=f"Processing {data_path.name}", total=(total_files + batch_size - 1) // batch_size
    ):
        batch_emb = model.predict(imgs, batch_size=batch_size, verbose=0)
        embeddings_list.append(batch_emb)
        labels_list.append(labels)
        filenames_list.append(data_path / files)
        total_processed += len(imgs)

    all_embeddings = np.concatenate(embeddings_list, axis=0) if embeddings_list else np.array([])
    all_labels = np.concatenate(labels_list, axis=0) if labels_list else np.array([])
    all_filenames = np.concatenate(filenames_list, axis=0) if filenames_list else np.array([])

    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        np.save(save_path / "embeddings.npy", all_embeddings)
        np.save(save_path / "labels.npy", all_labels)
        np.save(save_path / "filenames.npy", all_filenames)

    return total_processed, all_embeddings, all_labels, all_filenames


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


def get_model_hash(args):
    """Generate hash for model-related parameters only."""
    model_str = f"{args.saved_model_name}_{args.margin}_{args.emb_dir.name}_{args.weights_only}_{args.weights_path}_{args.phish_folder}_{args.benign_folder}"
    return str(hash(model_str))


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
        data_emb=embeddings_data["data_emb"],
        pairwise_distance=embeddings_data["pairwise_distance"],
        y=embeddings_data["y"],
        file_names=embeddings_data["file_names"],
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
            logger.debug(
                f"Cached config hash: {cached_config.get('config_hash')}, Current config hash: {current_config_hash}"
            )
            logger.warning("Configuration changed, invalidating cache")
            return None

        npz_data = np.load(cache_file, allow_pickle=True)
        embeddings_data = {
            "data_emb": npz_data["data_emb"],
            "pairwise_distance": npz_data["pairwise_distance"],
            "y": npz_data["y"],
            "file_names": npz_data["file_names"],
        }

        logger.info("Using cached embeddings")
        return embeddings_data

    except Exception as e:
        logger.error(f"Cache load failed: {e}")
        return None


def evaluate_threshold(
    pairwise_distance, data_emb, targetlist_emb, all_file_names, file_names, y, threshold, result_path, all_labels
):
    """
    Evaluate model performance for a given threshold and return results as a pandas DataFrame
    with additional metrics including F1 score, ROC AUC, and Matthews correlation coefficient
    0 - benign
    1 - phishing

    Note: y now contains actual target names instead of just "benign" or "phish"
    """
    y_true = []
    for i, label in enumerate(y):
        # Check if the file is from benign dataset based on the filename structure
        # file_names[i] format: "[...]/dataset_type/target_name/image_name.png"
        filename_parts = str(file_names[i]).split("/")
        is_benign = any("trusted_list" in part.lower() or "benign" in part.lower() for part in filename_parts)
        y_true.append(0 if is_benign else 1)

    y_true = np.array(y_true)
    y_pred = np.zeros([len(data_emb), 1])
    n = 1  # Top-1 match

    data = []

    true_targets = []
    pred_targets = []

    for i in tqdm(range(data_emb.shape[0]), desc=f"Processing threshold {threshold}"):
        distances_to_target = pairwise_distance[i, :]
        idx, values = Evaluate.find_min_distances(np.ravel(distances_to_target), n)
        names_min_distance, only_names, min_distances = find_names_min_distances(idx, values, all_file_names)

        filename = str(file_names[i])
        true_target = y[i]

        # Set VP class (0 for benign, 1 for phishing)
        vp_class = 0
        # distance lower than threshold ==> report as phishing
        if float(min_distances) <= threshold:
            vp_class = 1
            y_pred[i] = vp_class

        true_targets.append(true_target)
        vp_target = "benign" if vp_class == 0 else all_labels[idx[0]]
        logger.debug(f"vp_target: {vp_target}")
        pred_targets.append(vp_target)

        # Add data to the list
        data.append(
            {
                "file": filename,
                "vp_class": int(vp_class),
                "vp_distance": float(min_distances),
                "vp_target": vp_target,
                "true_class": y_true[i],
                "true_target": true_target,
            }
        )

    class_metrics, target_metrics = calculate_metrics(y_true, y_pred, true_targets, pred_targets)

    results_df = pd.DataFrame(data)

    print(f"clss_metrics: {class_metrics}")
    print(f"target_metrics: {target_metrics}")
    metrics_to_log = {**class_metrics, **{f"{k}": v for k, v in target_metrics.items()}}
    wandb.log(metrics_to_log)

    if result_path:
        result_path.mkdir(parents=True, exist_ok=True)
        csv_path = result_path / f"results_threshold_{threshold}.csv"
        results_df.to_csv(csv_path, index=False)

        metrics_path = result_path / f"metrics_threshold_{threshold}.txt"
        with open(metrics_path, "w") as f:
            f.write("Class-based metrics:\n")
            for key, value in class_metrics.items():
                f.write(f"{key}: {value}\n")
            f.write("\nTarget-based metrics:\n")
            for key, value in target_metrics.items():
                f.write(f"{key}: {value}\n")

    precision, recall, _ = precision_recall_curve(y_true, y_pred, pos_label=1, drop_intermediate=False)
    wandb.log({"precision": precision, "recall": recall})
    all_metrics = {**class_metrics, **target_metrics}

    return class_metrics["roc_auc"], precision, recall, results_df, all_metrics


def process_and_evaluate(args, model, targetlist_emb, all_file_names, phish_folder, benign_folder, batch_size, logger):
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
    phish_count, phish_emb, phish_labels, phish_files = process_dataset(
        args.data_dir / phish_folder,
        args.reshape_size,
        model,
        save_path=args.save_folder / phish_folder if args.save_intermediate else None,
        batch_size=batch_size,
        logger=logger,
    )
    logger.info(f"Processed {phish_count} phishing images")

    if phish_labels is not None:
        unique_phish_targets = np.unique(phish_labels)
        logger.info(f"Found {len(unique_phish_targets)} unique phishing targets: {unique_phish_targets[:10]}...")
        wandb.log({"phishing_targets_count": len(unique_phish_targets)})

    benign_count, benign_emb, benign_labels, benign_files = process_dataset(
        args.data_dir / benign_folder,
        args.reshape_size,
        model,
        save_path=args.save_folder / benign_folder if args.save_intermediate else None,
        batch_size=batch_size,
        logger=logger,
    )
    logger.info(f"Processed {benign_count} benign images")

    if benign_labels is not None:
        unique_benign_targets = np.unique(benign_labels)
        logger.info(f"Found {len(unique_benign_targets)} unique benign targets: {unique_benign_targets[:10]}...")
        wandb.log({"benign_targets_count": len(unique_benign_targets)})

    args.save_folder.mkdir(parents=True, exist_ok=True)

    data_emb = np.concatenate([benign_emb, phish_emb], axis=0) if benign_count > 0 and phish_count > 0 else np.array([])
    np.save(args.save_folder / "all_embeddings.npy", data_emb)

    y = np.concatenate([benign_labels, phish_labels], axis=0)
    file_names = np.concatenate([benign_files, phish_files], axis=0)

    np.save(args.save_folder / "all_labels.npy", y)
    np.save(args.save_folder / "all_file_names.npy", file_names)

    pairwise_distance = Evaluate.compute_all_distances_batched(data_emb, targetlist_emb)
    np.save(args.save_folder / "pairwise_distances.npy", pairwise_distance)

    wandb.log(
        {
            "phishing_images_count": phish_count,
            "benign_images_count": benign_count,
            "total_images_count": phish_count + benign_count,
        }
    )

    return data_emb, pairwise_distance, y, file_names


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
    parser.add_argument("--weights_path", type=Path, help="Used with --weights-only to specify path to a file")
    parser.add_argument("--wandb-project", type=str, default="VisualPhish-eval", help="Weights & Biases project name")
    parser.add_argument("--wandb-run-name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--wandb-tags", nargs="+", default=[], help="Weights & Biases tags for the run")
    parser.add_argument(
        "--force-recompute", action="store_true", help="Force recomputation of embeddings even if cache exists"
    )

    args = parser.parse_args()
    logger.info("Evaluating VisualPhishNet")

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or f"eval_{args.phish_folder}_{args.benign_folder}",
        tags=args.wandb_tags,
        config={
            "model": args.saved_model_name,
            "threshold": args.threshold,
            "batch_size": args.batch_size,
            "reshape_size": args.reshape_size,
            "phish_folder": args.phish_folder,
            "benign_folder": args.benign_folder,
            "weights_only": args.weights_only,
            "margin": args.margin,
            "force_recompute": args.force_recompute,
        },
    )

    targetlist_emb, all_labels, all_file_names = load_targetemb(
        args.emb_dir / "whitelist_emb.npy",
        args.emb_dir / "whitelist_labels.npy",
        args.emb_dir / "whitelist_file_names.npy",
    )
    logger.info("Loaded targetlist, number of protected target screenshots {}".format(len(targetlist_emb)))

    logger.info("Checking for cached embeddings...")
    config_hash = get_model_hash(args)
    embeddings_data = None if args.force_recompute else load_embeddings_data(args.save_folder, config_hash, logger)

    if embeddings_data is not None:
        logger.info("Using cached embeddings")
        data_emb = embeddings_data["data_emb"]
        pairwise_distance = embeddings_data["pairwise_distance"]
        y = embeddings_data["y"]
        file_names = embeddings_data["file_names"]
    else:
        logger.info("Computing embeddings from scratch...")
        modelHelper = ModelHelper()
        model = None
        if args.weights_only:
            input_shape = [224, 224, 3]
            new_conv_params = [3, 3, 512]
            model = modelHelper.define_triplet_network(input_shape, new_conv_params)
            model.load_weights(args.weights_path)
        else:
            model = modelHelper.load_model(args.emb_dir, args.saved_model_name, args.margin)
        model = model.layers[3]
        logger.info("Loaded model")

        data_emb, pairwise_distance, y, file_names = process_and_evaluate(
            args,
            model,
            targetlist_emb,
            all_file_names,
            phish_folder=args.phish_folder,
            benign_folder=args.benign_folder,
            batch_size=args.batch_size,
            logger=logger,
        )

        embeddings_data = {
            "data_emb": data_emb,
            "pairwise_distance": pairwise_distance,
            "y": y,
            "file_names": file_names,
        }
        save_embeddings_data(args.save_folder, embeddings_data, config_hash, logger)

    evaluate_threshold(
        pairwise_distance,
        data_emb,
        targetlist_emb,
        all_file_names,
        file_names,
        y,
        args.threshold,
        args.result_path,
        all_labels,
    )

    wandb.finish()
