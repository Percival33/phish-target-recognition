import argparse
import csv
import logging
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

from tools.config import setup_logging
from tools.metrics import calculate_metrics

# Colorlog setup for colored logging
try:
    import colorlog

    def setup_colorlog_logging():
        """Setup colorlog with colored output: blue for info, green for debug, yellow for warning, red for error."""
        # Create a colored formatter
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(levelname)-8s%(reset)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            reset=True,
            log_colors={
                "DEBUG": "green",
                "INFO": "blue",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        # Get the root logger and clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()

        # Create and configure handler
        handler = colorlog.StreamHandler()
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        root_logger.setLevel(logging.INFO)

        return root_logger

except ImportError:

    def setup_colorlog_logging():
        """Fallback to standard logging if colorlog is not available."""
        setup_logging()
        return logging.getLogger()


def _read_results_txt(file_path: Path) -> pd.DataFrame:
    rows = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if row:
                rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=["file", "pp_class", "pp_target", "true_class", "true_target"]
        )  # empty

    logging.info("Columns %s", rows[0])
    df_raw = pd.DataFrame(
        rows,
        columns=[
            "folder",
            "url",
            "phish_category",
            "pred_target",
            "matched_domain",
            "siamese_conf",
            "logo_recog_time",
            "logo_match_time",
        ],
    )
    true_class = []
    true_target = []
    for folder in df_raw["folder"].astype(str):
        if folder.startswith("benign"):
            true_class.append(0)
            true_target.append("benign")
        else:
            true_class.append(1)
            true_target.append(folder.split("+", 1)[0])

    return pd.DataFrame(
        {
            "file": df_raw["folder"],
            "pp_class": pd.to_numeric(df_raw["phish_category"], errors="coerce")
            .fillna(0)
            .astype(int),
            "pp_target": (
                df_raw["pred_target"]
                .fillna("benign")
                .astype(str)
                .str.strip()
                .replace(
                    {
                        "None": "benign",
                        "": "benign",
                        "nan": "benign",
                        "NA": "benign",
                        "N/A": "benign",
                    }
                )
            ),
            "true_class": true_class,
            "true_target": true_target,
        }
    )


def process_and_evaluate(
    path_to_csv_or_results: str,
    plot: bool = False,
    out_dir: Optional[str] = None,
    is_phish: bool = False,
    is_benign: bool = False,
    save_csv: Optional[str] = None,
):
    path = Path(path_to_csv_or_results)
    if path.suffix.lower() == ".txt":
        df = _read_results_txt(path)
    else:
        df = pd.read_csv(path)

    # Normalize pp_target: fill missing/None-like with "benign"
    # if "pp_target" in df.columns:
    #     s = df["pp_target"].copy()
    #     s = s.fillna("benign")
    #     s = s.astype(str).str.strip()
    #     s = s.replace({"None": "benign", "": "benign", "nan": "benign", "NA": "benign", "N/A": "benign"})
    #     df["pp_target"] = s

    # Optional override of true labels based on flags
    if is_phish and is_benign:
        raise ValueError("Only one of --is-phish or --is-benign can be provided")
    if is_phish:
        logging.info(
            "Overriding true labels: setting true_class=1 and true_target from folder name"
        )
        df["true_class"] = 1
        if "file" in df.columns:
            df["true_target"] = (
                df["file"].astype(str).str.split("+", n=1, expand=False).str[0]
            )
        elif "folder" in df.columns:
            df["true_target"] = (
                df["folder"].astype(str).str.split("+", n=1, expand=False).str[0]
            )
        else:
            logging.error(
                "Could not find 'file' or 'folder' column to derive target; keeping existing true_target"
            )
            raise ValueError(
                "Could not find 'file' or 'folder' column to derive target; keeping existing true_target"
            )
    elif is_benign:
        logging.info(
            "Overriding true labels: setting true_class=0 and true_target='benign'"
        )
        df["true_class"] = 0
        df["true_target"] = "benign"

    # Optional export of minimal evaluation CSV
    if save_csv:
        cols = ["true_target", "true_class", "pp_class", "pp_target"]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns for export: {missing}")
        export_df = df[cols].copy()
        Path(save_csv).parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(save_csv, index=False)
        logging.info("Saved evaluation CSV to %s", save_csv)

    logging.info(f"df true class: {df['true_class'].head(n=10)}")
    logging.info(f"df pp class: {df['pp_class'].head(n=10)}")
    logging.info(f"df true target: {df['true_target'].head(n=10)}")
    logging.info(f"df pp target: {df['pp_target'].head(n=10)}")

    class_metrics, target_metrics = calculate_metrics(
        cls_true=df["true_class"],
        cls_pred=df["pp_class"],
        targets_true=df["true_target"],
        targets_pred=df["pp_target"],
    )

    logging.info(
        "samples=%d, unique_pred_targets=%d", len(df), df["pp_target"].nunique()
    )

    logging.info(f"Class metrics: {class_metrics}")
    logging.info(f"Target metrics: {target_metrics}")

    cm = confusion_matrix(df["true_class"], df["pp_class"], labels=[0, 1])
    cm_df = pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"])

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        cm_df.to_csv(out / "confusion_matrix.csv")

        # Simple per-run LaTeX tables
        def _latex(metrics: dict) -> str:
            items = [f"{k}={metrics[k]:.4f}" for k in sorted(metrics.keys())]
            return (
                "\\begin{tabular}{ll}\n"
                "\\toprule\n"
                "ID & Metrics \\ \\ \n"
                "\\midrule\n"
                f"Phishpedia & {', '.join(items)} \\ \n"
                "\\bottomrule\n"
                "\\end{tabular}\n"
            )

        (out / "class_metrics.tex").write_text(_latex(class_metrics))
        (out / "target_metrics.tex").write_text(_latex(target_metrics))

    if plot:
        auc_score = roc_auc_score(df["true_class"], df["pp_class"])
        fpr, tpr, _ = roc_curve(df["true_class"], df["pp_class"])
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC (AUC={auc_score:.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.show()


def _download_wandb_results(run_ref: str, dest_dir: Path) -> Optional[Path]:
    try:
        from wandb.apis.public import Api
    except Exception as e:
        logging.error(
            "wandb is required for --wandb-run. Install with: uv pip install wandb (%s)",
            e,
        )
        return None

    api = Api()
    try:
        run = api.run(run_ref)
    except Exception as e:
        logging.error("Failed to access W&B run %s: %s", run_ref, e)
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    for f in run.files():
        try:
            f.download(root=str(dest_dir), exist_ok=True)
        except Exception as e:
            logging.warning("Failed to download %s: %s", f.name, e)

    # Prefer top-level results.txt else first match
    candidate = dest_dir / "results.txt"
    if candidate.exists():
        return candidate
    for p in dest_dir.rglob("results.txt"):
        return p
    logging.error("results.txt not found in W&B run %s", run_ref)
    return None


# Example usage
if __name__ == "__main__":
    logger = setup_colorlog_logging()
    parser = argparse.ArgumentParser(
        description="Evaluate Phishpedia results (CSV, results.txt, or W&B run)"
    )
    parser.add_argument(
        "path", nargs="?", default=None, help="Path to CSV or results.txt"
    )
    parser.add_argument(
        "--wandb-run",
        dest="wandb_run",
        default=None,
        help="W&B run ref, e.g. entity/project/runid",
    )
    parser.add_argument("--plot", action="store_true", help="Show ROC curve plot")
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for confusion matrix and LaTeX",
    )
    parser.add_argument(
        "--is-phish",
        action="store_true",
        help="Override labels: mark all samples as phishing (true_class=1) and true_target from folder name",
    )
    parser.add_argument(
        "--is-benign",
        action="store_true",
        help="Override labels: mark all samples as benign (true_class=0) and true_target='benign'",
    )
    parser.add_argument(
        "--save-csv",
        dest="save_csv",
        type=str,
        default=None,
        help="Save CSV with columns: true_target,true_class,pp_class,pp_target",
    )
    args = parser.parse_args()

    if (args.path is None and args.wandb_run is None) or (
        args.path is not None and args.wandb_run is not None
    ):
        parser.error("Provide exactly one input: either a local path or --wandb-run")

    if args.is_phish and args.is_benign:
        parser.error("Provide at most one of --is-phish or --is-benign")

    if args.wandb_run:
        # Use a temp directory to download; if --out provided, reuse under it for traceability
        if args.out:
            dl_dir = Path(args.out) / "wandb_files"
            dl_dir.mkdir(parents=True, exist_ok=True)
        else:
            dl_dir = Path(tempfile.mkdtemp(prefix="pp_eval_wandb_"))
        logging.info("Downloading W&B run %s to %s", args.wandb_run, dl_dir)
        results_file = _download_wandb_results(args.wandb_run, dl_dir)
        if not results_file:
            raise SystemExit(1)
        process_and_evaluate(
            str(results_file),
            args.plot,
            args.out,
            args.is_phish,
            args.is_benign,
            args.save_csv,
        )
    else:
        process_and_evaluate(
            args.path, args.plot, args.out, args.is_phish, args.is_benign, args.save_csv
        )
