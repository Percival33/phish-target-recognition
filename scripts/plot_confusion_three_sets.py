import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from font_config import get_font_size, get_figure_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot confusion matrices for three datasets"
    )
    parser.add_argument(
        "--folder-root",
        "-f",
        type=Path,
        default=None,
        help="Path to folder containing CSV files (default: ../scripts-data relative to script)",
    )
    return parser.parse_args()


def load_results(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names across variants
    pred_col = None
    for candidate in ["vp_class", "baseline_class", "pp_class"]:
        if candidate in df.columns:
            pred_col = candidate
            break
    if pred_col is None:
        raise ValueError(
            f"Nie znaleziono kolumny z przewidywaną klasą w {csv_path}. "
            "Oczekiwane: vp_class, baseline_class, pp_class"
        )

    df = df.rename(columns={pred_col: "pred_class"})
    df["pred_class"] = df["pred_class"].astype(int)
    df["true_class"] = df["true_class"].astype(int)
    return df


def compute_confusion(df: pd.DataFrame) -> pd.DataFrame:
    matrix = pd.crosstab(
        df["true_class"].astype(int),
        df["pred_class"].astype(int),
        rownames=["Prawdziwa klasa"],
        colnames=["Przewidywana klasa"],
        dropna=False,
    )
    matrix = matrix.reindex(index=[0, 1], columns=[0, 1]).fillna(0).astype(int)
    matrix.index = ["benign", "phishing"]
    matrix.columns = ["benign", "phishing"]
    return matrix


def _plot_single_confusion(cm: pd.DataFrame, title: str, out_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=get_figure_size("single_confusion"))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        ax=ax,
        square=True,
        annot_kws={"size": get_font_size("heatmap_annot")},
    )
    ax.set_title(title, fontsize=get_font_size("title"), weight="bold")
    ax.set_xlabel("Przewidywana klasa", fontsize=get_font_size("xlabel"))
    ax.set_ylabel("Prawdziwa klasa", fontsize=get_font_size("ylabel"))
    ax.tick_params(axis="both", which="major", labelsize=get_font_size("tick_labels"))
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_and_save_all(
    vp_csv: Path, cert_csv: Path, pp_csv: Path, out_dir: Path
) -> None:
    vp_cm = compute_confusion(load_results(vp_csv))
    cert_cm = compute_confusion(load_results(cert_csv))
    pp_cm = compute_confusion(load_results(pp_csv))

    cert_path = out_dir / "cert-confusion.png"
    vp_path = out_dir / "vp-confusion.png"
    pp_path = out_dir / "pp-confusion.png"

    _plot_single_confusion(cert_cm, "Zbiór CERT", cert_path)
    _plot_single_confusion(vp_cm, "Zbiór VP", vp_path)
    _plot_single_confusion(pp_cm, "Zbiór PP", pp_path)

    print("CERT (wiersze: prawdziwa klasa, kolumny: przewidywana klasa)")
    print(cert_cm.to_string())
    print("\nVP (wiersze: prawdziwa klasa, kolumny: przewidywana klasa)")
    print(vp_cm.to_string())
    print("\nPP (wiersze: prawdziwa klasa, kolumny: przewidywana klasa)")
    print(pp_cm.to_string())

    # Print LaTeX snippet with vertically stacked images
    print("\nLaTeX (wklej do dokumentu):\n")
    latex = (
        "\\begin{figure}[t]\n"
        "  \\centering\n"
        f"  \\includegraphics[width=0.75\\textwidth]{{{cert_path.as_posix()}}}\\\n"
        f"  \\includegraphics[width=0.75\\textwidth]{{{vp_path.as_posix()}}}\\\n"
        f"  \\includegraphics[width=0.75\\textwidth]{{{pp_path.as_posix()}}}\\\n"
        "  \\caption{Macierze pomyłek dla zbiorów: CERT (góra), VP (środek), PP (dół).}\n"
        "  \\label{fig:confusion-three-sets}\n"
        "\\end{figure}\n"
    )
    print(latex)


if __name__ == "__main__":
    args = parse_args()

    # Use provided folder root or default to ../scripts-data
    if args.folder_root is not None:
        repo_root = args.folder_root
    else:
        repo_root = Path(__file__).resolve().parents[1] / "scripts-data"

    vp_csv = repo_root / "vp-result.csv"
    cert_csv = repo_root / "cert-result.csv"
    pp_csv = repo_root / "pp-result.csv"
    plot_and_save_all(vp_csv, cert_csv, pp_csv, repo_root)
