#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = [
#     "pandas",
#     "tqdm",
# ]
# ///

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from models.phishpedia.scripts.organize import get_special_domain_mapping


def parse_true_target(true_target: str) -> tuple[str, str]:
    """Parse `true_target` as 'domain--Txx_x', return (domain, identifier)."""
    if not true_target or "--" not in true_target:
        raise ValueError(f"Invalid true_target format: {true_target}")

    parts = true_target.split("--", 1)
    domain, identifier = parts[0], parts[1]

    if not identifier.startswith("T") or "_" not in identifier:
        raise ValueError(f"Invalid identifier format: {identifier}")

    return domain, identifier


def build_inverse_mapping(forward_map: dict[str, str]) -> dict[str, str]:
    """Invert mapping, ensuring no domain collisions."""
    inverse = {}
    for label, domain in forward_map.items():
        if domain in inverse and inverse[domain] != label:
            raise ValueError(f"Domain '{domain}' maps to multiple labels")
        inverse[domain] = label
    return inverse


def organize_vp_by_target(
    csv_path: Path, output_dir: Path, no_progress: bool = False
) -> None:
    """Create symlinks organized by target from CSV data."""

    df = pd.read_csv(csv_path)
    required_cols = {"file", "true_target", "true_class"}
    if missing := required_cols - set(df.columns):
        raise ValueError(f"Missing columns: {missing}")

    domain_to_label = build_inverse_mapping(get_special_domain_mapping())
    domain_to_label["miamidade.gov"] = (
        "mdpd"  # Map miamidade.gov to mdpd label (same as mps.it)
    )

    phishing_dir = output_dir / "phishing"
    trusted_dir = output_dir / "trusted_list"
    phishing_dir.mkdir(parents=True, exist_ok=True)
    trusted_dir.mkdir(parents=True, exist_ok=True)

    phishing_labels = set()
    trusted_labels = set()
    created, skipped = 0, 0

    iterator = df.iterrows()
    if not no_progress and tqdm:
        iterator = tqdm(iterator, total=len(df))

    for idx, row in iterator:
        source_path = Path(row["file"]).resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"Row {idx}: File not found: {source_path}")

        domain, identifier = parse_true_target(row["true_target"])
        if domain not in domain_to_label:
            raise ValueError(f"Row {idx}: Unknown domain: {domain}")
        label = domain_to_label[domain]

        is_phishing = int(row["true_class"]) == 1
        dest_base = phishing_dir if is_phishing else trusted_dir
        dest_dir = dest_base / label
        dest_dir.mkdir(exist_ok=True)

        dest_file = dest_dir / f"{identifier}{source_path.suffix}"
        if dest_file.exists():
            skipped += 1
        else:
            os.symlink(source_path, dest_file)
            created += 1

        (phishing_labels if is_phishing else trusted_labels).add(label)

    if phishing_labels:
        (phishing_dir / "targets2.txt").write_text(
            "\n".join(sorted(phishing_labels)) + "\n"
        )
    if trusted_labels:
        (trusted_dir / "targets.txt").write_text(
            "\n".join(sorted(trusted_labels)) + "\n"
        )

    print(f"Created {created} symlinks, skipped {skipped} existing.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Organize VisualPhish dataset by target"
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV file with columns: file,true_target,true_class",
    )
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bar"
    )

    args = parser.parse_args()

    try:
        organize_vp_by_target(Path(args.csv), Path(args.output), args.no_progress)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
