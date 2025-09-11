#!/usr/bin/env python3
"""
Combine three images into a single figure with separate subplots.
Each subplot maintains (14,9) equivalent spacing.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from font_config import get_font_size, get_figure_size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine three images into subplots")
    parser.add_argument(
        "--images",
        nargs=3,
        default=["T2_2.png", "T2_26.png", "T107_16_aug4.png"],
        help="Three image files to combine",
    )
    parser.add_argument(
        "--layout",
        choices=["horizontal", "vertical"],
        default="horizontal",
        help="Layout arrangement (default: horizontal)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default="combined_images.png",
        help="Output filename (default: combined_images.png)",
    )
    parser.add_argument(
        "--titles",
        nargs=3,
        default=["T2_2", "T2_26", "T107_16_aug4"],
        help="Titles for each subplot",
    )
    return parser.parse_args()


def load_and_validate_images(image_paths):
    """Load images and validate they exist."""
    images = []
    for path in image_paths:
        img_path = Path(path)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        try:
            img = mpimg.imread(img_path)
            images.append(img)
            print(f"Loaded: {img_path} (shape: {img.shape})")
        except Exception as e:
            raise ValueError(f"Failed to load {img_path}: {e}")

    return images


def create_combined_plot(images, titles, layout, output_path):
    """Create combined plot with three subplots."""

    if layout == "horizontal":
        rows, cols = 1, 3
        # For horizontal: wider figure to accommodate 3 plots side by side
        figsize = (42, 14)  # 3 * 14 width, 14 height
    else:
        rows, cols = 3, 1
        # For vertical: taller figure to accommodate 3 plots stacked
        figsize = (14, 27)  # 14 width, 3 * 9 height

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    if layout == "horizontal":
        axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]
    else:
        axes_list = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, (img, title, ax) in enumerate(zip(images, titles, axes_list)):
        ax.imshow(img)
        ax.set_title(title, fontsize=get_font_size("title"), weight="bold")

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_color("black")

    if layout == "horizontal":
        plt.subplots_adjust(wspace=0.1, hspace=0.1)
    else:
        plt.subplots_adjust(wspace=0.1, hspace=0.2)

    plt.savefig(output_path, bbox_inches="tight", dpi=300, facecolor="white")
    print(f"Combined image saved to: {output_path}")

    save_individual_plots(images, titles, output_path)


def save_individual_plots(images, titles, base_output_path):
    """Save each image as individual plot with (14,9) size."""
    base_name = base_output_path.stem
    base_dir = base_output_path.parent

    for i, (img, title) in enumerate(zip(images, titles)):
        fig, ax = plt.subplots(figsize=get_figure_size("single_plot"))  # (14,9)

        ax.imshow(img)
        ax.set_title(title, fontsize=get_font_size("title"), weight="bold")

        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_color("black")

        plt.tight_layout()

        individual_path = base_dir / f"{base_name}_individual_{i+1}_{title}.png"
        plt.savefig(individual_path, bbox_inches="tight", dpi=300, facecolor="white")
        print(f"Individual plot saved: {individual_path}")
        plt.close(fig)


def main():
    args = parse_args()

    image_paths = []
    for img_path in args.images:
        path = Path(img_path)
        if not path.is_absolute():
            if not path.exists():
                repo_root_path = Path(__file__).resolve().parents[1] / img_path
                if repo_root_path.exists():
                    path = repo_root_path
        image_paths.append(path)

    images = load_and_validate_images(image_paths)

    create_combined_plot(images, args.titles, args.layout, args.output)

    print(f"\nSuccessfully combined {len(images)} images!")
    print(f"Layout: {args.layout}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
