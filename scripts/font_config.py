#!/usr/bin/env python3
"""
Font configuration for plotting scripts.
All plotting scripts should import and use these font sizes for consistency.
"""

# Font sizes for different plot elements
FONT_SIZES = {
    # Main plot elements
    "title": 28,
    "xlabel": 26,
    "ylabel": 26,
    "tick_labels": 28,  # Increased for better axis number visibility
    "legend": 16,  # Increased for better legend readability
    # Annotations and text
    "annotation": 22,
    "text": 20,
    # Special elements
    "eer_text": 24,  # Increased for better EER threshold visibility
    "heatmap_annot": 48,  # Increased for better confusion matrix numbers
    # Small text
    "small": 16,
}

# Figure sizes for consistency
FIGURE_SIZES = {
    "single_plot": (14, 9),  # Larger for EER/density plots with bigger fonts
    "triple_horizontal": (36, 12),  # Proportionally larger
    "wide_single": (18, 10),  # Wider for better text accommodation
    "single_confusion": (10, 10),  # Kept square but optimized for readability
}

# Colors for consistency
COLORS = {
    "benign": "blue",
    "phishing": "red",
    "eer_threshold": "green",
    "background": "white",
}


# Helper function to get font size
def get_font_size(element_type: str) -> int:
    """Get font size for a specific plot element."""
    return FONT_SIZES.get(element_type, FONT_SIZES["text"])


# Helper function to get figure size
def get_figure_size(size_type: str) -> tuple:
    """Get figure size for a specific plot type."""
    return FIGURE_SIZES.get(size_type, FIGURE_SIZES["single_plot"])
