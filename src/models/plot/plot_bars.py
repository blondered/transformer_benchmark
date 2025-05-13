import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_combined_metrics_bars(
    metrics_dict, title="Negative Sampling Comparison", color_palettes=None
):
    """
    Create a combined horizontal bar plot visualization for metrics data from multiple datasets.
    Each metric-dataset combination has its own color scale and x-axis limits.
    The bar with the maximum value in each column has a black border and bold text value.

    Parameters:
    -----------
    metrics_dict : dict
        Dictionary where keys are dataset names and values are metrics dataframes
    title : str, optional
        Title for the plot (default: 'Negative Sampling Comparison')
    color_palettes : list of str, optional
        List of color palette names for each metric type. Must be valid matplotlib colormap names.
        Default: ['viridis', 'plasma', 'inferno'] for NDCG@10, Coverage@10, and Debiased NDCG@10 respectively
    """
    # Set default color palettes if none provided
    if color_palettes is None:
        color_palettes = ["viridis", "plasma", "inferno"]
    elif len(color_palettes) != 3:
        raise ValueError("color_palettes must be a list of exactly 3 color scale names")

    # Create a figure with 6 subplots with more width
    fig, axes = plt.subplots(1, 6, figsize=(24, 6), sharey=True)
    fig.suptitle(title, fontsize=24)

    # Process each dataset and metric combination
    col = 0
    for dataset_name, metrics_df in metrics_dict.items():
        # Get labels for this dataset, preserving original order but reversed
        y_labels = list(metrics_df.index)[::-1]  # Reverse the order
        # Replace "0.0 In-batch" with "Uniform" in y_labels
        y_labels = [
            "Uniform" if label == "0.0 In-batch" else label for label in y_labels
        ]

        # Process NDCG@10
        ndcg_values = metrics_df["NDCG@10"].values[
            ::-1
        ]  # Reverse the values to match labels
        ndcg_colors = plt.cm.get_cmap(color_palettes[0])(
            np.linspace(0, 1, len(ndcg_values))
        )
        # Find index of maximum value
        max_idx = np.argmax(ndcg_values)

        # Plot all bars
        bars = axes[col].barh(y_labels, ndcg_values, color=ndcg_colors)
        # Add black border to the bar with maximum value
        bars[max_idx].set_edgecolor("black")
        bars[max_idx].set_linewidth(2)

        max_val = ndcg_values.max()
        axes[col].set_xlim(0, max_val * 1.2)  # Add 20% padding for text
        for i, v in enumerate(ndcg_values):
            # Add small offset to x position for text
            text_x = v + max_val * 0.02  # 2% of max value as offset
            if i == max_idx:
                axes[col].text(
                    text_x,
                    i,
                    f"{v:.4f}",
                    ha="left",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                )
            else:
                axes[col].text(
                    text_x, i, f"{v:.4f}", ha="left", va="center", fontsize=18
                )

        if col % 3 == 0:  # First metric in the group
            axes[col].set_title(
                f"$\\bf{{{dataset_name}}}$\nNDCG@10", pad=5, fontsize=24
            )
        else:
            axes[col].set_title("NDCG@10", pad=5, fontsize=24)
        axes[col].tick_params(axis="y", labelsize=18)
        axes[col].tick_params(axis="x", labelsize=14)
        # Remove borders
        for spine in axes[col].spines.values():
            spine.set_visible(False)
        col += 1

        # Process Coverage@10
        coverage_values = metrics_df["Coverage@10"].values[::-1]  # Reverse the values
        coverage_colors = plt.cm.get_cmap(color_palettes[1])(
            np.linspace(0, 1, len(coverage_values))
        )
        # Find index of maximum value
        max_idx = np.argmax(coverage_values)

        # Plot all bars
        bars = axes[col].barh(y_labels, coverage_values, color=coverage_colors)
        # Add black border to the bar with maximum value
        bars[max_idx].set_edgecolor("black")
        bars[max_idx].set_linewidth(2)

        max_val = coverage_values.max()
        axes[col].set_xlim(0, max_val * 1.2)  # Add 20% padding for text
        for i, v in enumerate(coverage_values):
            # Add small offset to x position for text
            text_x = v + max_val * 0.02  # 2% of max value as offset
            if i == max_idx:
                axes[col].text(
                    text_x,
                    i,
                    f"{v:.4f}",
                    ha="left",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                )
            else:
                axes[col].text(
                    text_x, i, f"{v:.4f}", ha="left", va="center", fontsize=18
                )

        axes[col].set_title("Coverage@10", pad=5, fontsize=24)
        axes[col].tick_params(axis="y", labelsize=18)
        axes[col].tick_params(axis="x", labelsize=14)
        # Remove borders
        for spine in axes[col].spines.values():
            spine.set_visible(False)
        col += 1

        # Process Debiased NDCG@10
        debiased_values = metrics_df["Debiased NDCG@10"].values[
            ::-1
        ]  # Reverse the values
        debiased_colors = plt.cm.get_cmap(color_palettes[2])(
            np.linspace(0, 1, len(debiased_values))
        )
        # Find index of maximum value
        max_idx = np.argmax(debiased_values)

        # Plot all bars
        bars = axes[col].barh(y_labels, debiased_values, color=debiased_colors)
        # Add black border to the bar with maximum value
        bars[max_idx].set_edgecolor("black")
        bars[max_idx].set_linewidth(2)

        max_val = debiased_values.max()
        axes[col].set_xlim(0, max_val * 1.2)  # Add 20% padding for text
        for i, v in enumerate(debiased_values):
            # Add small offset to x position for text
            text_x = v + max_val * 0.02  # 2% of max value as offset
            if i == max_idx:
                axes[col].text(
                    text_x,
                    i,
                    f"{v:.4f}",
                    ha="left",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                )
            else:
                axes[col].text(
                    text_x, i, f"{v:.4f}", ha="left", va="center", fontsize=18
                )

        axes[col].set_title("Debiased NDCG@10", pad=5, fontsize=24)
        axes[col].tick_params(axis="y", labelsize=18)
        axes[col].tick_params(axis="x", labelsize=14)
        # Remove borders
        for spine in axes[col].spines.values():
            spine.set_visible(False)
        col += 1

    # Adjust layout with more spacing between subplots
    plt.subplots_adjust(wspace=0.3)
    plt.tight_layout()

    return fig
