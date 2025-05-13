import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots


def create_combined_metrics_heatmap(
    metrics_dict, title="Negative Sampling Comparison", color_palettes=None
):
    """
    Create a combined heatmap visualization for metrics data from multiple datasets.
    Each metric-dataset combination has its own color scale.

    Parameters:
    -----------
    metrics_dict : dict
        Dictionary where keys are dataset names and values are metrics dataframes
    title : str, optional
        Title for the plot (default: 'Negative Sampling Comparison')
    color_palettes : list of str, optional
        List of color palette names for each metric type. Must be valid Plotly colorscale names.
        Default: ['viridis', 'plasma', 'inferno'] for NDCG@10, Coverage@10, and Debiased NDCG@10 respectively
    """
    # Set default color palettes if none provided
    if color_palettes is None:
        color_palettes = ["viridis", "plasma", "inferno"]
    elif len(color_palettes) != 3:
        raise ValueError("color_palettes must be a list of exactly 3 color scale names")

    # Create a figure with 6 subplots with minimal spacing
    fig, axes = plt.subplots(1, 6, figsize=(20, 6), sharey=True)
    fig.suptitle(title, fontsize=24)

    # Get all unique y labels across datasets
    all_y_labels = set()
    for df in metrics_dict.values():
        all_y_labels.update(df.index)

    # Use the order from the first dataset
    first_df = next(iter(metrics_dict.values()))
    y_labels = [label for label in first_df.index if label in all_y_labels]

    # Add any remaining labels that weren't in the first dataset
    remaining_labels = sorted(list(all_y_labels - set(y_labels)))
    y_labels.extend(remaining_labels)

    # Replace "0.0 In-batch" with "Uniform" in y_labels
    y_labels = ["Uniform" if label == "0.0 In-batch" else label for label in y_labels]

    # Process each dataset and metric combination
    col = 0
    for dataset_name, metrics_df in metrics_dict.items():
        # Process NDCG@10
        ndcg_values = metrics_df["NDCG@10"].values
        ndcg_df = pd.DataFrame(
            ndcg_values.reshape(-1, 1), index=y_labels, columns=[dataset_name]
        )

        sns.heatmap(
            ndcg_df,
            ax=axes[col],
            cmap=color_palettes[0],
            annot=True,
            fmt=".4f",
            annot_kws={"size": 18},
            cbar=False,
            square=False,
            xticklabels=False,
            yticklabels=True,
        )
        if col % 3 == 0:  # First metric in the group
            axes[col].set_title(
                f"$\\bf{{{dataset_name}}}$\nNDCG@10", pad=5, fontsize=20
            )
        else:
            axes[col].set_title("NDCG@10", pad=5, fontsize=20)
        axes[col].tick_params(axis="y", labelsize=18)
        col += 1

        # Process Coverage@10
        coverage_values = metrics_df["Coverage@10"].values
        coverage_df = pd.DataFrame(
            coverage_values.reshape(-1, 1), index=y_labels, columns=[dataset_name]
        )

        sns.heatmap(
            coverage_df,
            ax=axes[col],
            cmap=color_palettes[1],
            annot=True,
            fmt=".4f",
            annot_kws={"size": 18},
            cbar=False,
            square=False,
            xticklabels=False,
            yticklabels=True,
        )
        axes[col].set_title("Coverage@10", pad=5, fontsize=20)
        axes[col].tick_params(axis="y", labelsize=18)
        col += 1

        # Process Debiased NDCG@10
        debiased_values = metrics_df["Debiased NDCG@10"].values
        debiased_df = pd.DataFrame(
            debiased_values.reshape(-1, 1), index=y_labels, columns=[dataset_name]
        )

        sns.heatmap(
            debiased_df,
            ax=axes[col],
            cmap=color_palettes[2],
            annot=True,
            fmt=".4f",
            annot_kws={"size": 18},
            cbar=False,
            square=False,
            xticklabels=False,
            yticklabels=True,
        )
        axes[col].set_title("Debiased NDCG@10", pad=5, fontsize=20)
        axes[col].tick_params(axis="y", labelsize=18)
        col += 1

    # Adjust layout with minimal spacing
    plt.subplots_adjust(wspace=0.05)
    plt.tight_layout()

    # Save the figure as SVG
    plt.savefig("neg.svg", format="svg", bbox_inches="tight")
    plt.close()

    return fig
