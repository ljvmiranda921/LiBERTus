from enum import Enum
from pathlib import Path
from typing import Dict, List

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import typer
from srsly import read_json
from wasabi import msg

from .constants import ACL_STYLE, COMPONENT_TO_METRIC, COMPONENT_TO_TASK

pylab.rcParams.update(ACL_STYLE)


class Split(str, Enum):
    dev = "dev"
    test = "test"


def plot_crosseval_heatmap(
    input_dir: Path,
    output_file: Path,
    split: Split = typer.Option(Split.dev, help="Dataset split to plot."),
):
    """Plot heatmap to evaluate cross-lingual transfer"""
    lang_paths = sorted([f for f in input_dir.iterdir() if f.is_dir()])  # rows
    languages = [path.stem for path in lang_paths]  # cols

    # Add containers for the data
    component_to_heatmap: Dict[str, List[List[float]]] = {
        "tagger": [],
        "morphologizer": [],
        "trainable_lemmatizer": [],
    }

    components = COMPONENT_TO_METRIC.keys()

    for component in components:
        for row in lang_paths:
            rowline = []
            for col in languages:
                metrics_path = row / f"metrics-{col}-{split.value}.json"
                metrics = read_json(metrics_path)
                score = (
                    metrics["morphologizer"].get("pos_acc")
                    if component == "tagger"
                    else metrics[component].get(COMPONENT_TO_METRIC[component])
                )
                rowline.append(score)
            component_to_heatmap[component].append(rowline)

    # Plotting proper
    fig, axs = plt.subplots(1, len(component_to_heatmap), figsize=(14, 5))

    for ax, (component, matrix) in zip(axs, component_to_heatmap.items()):
        matrix = np.asarray(matrix)
        ax.imshow(matrix, cmap="Greys")
        ax.set_xticks(np.arange(len(languages)), labels=languages)
        ax.set_yticks(np.arange(len(languages)), labels=languages)
        ax.set_xlabel("Target language")
        ax.set_ylabel("Source language")
        ax.set_title(COMPONENT_TO_TASK.get(component))

        # Loop over data dimensions and create text annotations
        for i in range(len(languages)):
            for j in range(len(languages)):
                value = matrix[i, j]
                color = "w" if value >= 0.5 else "k"
                text = ax.text(
                    j,
                    i,
                    "{:.2f}".format(value),
                    ha="center",
                    va="center",
                    color=color,
                )

    fig.tight_layout()
    if output_file.suffix != ".pdf":
        msg.warn(
            "File extension is not PDF. I highly recommend"
            " using that filetype for better resolution."
        )

    plt.savefig(output_file, transparent=True)
    msg.good(f"Saved file to {output_file}")


if __name__ == "__main__":
    typer.run(plot_crosseval_heatmap)
