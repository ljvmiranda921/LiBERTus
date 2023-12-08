import csv
from pathlib import Path

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import typer
from wasabi import msg

from .constants import MATPLOTLIB_STYLE

pylab.rcParams.update(MATPLOTLIB_STYLE)


def plot_training_curve(input_file: Path, output_file: Path):
    """Plot training curve for the base model"""

    losses = []
    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader, None)  # skip the headers
        for row in csv_reader:
            _, loss, _, _ = row
            losses.append(float(loss))

    fig, ax = plt.subplots(figsize=(8, 4))

    # Plot actual data
    ax.plot(losses, color="k")
    ax.set_ylabel("Training Loss")
    ax.set_xlabel("Steps")
    ax.set_ylim(bottom=0)

    def formatter(x, pos):
        del pos
        return int(x * 100)

    ax.xaxis.set_major_formatter(formatter)

    # ax.set_xticks(
    #     range(0, len(losses)), labels=[i * 100 for i in range(0, len(losses))]
    # )

    # Hide the right and top splines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    fig.tight_layout()

    if output_file.suffix != ".pdf":
        msg.warn(
            "File extension is not PDF. I highly recommend"
            " using that filetype for better resolution."
        )

    plt.savefig(output_file, transparent=True)


if __name__ == "__main__":
    typer.run(plot_training_curve)
