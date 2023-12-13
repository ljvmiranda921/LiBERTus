import csv
from pathlib import Path

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import typer
from wasabi import msg

from .constants import ACL_STYLE

pylab.rcParams.update(ACL_STYLE)


def plot_training_curve(input_file: Path, output_file: Path):
    """Plot training curve for the base model"""

    losses = []
    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader, None)  # skip the headers
        for row in csv_reader:
            _, loss, _, _ = row
            losses.append(float(loss))

    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot actual data
    ax.plot(losses, color="k")
    ax.set_ylabel("Pretraining Loss")
    ax.set_xlabel("Steps")
    ax.set_ylim(bottom=0)

    def formatter(x, pos):
        del pos
        actual_steps = int(x * 100)
        return f"{actual_steps // 1000}k"

    ax.xaxis.set_major_formatter(formatter)

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
    msg.good(f"Saved file to {output_file}")


if __name__ == "__main__":
    typer.run(plot_training_curve)