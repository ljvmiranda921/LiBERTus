import csv
from pathlib import Path

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import pandas as pd
import typer
from wasabi import msg

from .constants import ACL_STYLE

pylab.rcParams.update(ACL_STYLE)


def plot_pretraining_eval_scores(input_file: Path, output_file: Path):
    """Plot eval curve for pretraining comparison"""

    sampling_strategy = {"none": [], "upsampling": [], "averaging": []}
    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader, None)  # skip the headers
        for row in csv_reader:
            _, averaging, _, _, upsampling, _, _, none, _, _ = row
            sampling_strategy["none"].append(float(none) if not None else None)
            sampling_strategy["averaging"].append(
                float(averaging) if not None else None
            )
            sampling_strategy["upsampling"].append(
                float(upsampling) if not None else None
            )

    df = pd.DataFrame().from_dict(sampling_strategy)
    df = df.fillna(method="bfill")
    sampling_strategy = df.to_dict(orient="list")

    fig, ax = plt.subplots(figsize=(6, 4))

    LINE_STYLE = {
        "none": {"color": "k"},
        "upsampling": {"color": "k", "linestyle": "--"},
        "averaging": {"color": "k", "linestyle": ":"},
    }

    # Plot actual data
    for strategy_name, eval_curve in sampling_strategy.items():
        ax.plot(
            eval_curve,
            **LINE_STYLE.get(strategy_name),
            label=strategy_name.title(),
        )
    ax.set_ylabel("Validation Loss")
    ax.set_xlabel("Steps")
    ax.set_ylim(bottom=0)

    eval_steps = 500

    def formatter(x, pos):
        del pos
        actual_steps = int(x * eval_steps)
        return f"{actual_steps // 1000}k"

    ax.xaxis.set_major_formatter(formatter)

    # Hide the right and top splines
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)

    fig.tight_layout()
    ax.legend(loc="best", ncol=1, frameon=False)

    if output_file.suffix != ".pdf":
        msg.warn(
            "File extension is not PDF. I highly recommend"
            " using that filetype for better resolution."
        )

    plt.savefig(output_file, transparent=True)
    msg.good(f"Saved file to {output_file}")


if __name__ == "__main__":
    typer.run(plot_pretraining_eval_scores)
