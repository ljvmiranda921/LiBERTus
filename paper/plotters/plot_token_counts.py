from enum import Enum
import csv
from pathlib import Path

import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import typer
from wasabi import msg

from .constants import ACL_STYLE

pylab.rcParams.update(ACL_STYLE)


class Label(str, Enum):
    before = "before upsampling"
    after = "after upsampling"


def plot_token_counts(input_file: Path, output_file: Path):
    """Plot token counts before and after sampling"""

    languages = []
    num_tokens = {Label.before.value: [], Label.after.value: []}
    bar_settings = {
        Label.before.value: {"color": "k", "hatch": None},
        Label.after.value: {"color": "w", "edgecolor": "k"},
    }

    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader, None)  # skip the headers
        for row in csv_reader:
            lang, before, after = row
            languages.append(lang)
            num_tokens[Label.before.value].append(int(before))
            num_tokens[Label.after.value].append(int(after))

    # Plot actual data
    x = np.arange(len(languages))
    width = 0.30
    multiplier = 0

    fig, ax = plt.subplots(figsize=(6, 4))
    for label, counts in num_tokens.items():
        offset = width * multiplier
        rects = ax.bar(
            x + offset + 0.15,
            counts,
            width,
            label=label,
            **bar_settings.get(label),
        )
        multiplier += 1

    ax.set_ylabel("Unique token counts")
    ax.set_xlabel("Language")
    ax.set_xticks(x + width, languages)
    ax.legend(loc=(0.10, 1.025), ncol=2, frameon=False)

    def formatter(x, pos):
        del pos
        return f"{x // 1000}k"

    ax.yaxis.set_major_formatter(formatter)

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
    typer.run(plot_token_counts)
