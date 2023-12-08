import csv
from pathlib import Path

import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import typer
from wasabi import msg

from .constants import ACL_STYLE

pylab.rcParams.update(ACL_STYLE)


def plot_token_counts(input_file: Path, output_file: Path):
    """Plot token counts before and after sampling"""

    languages = []
    num_tokens = {"before": [], "after": []}

    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader, None)  # skip the headers
        for row in csv_reader:
            lang, before, after = row
            languages.append(lang)
            num_tokens["before"].append(int(before))
            num_tokens["after"].append(int(after))

    # Plot actual data
    x = np.arange(len(languages))
    width = 0.25
    multiplier = 0

    fig, ax = plt.subplots(figsize=(8, 6))
    for label, counts in num_tokens.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, counts, width, label=label)
        ax.bar_label(rects, padding=3)
        multiplier += 1

    ax.set_ylabel("Token counts")
    ax.set_xlabel("Language")
    ax.set_xticks(x + width, languages)
    ax.legend(loc="upper left")

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
