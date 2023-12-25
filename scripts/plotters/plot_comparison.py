from enum import Enum
from pathlib import Path
from typing import Dict, List

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import srsly
import typer
from wasabi import msg

from .constants import ACL_STYLE, COMPONENT_TO_METRIC, COMPONENT_TO_TASK

pylab.rcParams.update(ACL_STYLE)


class Split(str, Enum):
    dev = "dev"
    test = "test"


def plot_comparison(
    our_scores_dir: Path,
    their_scores_dir: Path,
    output_file: Path,
    ours_name: str = typer.Option(
        "RoBERTa-based", help="Name for our_scores_dir to appear in plot."
    ),
    theirs_name: str = typer.Option(
        "MultiHashEmbed", help="Name of their_scores_dir to appear in plot."
    ),
    split: Split = typer.Option(Split.dev, help="Dataset split to plot."),
):
    """Plot comparison"""

    languages = sorted([dir.stem for dir in our_scores_dir.iterdir() if dir.is_dir()])
    msg.text(f"Found languages: {', '.join(lang for lang in languages)}")

    # Get score filepaths
    our_scores_fp = [
        our_scores_dir / lang / f"metrics-{lang}-{split.value}.json"
        for lang in languages
    ]
    their_scores_fp = [
        their_scores_dir / f"metrics-{lang}-{split.value}.json" for lang in languages
    ]

    # Get actual scores
    component_to_rects: Dict[str, Dict[str, List]] = {
        "tagger": {"ours": [], "theirs": []},
        "morphologizer": {"ours": [], "theirs": []},
        "trainable_lemmatizer": {"ours": [], "theirs": []},
    }

    components = COMPONENT_TO_METRIC.keys()

    def _get_score(scores_dict, component: str) -> float:
        score = (
            scores_dict.get("morphologizer").get("pos_acc")
            if component == "tagger"
            else scores_dict.get(component).get(COMPONENT_TO_METRIC[component])
        )
        return round(score, 2)

    for ours_fp, theirs_fp in zip(our_scores_fp, their_scores_fp):
        our_scores = srsly.read_json(ours_fp)
        their_scores = srsly.read_json(theirs_fp)
        for component in components:
            component_to_rects[component]["ours"].append(
                _get_score(our_scores, component)
            )

            component_to_rects[component]["theirs"].append(
                _get_score(their_scores, component)
            )

    x = np.arange(len(languages))

    bar_settings = {
        "ours": {"color": "gray", "edgecolor": "k"},
        "theirs": {"color": "w", "edgecolor": "k"},
    }

    fig, axs = plt.subplots(1, len(component_to_rects), figsize=(14, 5))
    for component, ax in zip(component_to_rects, axs):
        width = 0.30
        multiplier = 0
        for label, scores in component_to_rects[component].items():
            offset = width * multiplier
            rects = ax.bar(
                x + offset + 0.15,
                np.array(scores),
                width,
                label=ours_name if label == "ours" else theirs_name,
                **bar_settings.get(label),
            )
            multiplier += 1

        ax.set_ylabel("F1-score")
        ax.set_xlabel("Language")
        ax.set_xticks(x + width, languages)
        ax.set_title(COMPONENT_TO_TASK[component])

        # Hide the right and top splines
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)

        # Hack to show a single legend

    fig.tight_layout()
    ax.legend(loc=(-1.15, -0.2), ncol=2, frameon=False)

    if output_file.suffix != ".pdf":
        msg.warn(
            "File extension is not PDF. I highly recommend"
            " using that filetype for better resolution."
        )

    plt.savefig(output_file, transparent=True, bbox_inches="tight")
    msg.good(f"Saved file to {output_file}")


if __name__ == "__main__":
    typer.run(plot_comparison)
