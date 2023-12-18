from pathlib import Path
from enum import Enum
from typing import Any, Dict, List

import typer
from wasabi import msg
import srsly

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

from .constants import ACL_STYLE, COMPONENT_TO_TASK, COMPONENT_TO_METRIC

pylab.rcParams.update(ACL_STYLE)


class Split(str, Enum):
    dev = "dev"
    test = "test"


def plot_hashembed_comparison(
    our_scores_dir: Path,
    their_scores_dir: Path,
    output_file: Path,
    split: Split = typer.Option(Split.dev, help="Dataset split to plot."),
):
    """Plot hash embed comparison"""

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

    for ours_fp, theirs_fp in zip(our_scores_fp, their_scores_fp):
        our_scores = srsly.read_json(ours_fp)
        their_scores = srsly.read_json(theirs_fp)
        for component in components:
            component_to_rects[component]["ours"].append(
                our_scores.get(component).get(COMPONENT_TO_METRIC[component])
            )

            component_to_rects[component]["theirs"].append(
                their_scores.get(component).get(COMPONENT_TO_METRIC[component])
            )
    breakpoint()


if __name__ == "__main__":
    typer.run(plot_hashembed_comparison)
