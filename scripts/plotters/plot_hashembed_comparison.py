from pathlib import Path

import typer
from wasabi import msg

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

from .constants import ACL_STYLE

pylab.rcParams.update(ACL_STYLE)


def plot_hashembed_comparison(our_scores: Path, their_scores: Path):
    """Plot hash embed comparison"""

    lang_paths_ours = sorted([f for f in our_scores.iterdir() if f.is_dir()])
    lang_paths_theirs = sorted([f for f in their_scores.iterdir() if f.is_dir()])

    if set(lang_paths_ours) == set(lang_paths_theirs):
        msg.warn(
            "Number of language score directories not the same! "
            f"missing: {set(lang_paths_ours).difference(lang_paths_theirs)}"
        )
    breakpoint()

    languages = [path.stem for path in lang_paths_ours]


if __name__ == "__main__":
    typer.run(plot_hashembed_comparison)
