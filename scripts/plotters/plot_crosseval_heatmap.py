from enum import Enum
import typer
from pathlib import Path

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import typer
from wasabi import msg

from .constants import ACL_STYLE

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
    breakpoint()


if __name__ == "__main__":
    typer.run(plot_crosseval_heatmap)
