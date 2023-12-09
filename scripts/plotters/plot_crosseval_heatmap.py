from enum import Enum
from pathlib import Path

import numpy as np
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import typer
from srsly import read_json
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
    lang_paths = sorted([f for f in input_dir.iterdir() if f.is_dir()])
    languages = [path.stem for path in lang_paths]
    breakpoint()


if __name__ == "__main__":
    typer.run(plot_crosseval_heatmap)
