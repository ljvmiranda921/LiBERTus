import csv
from pathlib import Path

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import typer

from .constants import MATPLOTLIB_STYLE

pylab.rcParams.update(MATPLOTLIB_STYLE)


def plot_training_curve(input_file: Path, output_file: Path):
    """Plot training curve for the base model"""
    with open(input_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        next(csv_reader, None)  # skip the headers
        for row in csv_reader:
            breakpoint()


if __name__ == "__main__":
    typer.run(plot_training_curve)
