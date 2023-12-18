from pathlib import Path

import typer
from wasabi import msg

import matplotlib.pylab as pylab
import matplotlib.pyplot as plt

from .constants import ACL_STYLE

pylab.rcParams.update(ACL_STYLE)
