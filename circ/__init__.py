"""
CIRC — Circadian Integrated Research Core

Submodules
----------
circ.limbr      KNN imputation and SVA-based batch-effect removal
circ.pirs       Prediction Interval Ranking Score (constitutive expression)
circ.bootjtk               Bootstrap empirical JTK circadian rhythm detection
circ.expression_classification  Unified PIRS + BooteJTK expression classifier
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("circ")
except PackageNotFoundError:
    __version__ = "unknown"
