"""
CIRC — Circadian Integrated Research Core

Submodules
----------
circ.limbr      KNN imputation and SVA-based batch-effect removal
circ.pirs       Prediction Interval Ranking Score (constitutive expression)
circ.rhythmicity               BooteJTK + ECHO circadian rhythmicity detection and classification
circ.expression_classification  Unified PIRS + BooteJTK expression classifier
circ.compare    Statistical comparison of results between two conditions
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("circ")
except PackageNotFoundError:
    __version__ = "unknown"
