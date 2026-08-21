"""
Legacy: native PyTorch (timm-based) Perch v2 backbone.

Not part of the installed perchv2_pytorch package -- this directory is
kept for reference and possible future investigation, not for active
use. See legacy/README.md for why, and the main repo README for the
current recommended path (the ONNX-converted backbone).

To use this directly, add the repo root to sys.path so `legacy` is
importable as a plain top-level package -- it is NOT installed via
pip install -e . the way perchv2_pytorch is:

    import sys
    sys.path.insert(0, "/path/to/repo/root")
    from legacy import PerchFrontend, Perch2Backbone, Perch2Classifier, Perch2Embedder
"""

from .frontend import PerchFrontend
from .model import Perch2Backbone, Perch2Classifier, Perch2Embedder

__all__ = [
    "PerchFrontend",
    "Perch2Backbone",
    "Perch2Classifier",
    "Perch2Embedder",
]
