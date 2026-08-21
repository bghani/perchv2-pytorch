"""
perchv2_pytorch: PyTorch backbone for Google's Perch v2, built via direct
ONNX graph conversion (onnx2torch) rather than hand reconstruction.

The native, timm-based reconstruction that used to live in this package
(PerchFrontend, Perch2Backbone, Perch2Classifier, Perch2Embedder) has
moved to legacy/ at the repo root -- it is NOT part of this installed
package anymore. See legacy/README.md for why (measured ~27% relative
L2 error against the true model on frozen embeddings, not yet resolved)
and the main repo README for the currently recommended usage.
"""

from .onnx_backbone import (
    PerchONNXBackbone,
    PerchONNXClassifier,
    PerchONNXEmbedder,
    build_block_map,
)

__all__ = [
    "PerchONNXBackbone",
    "PerchONNXClassifier",
    "PerchONNXEmbedder",
    "build_block_map",
]

__version__ = "0.3.0"
