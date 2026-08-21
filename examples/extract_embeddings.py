"""
Mode 1: frozen features.

Use this when you just want fixed Perch v2 embeddings to feed into a
downstream classifier of your own choosing (sklearn, XGBoost, a nearest-
neighbour search index, etc). No training happens here at all.
"""

import torch
from pathlib import Path

from perchv2_pytorch import PerchONNXEmbedder

ONNX_PATH = Path(__file__).resolve().parent.parent / "weights" / "perch_v2.onnx"
CACHE_DIR = Path(__file__).resolve().parent.parent / "weights" / "onnx_cache"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embedder = PerchONNXEmbedder(str(ONNX_PATH), cache_dir=str(CACHE_DIR)).to(device)
    embedder.eval()

    # Perch v2 expects 5s mono clips at 32kHz = 160,000 samples.
    # Replace this with your own real audio loading.
    batch = torch.zeros(4, 160_000).to(device)

    with torch.no_grad():
        embeddings = embedder(batch)  # (4, 1536)

    print("Embedding shape:", embeddings.shape)
    # -> feed `embeddings` into your classifier / clustering / search index


if __name__ == "__main__":
    main()
