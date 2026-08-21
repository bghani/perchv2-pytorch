"""
Regression tests for PerchFrontend. These don't require the converted
Perch v2 weights -- they only check the frontend's shape contract and
basic numerical sanity, so they can run in CI without downloading
anything.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from legacy import PerchFrontend


def test_output_shape():
    mel = PerchFrontend()
    mel.eval()
    x = torch.zeros(2, 160_000)  # 5s @ 32kHz, batch of 2
    out = mel(x)
    assert out.shape == (2, 500, 128), f"unexpected shape: {out.shape}"


def test_no_nans_on_silence():
    # Silence is exactly the case that broke an earlier PCEN-based
    # reconstruction of this frontend -- guard against regressing that.
    mel = PerchFrontend()
    mel.eval()
    x = torch.zeros(1, 160_000)
    out = mel(x)
    assert torch.isfinite(out).all(), "non-finite values in frontend output on silence"


def test_no_nans_on_random_audio():
    mel = PerchFrontend()
    mel.eval()
    x = torch.randn(2, 160_000) * 0.1
    out = mel(x)
    assert torch.isfinite(out).all(), "non-finite values in frontend output on random audio"



def test_handles_non_exact_multiple_length():
    # Input length not lining up cleanly with hop size shouldn't crash --
    # the padding logic should handle arbitrary lengths, not just exactly
    # 160000.
    mel = PerchFrontend()
    mel.eval()
    x = torch.zeros(1, 159_137)
    out = mel(x)
    assert out.shape[0] == 1
    assert out.shape[2] == 128
