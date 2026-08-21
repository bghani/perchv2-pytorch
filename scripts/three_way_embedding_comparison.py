"""
Three-way frozen-embedding comparison: real ONNX (onnxruntime, ground
truth), onnx2torch-converted PyTorch (the recommended backbone), and the
legacy timm-based PyTorch backbone -- reporting cosine similarity,
Pearson correlation, and relative L2 error between all three.

This exists because cosine similarity alone can be misleading: the
legacy timm backbone measures ~0.96 cosine similarity against the true
model, which sounds close to 1.0, but its relative L2 error is ~0.23-0.29
-- meaning the embedding vector differs from the correct one by roughly
a quarter of its own magnitude. Cosine similarity (an angle between
vectors) can look reassuringly high while the magnitude of the error
stays large enough to matter for downstream tasks. See legacy/README.md
for the full story and what this finding meant in practice (a linear
probe trained on the legacy backbone's embeddings converged slower and
lower than one trained on the ONNX-converted backbone's).

SETUP -- three files need to be in place before running:
  1. weights/perch_v2.onnx              -- see README "Getting the weights"
  2. legacy/weights/perch_v2_backbone_timm.pt  -- from bghani/perch2-pytorch-weights on HF
  3. data/<some audio file>.wav          -- a real recording; this repo ships
     with a placeholder expectation of data/sample.wav (see README for the
     exact file used during development, from xeno-canto)

Usage (with all three files in their default locations):
    python scripts/three_way_embedding_comparison.py

Or pointing at different locations:
    python scripts/three_way_embedding_comparison.py \
        --onnx weights/perch_v2.onnx \
        --timm-weights legacy/weights/perch_v2_backbone_timm.pt \
        --audio data/sample.wav

Add --include-synthetic to also test against synthetic sine-tone inputs
(distinct frequencies + noise) -- this was used during development to
test (and rule out) the hypothesis that the legacy backbone's error was
specific to out-of-distribution/synthetic input; it isn't, real audio
and synthetic tones show the same ~0.96 cosine similarity / ~0.25
relative L2 error pattern, so this is off by default to keep the normal
run simple and fast.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def load_audio(audio_path, target_len=160_000, sr=32000):
    import soundfile as sf
    data, orig_sr = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data).mean(dim=1)  # mono
    if orig_sr != sr:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, orig_sr, sr)
    if waveform.shape[0] < target_len:
        waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[0]))
    else:
        waveform = waveform[:target_len]
    return waveform.unsqueeze(0)


def make_sine_tone(freq_hz, duration_s=5.0, sr=32000, noise_std=0.05, seed=0):
    n_samples = int(duration_s * sr)
    t = torch.arange(n_samples, dtype=torch.float32) / sr
    tone = 0.3 * torch.sin(2 * torch.pi * freq_hz * t)
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(n_samples, generator=g) * noise_std
    return (tone + noise).unsqueeze(0)


def cosine_similarity(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def pearson_correlation(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def relative_l2_error(a, b):
    a, b = a.flatten().astype(np.float64), b.flatten().astype(np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))


def report(label, a, b):
    cos = cosine_similarity(a, b)
    pear = pearson_correlation(a, b)
    rel = relative_l2_error(a, b)
    print(f"  {label}:")
    print(f"    cosine similarity:   {cos:.6f}")
    print(f"    Pearson correlation: {pear:.6f}")
    print(f"    relative L2 error:   {rel:.6f}")
    return cos, pear, rel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--onnx", default=str(REPO_ROOT / "weights" / "perch_v2.onnx"))
    parser.add_argument("--timm-weights", default=str(REPO_ROOT / "legacy" / "weights" / "perch_v2_backbone_timm.pt"))
    parser.add_argument("--audio", default=str(REPO_ROOT / "data" / "sample.wav"))
    parser.add_argument("--cache-dir", default=str(REPO_ROOT / "weights" / "onnx_cache"),
                         help="Cache the converted ONNX model here to speed up repeat runs.")
    parser.add_argument("--include-synthetic", action="store_true",
                         help="Also test against synthetic sine-tone inputs (off by default).")
    args = parser.parse_args()

    for label, path in [("--onnx", args.onnx), ("--timm-weights", args.timm_weights), ("--audio", args.audio)]:
        if not Path(path).exists():
            print(f"ERROR: {label} points at a file that doesn't exist: {path}")
            print("See this script's module docstring (or the README) for where each file comes from.")
            sys.exit(1)

    import onnxruntime as ort
    from perchv2_pytorch import PerchONNXEmbedder
    from legacy import Perch2Embedder

    print("Loading models...")
    onnx_session = ort.InferenceSession(args.onnx)
    input_name = onnx_session.get_inputs()[0].name

    onnx2torch_embedder = PerchONNXEmbedder(args.onnx, cache_dir=args.cache_dir)
    onnx2torch_embedder.eval()

    timm_embedder = Perch2Embedder(weights_path=args.timm_weights)
    timm_embedder.eval()

    test_inputs = {"real_audio": load_audio(args.audio)}
    if args.include_synthetic:
        test_inputs["sine_500hz"] = make_sine_tone(500.0, seed=0)
        test_inputs["sine_1500hz"] = make_sine_tone(1500.0, seed=1)
        test_inputs["sine_4000hz"] = make_sine_tone(4000.0, seed=2)

    results = {}
    for label, waveform in test_inputs.items():
        print(f"\n{'='*70}")
        print(f"Input: {label}")
        print(f"{'='*70}")

        waveform_np = waveform.numpy().astype(np.float32)

        onnx_emb = onnx_session.run(["embedding"], {input_name: waveform_np})[0]
        with torch.no_grad():
            onnx2torch_emb = onnx2torch_embedder(waveform).numpy()
            timm_emb = timm_embedder(waveform).numpy()

        print("\nonnx2torch vs ONNX Runtime (sanity check -- should be ~1.0):")
        report("onnx2torch vs onnx", onnx2torch_emb, onnx_emb)

        print("\ntimm (legacy) vs ONNX Runtime (the real question):")
        r2 = report("timm vs onnx", timm_emb, onnx_emb)

        print("\ntimm (legacy) vs onnx2torch:")
        report("timm vs onnx2torch", timm_emb, onnx2torch_emb)

        results[label] = r2

    if len(results) > 1:
        print(f"\n\n{'='*70}")
        print("SUMMARY: timm (legacy) vs ONNX, by input")
        print(f"{'='*70}")
        for label, (cos, pear, rel) in results.items():
            print(f"  {label:15s}: cosine={cos:.6f}  relative_L2_error={rel:.6f}")


if __name__ == "__main__":
    main()
