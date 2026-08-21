# perchv2-pytorch

Unofficial PyTorch implementation of [Perch v2: The Bittern Lesson for Bioacoustics](https://arxiv.org/abs/2508.04665). Google's official released artifacts ([Kaggle](https://www.kaggle.com/models/google/bird-vocalization-classifier), [ONNX](https://huggingface.co/justinchuby/Perch-onnx)) are inference-only; this repo exists to make deep fine-tuning possible.

Built by converting Google's real ONNX computation graph directly into trainable PyTorch modules — confirmed **1.00000000 cosine similarity** against `onnxruntime`'s own output on real audio, and fully differentiable (gradients confirmed reaching every parameter). Supports:

1. **Full (deep) fine-tuning** — unfreeze the whole backbone and train it end-to-end on your own dataset.
2. **Partial fine-tuning** — freeze the stem and early blocks, train later blocks and the head.
3. **Linear probing** — freeze the backbone entirely, train only a new head.
4. **Frozen feature extraction** — pull fixed 1536-dim embeddings for downstream use in your own classifier, clustering, or search pipeline.

## Why this exists

Google's official Perch v2 releases (TF SavedModel, [ONNX](https://huggingface.co/justinchuby/Perch-onnx), tflite) are all inference-only formats. ONNX and tflite in particular have no autograd graph — there's no backward pass, so you cannot fine-tune through them, only run forward inference for embeddings or logits. If you want gradients flowing back through Perch's convolutional layers — actually adapting the backbone to your domain rather than just linear-probing on top of frozen features — you need a training-capable framework.

This repo provides that: the Perch v2 backbone (stock EfficientNet-B3, single-channel log-mel input) reimplemented in PyTorch, with weights converted from the original JAX/Flax checkpoint. This is a community conversion, not an official PyTorch release from Google — see "How the conversion was done" below for validation details and its known fidelity limits.

## How it was validated

`onnx2torch` translates each node in Google's actual released ONNX graph directly into an equivalent PyTorch operation — there's no architecture to guess at, since every op's exact parameters (weights, biases, padding, strides) are already fully specified in the graph itself. Three op(version) combinations needed custom converters not built into `onnx2torch` out of the box (`Pad`@18, `DFT`@17, `ReduceL2`/`ReduceMax`@18 — see [`perchv2_pytorch/onnx_backbone.py`](perchv2_pytorch/onnx_backbone.py) for what each does and how it was verified against `onnxruntime` directly before being trusted).


## Install

Requires Python ≥3.9.

### With [uv](https://docs.astral.sh/uv/) (recommended)

```bash
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

uv pip install -e ".[onnx]"
# for development (adds pytest):
uv pip install -e ".[onnx,dev]"
```

### With plain `venv` + `pip`

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -e ".[onnx]"
```

## Getting the weights

You need the original `perch_v2.onnx` file — this repo doesn't redistribute it. Get it from [`justinchuby/Perch-onnx`](https://huggingface.co/justinchuby/Perch-onnx/tree/main), and place it at `weights/perch_v2.onnx`.

Perch v2 and this derivative conversion are both licensed Apache 2.0 (see [LICENSE](LICENSE) and [NOTICE](NOTICE)) — redistribution and modification are permitted, but this is an unofficial community conversion, not something produced or endorsed by Google.

## Usage

```python
import torch
from perchv2_pytorch import PerchONNXEmbedder, PerchONNXBackbone, PerchONNXClassifier

# 1. Frozen features
embedder = PerchONNXEmbedder("weights/perch_v2.onnx", cache_dir="weights/onnx_cache")
embedder.eval()
waveform = torch.zeros(4, 160_000)  # 5s clips @ 32kHz, batch of 4
with torch.no_grad():
    embeddings = embedder(waveform)  # (4, 1536)

# 2. Full fine-tuning
backbone = PerchONNXBackbone("weights/perch_v2.onnx", cache_dir="weights/onnx_cache")
embeddings = backbone(waveform)  # gradients flow through everything by default

# 3. Partial fine-tuning -- freeze the stem and early blocks, train the rest
frozen, trainable = backbone.freeze_up_to_block(13)  # freezes stem + blocks 0-12

# 4. Classifier wrapper, mode="frozen" / "linear_probe" / "finetune"
model = PerchONNXClassifier(num_classes=42, onnx_path="weights/perch_v2.onnx", mode="linear_probe", cache_dir="weights/onnx_cache")
```

`PerchONNXBackbone`/`PerchONNXClassifier`/`PerchONNXEmbedder` all take raw waveform directly — framing, windowing, and the mel filterbank are part of the converted graph.

**`cache_dir` is worth using from the start.** Converting the raw ONNX graph takes several seconds (real, per-node overhead in `onnx2torch` itself, proportional to graph size) — `cache_dir` saves the converted model after the first run and reuses it on every subsequent one (across script runs, notebook kernel restarts, etc.), turning that cost into a one-time tax instead of a per-run one. Use the *same* `cache_dir` across every model you construct from the same `.onnx` file in a session for this to help.

Runnable, more complete versions of modes 1/3/4 are in [`examples/`](examples/) — `extract_embeddings.py`, `linear_probe.py`, `full_finetune.py` — and run out of the box against a synthetic toy dataset ([`examples/toy_dataset.py`](examples/toy_dataset.py)) once you've placed a real `perch_v2.onnx` at `weights/perch_v2.onnx`.

```bash
python examples/extract_embeddings.py
python examples/linear_probe.py
python examples/full_finetune.py
```

### Notebook

[`notebooks/quickstart_onnx.ipynb`](notebooks/quickstart_onnx.ipynb) — all four modes above, with explanations at each step.

## A note on the legacy backbone

An earlier version of this repo also shipped a hand-reconstructed `timm`-based backbone. It's been moved to [`legacy/`](legacy/) and is **not recommended**. A three-way comparison against real ONNX output found it sitting at a **relative L2 error of ~0.23–0.29** despite a reassuring-looking ~0.96 cosine similarity — cosine similarity measures an angle, not magnitude, and can look close to 1.0 while a large error still hides underneath it. This showed up concretely: a linear probe trained on its frozen embeddings converged slower and to a lower accuracy than one trained on the ONNX-converted backbone's embeddings on the same toy task. See [`legacy/README.md`](legacy/README.md) for the full story, including an unresolved architectural bug that may be the root cause — something to come back to later, not a dead end.

You can reproduce this comparison yourself — see below.

## Compare frozen embeddings across three implementations

[`scripts/three_way_embedding_comparison.py`](scripts/three_way_embedding_comparison.py) directly compares frozen embeddings from real ONNX (`onnxruntime`, ground truth), the `onnx2torch`-converted backbone (recommended), and the legacy `timm`-based backbone — reporting cosine similarity, Pearson correlation, and relative L2 error between each pair. This is the script that produced the ~0.96 cosine / ~0.25 relative-L2-error numbers cited above.

**Setup:**

Install both extras — this script needs the ONNX backbone *and* the legacy timm backbone to compare them, so one group alone isn't enough:

```bash
uv pip install -e ".[onnx,legacy]"
```

Then three files need to be in place:
1. `weights/perch_v2.onnx` — see "Getting the weights" above.
2. `legacy/weights/perch_v2_backbone_timm.pt` — download from [`bghani/perch2-pytorch-weights`](https://huggingface.co/bghani/perch2-pytorch-weights) on Hugging Face and place it there.
3. `data/sample.wav` — a real audio recording. See [`data/README.md`](data/README.md) — this repo ships with a real bioacoustic clip from [xeno-canto](https://xeno-canto.org/) at that path.

Then run:

```bash
python scripts/three_way_embedding_comparison.py
```

with no arguments needed if all three files are in their default locations (pass `--onnx`, `--timm-weights`, or `--audio` to point elsewhere). Add `--include-synthetic` to also test against synthetic sine-tone inputs — used during development to check (and rule out) whether the legacy backbone's error was specific to unusual/out-of-distribution input; it wasn't, real audio and synthetic tones show the same pattern.

## Architecture & frontend details

- **Backbone:** stock EfficientNet-B3, `in_chans=1` (Perch's stem takes single-channel log-mel input, not a 3-channel RGB-style image), returning a pooled **1536-dim** embedding. Framing, windowing, and the mel filterbank are all part of the converted ONNX graph — no separate frontend module needed.

## Tests

```bash
uv pip install -e ".[dev]"   # or: pip install -e ".[dev]"
pytest legacy/tests/
```

(The ONNX-converted backbone doesn't currently have its own dedicated unit tests beyond the manual verification documented in `perchv2_pytorch/onnx_backbone.py`'s docstrings — the tests that exist are for the legacy frontend.)

## Citation

If you use this in published work, please cite the original Perch v2 paper:

```
@article{van2025perch,
  title={Perch 2.0: The bittern lesson for bioacoustics},
  author={van Merri{\"e}nboer, Bart and Dumoulin, Vincent and Hamer, Jenny and Harrell, Lauren and Burns, Andrea and Denton, Tom},
  journal={arXiv preprint arXiv:2508.04665},
  year={2025}
}
```

and note that weights were obtained via community conversion, per [NOTICE](NOTICE).
