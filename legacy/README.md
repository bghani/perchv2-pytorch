# Legacy: native PyTorch (timm-based) backbone

**Not currently recommended.** Kept for reference and possible future investigation, not for active use — see the main repo [README](../README.md) for the currently recommended approach (the ONNX-converted backbone).

## Why this is here, and why it's demoted

This was the original approach: hand-reconstruct Perch v2's architecture in `timm`'s `tf_efficientnet_b3`, then copy over weights converted from Google's JAX/Flax checkpoint. Several real architectural bugs were found and fixed this way through direct ONNX graph inspection — a missing convolution bias at the stem and head, a stem padding convention mismatch (`timm`'s dynamic "SAME" vs. Google's actual fixed VALID padding), and a BatchNorm parameter correction — which took whole-embedding cosine similarity from an original, never-fully-verified ~0.80 up to a validated **~0.97** across multiple real recordings.

That sounded like a strong result. It isn't the full story.

**A later three-way comparison (real ONNX via `onnxruntime`, `onnx2torch`-converted PyTorch, and this timm backbone) found `timm` sitting at a `relative L2 error` of ~0.23–0.29 against the true model** — meaning the embedding vector differs from the correct one by roughly a quarter of its own magnitude, consistently, across both real audio and synthetic test inputs. That's a large gap hiding behind a cosine-similarity number that looks close to 1.0: cosine similarity is an *angle* between vectors, and it can look reassuringly high while the *magnitude* of the error is still large enough to plausibly matter for downstream tasks.

This showed up concretely: a linear probe trained on frozen embeddings from this backbone converged more slowly and to a lower accuracy on a toy classification task than the same probe trained on ONNX-converted embeddings, which reached 100% quickly and stayed there.

**What's unresolved:** a fourth architectural bug was found (blocks 5, 8, and 18 also use asymmetric padding in the real ONNX graph, the same class of issue as the original stem bug) but attempting to fix it caused a *regression* in whole-embedding fidelity for reasons that were never root-caused, even after finding and fixing an unrelated ordering bug in the fix itself. That fix is **not applied** in this codebase. Whether the remaining L2 gap traces back to that specific unresolved issue, or something else entirely, is an open question.

## If you want to pick this back up

- `model.py` / `frontend.py` — the backbone and frontend implementation, with each fix documented in its own docstring, including the one that was reverted and why.
- `data/*.npy` — the extracted stem/head conv biases and mel filterbank matrix, pulled directly from the real ONNX graph (small files, not weights).
- `quickstart_timm.ipynb` — a full walkthrough (frozen features, linear probing, full fine-tuning) using this backbone.
- `tests/test_frontend.py` — frontend shape/stability tests, still passing.

The most promising next step is probably re-investigating the blocks 5/8/18 padding fix with fresh eyes, given the earlier attempt found a real bug in the fix's *implementation* (parameters loading before the architectural change instead of after) and still regressed after fixing that — suggesting either a second implementation bug, or that the original diagnosis of what's wrong at those blocks was incomplete.

## Usage

Not part of the installed `perchv2_pytorch` package — add the repo root to `sys.path` first:

```python
import sys
sys.path.insert(0, "/path/to/repo/root")
from legacy import PerchFrontend, Perch2Backbone, Perch2Classifier, Perch2Embedder
```

Weights: same [`bghani/perch2-pytorch-weights`](https://huggingface.co/bghani/perch2-pytorch-weights) HF repo as before, filename `perch_v2_backbone_timm.pt`.
