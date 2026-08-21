"""
Mode 3: full (deep) fine-tuning.

Backbone is unfrozen - gradients flow through the whole EfficientNet-B3
stack, not just a new head. This is what the official ONNX/tflite Perch
v2 releases can't do (inference-only formats have no autograd graph);
it's the reason this repo exists.

Most data-hungry and slowest of the three modes. Consider warming up
with linear_probe.py first, then switching to this mode with a lower
learning rate on the backbone once the head has converged. For
something in between, backbone.freeze_up_to_block(n) lets you freeze
just the early layers and train the rest -- see
perchv2_pytorch/onnx_backbone.py and notebooks/quickstart_onnx.ipynb.

Runnable as-is: uses ToySineDataset (synthetic tones, no real audio, no
download) so you can see the training loop actually work end-to-end,
including gradients flowing through the backbone (check the printed
backbone-gradient norm below). Swap `train_loader` for your own
Dataset/DataLoader when you're ready to train on real recordings.
ONNX_PATH must point at a real perch_v2.onnx (see README "Getting the
weights").
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

from perchv2_pytorch import PerchONNXClassifier
from toy_dataset import ToySineDataset

ONNX_PATH = Path(__file__).resolve().parent.parent / "weights" / "perch_v2.onnx"
CACHE_DIR = Path(__file__).resolve().parent.parent / "weights" / "onnx_cache"
EPOCHS = 5


def main():
    # Swap this out for your own data: any torch.utils.data.Dataset that
    # returns (waveform, label) pairs, where waveform is a 1D float32 tensor
    # of shape (160000,) - 5 seconds of mono audio at 32kHz (Perch v2's
    # expected input) - and label is an integer class index. Update
    # num_classes to match your dataset (used below via dataset.num_classes).
    dataset = ToySineDataset(n_per_class=20)  # placeholder synthetic data, see examples/toy_dataset.py
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = PerchONNXClassifier(
        num_classes=dataset.num_classes,
        onnx_path=str(ONNX_PATH),
        mode="finetune",
        cache_dir=str(CACHE_DIR),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Discriminative learning rates: smaller for the pretrained backbone
    # (it's already a good initialization, don't wreck it early), larger
    # for the freshly-initialized head.
    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.parameters(), "lr": 1e-5},
            {"params": model.head.parameters(), "lr": 1e-3},
        ]
    )
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(EPOCHS):
        total_loss, correct, total = 0.0, 0, 0
        for waveforms, labels in train_loader:
            waveforms, labels = waveforms.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(waveforms)
            loss = criterion(logits, labels)
            loss.backward()

            # Sanity check: this should be nonzero, confirming gradients
            # are actually flowing through the backbone in this mode
            # (they'd be None/zero in "frozen" or "linear_probe" mode).
            backbone_grad_norm = sum(
                p.grad.norm().item()
                for p in model.backbone.parameters()
                if p.grad is not None
            )

            optimizer.step()

            total_loss += loss.item() * waveforms.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += waveforms.size(0)

        print(
            f"epoch {epoch+1}/{EPOCHS}  "
            f"loss={total_loss/total:.4f}  acc={correct/total:.2%}  "
            f"backbone_grad_norm={backbone_grad_norm:.4f}"
        )


if __name__ == "__main__":
    main()
