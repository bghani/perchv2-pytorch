"""
Mode 2: linear probing.

Backbone is frozen (weights don't update); only the new linear head is
trained. Good default starting point -- fast, works with limited
labelled data, and gives you a baseline to beat before paying for a
full fine-tune.

Runnable as-is: uses ToySineDataset (synthetic tones, no real audio, no
download) so you can see the training loop actually work end-to-end.
Swap `train_loader` for your own Dataset/DataLoader when you're ready
to train on real recordings. ONNX_PATH must point at a real
perch_v2.onnx (see README "Getting the weights") -- unlike the native
backbone, this one can't run with a random/missing checkpoint, since the
whole model (including its frontend) is converted from that one file.
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
        mode="linear_probe",
        cache_dir=str(CACHE_DIR),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"{len(trainable)} trainable parameter tensors (should just be the head's)")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
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
            optimizer.step()

            total_loss += loss.item() * waveforms.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += waveforms.size(0)

        print(
            f"epoch {epoch+1}/{EPOCHS}  "
            f"loss={total_loss/total:.4f}  acc={correct/total:.2%}"
        )


if __name__ == "__main__":
    main()
