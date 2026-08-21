"""
Synthetic 3-class "bioacoustic" dataset with no real audio and no
download - exists purely so linear_probe.py and full_finetune.py run
end-to-end out of the box and demonstrate the training loop mechanics
(shapes, loss, what's actually receiving gradients in each mode).

Each class is a fixed-frequency sine tone plus noise, standing in for
3 species with distinct call pitches. This is NOT biologically
meaningful data and won't teach the model anything useful - swap in
your own Dataset (real waveforms + real labels) for actual training.
"""

import math
import torch
from torch.utils.data import Dataset


class ToySineDataset(Dataset):
    """
    3 classes, one distinct sine-tone frequency each, 5s clips @ 32kHz
    (Perch v2's expected input shape) with a little Gaussian noise added
    per sample so the classes aren't trivially identical.
    """

    CLASS_FREQS_HZ = [500.0, 1500.0, 4000.0]  # one tone per "class"

    def __init__(
        self,
        n_per_class: int = 20,
        duration_s: float = 5.0,
        sr: int = 32000,
        noise_std: float = 0.05,
        seed: int = 0,
    ):
        self.sr = sr
        self.n_samples = int(duration_s * sr)
        self.noise_std = noise_std
        self._generator = torch.Generator().manual_seed(seed)

        self.items = [
            (freq, label)
            for label, freq in enumerate(self.CLASS_FREQS_HZ)
            for _ in range(n_per_class)
        ]

    @property
    def num_classes(self) -> int:
        return len(self.CLASS_FREQS_HZ)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        freq, label = self.items[idx]
        t = torch.arange(self.n_samples, dtype=torch.float32) / self.sr
        tone = 0.3 * torch.sin(2 * math.pi * freq * t)
        noise = torch.randn(self.n_samples, generator=self._generator) * self.noise_std
        waveform = tone + noise
        return waveform, label
