"""
Log-mel frontend matching Perch v2's documented spec.

Spec source: arXiv 2508.04665, Appendix A.2 (the specific per-model
description, not the general perch repo README, which documents the
broader framework / a legacy PCEN option rather than what Perch 2.0
actually uses at inference time).

    "Our frontend outputs mel-scaled log-spectrograms, taking in 5s of
    audio at 32kHz (160,000 samples). It uses a hop length of 10ms (320
    samples) and a window length of 20ms (640 samples). The FFT window is
    set to 1,024 samples for computational efficiency. The frames are
    uncentered (i.e., the first frame begins at the first sample) and a
    Hann window is used. We calculate the energy (magnitude) spectrogram
    (so not a power spectrogram). The mel-scale is calculated using the
    HTK formula. Similar to SciPy's STFT implementation the output is
    scaled by the reciprocal of the sum of the window values. After the
    calculation of the mel-spectrogram we apply a logarithm with a floor
    of 1e-5 and then multiply the output by 0.1."

Two details worth flagging, found by extracting the actual ONNX graph's
window constant (scripts/compare_window.py): the graph's window matches
the SYMMETRIC Hann variant (periodic=False) almost exactly, pre-divided
by its own sum before the STFT rather than dividing the magnitude
spectrum afterward. That's a genuine, non-coincidental finding -- but
applying it made end-to-end fidelity slightly WORSE, not better (see the
note in __init__ below), so this frontend still uses periodic=True with
post-hoc normalization on the empirical evidence of what actually
performs best, despite the more "textbook correct" alternative having
been directly confirmed against ONNX ground truth. A reminder that a
locally-correct fix doesn't always compose to a global improvement when
other approximations are still present elsewhere in the pipeline.

Getting this frontend exactly right matters: an earlier PCEN-based
reconstruction (following the general repo README instead of the
Perch-2.0-specific appendix) produced NaNs and outlier blowups on
silence-then-onset audio. This log-mel version is numerically stable.
"""

import torch
import torch.nn as nn
import torchaudio
import warnings
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent / "data"


class PerchFrontend(nn.Module):
    """
    Input:  x of shape (batch, samples) at 32kHz, 5s clips = 160000 samples
    Output: (batch, time=500, n_mels=128)

    SpecAugment (frequency/time masking) is applied only when the module
    is in training mode (self.training == True), i.e. call .eval() to
    disable it for inference / embedding extraction.
    """

    def __init__(
        self,
        sr: int = 32000,
        win_length: int = 640,
        hopsize: int = 320,
        n_fft: int = 1024,
        n_mels: int = 128,
        fmin: float = 60.0,
        fmax: float = 16000.0,
        log_floor: float = 1e-5,
        log_scalar: float = 0.1,
        freqm: int = 48,
        timem: int = 192,
        use_extracted_filterbank: bool = True,
    ):
        super().__init__()
        self.win_length = win_length
        self.hopsize = hopsize
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sr = sr
        self.log_floor = log_floor
        self.log_scalar = log_scalar

        # Window: periodic=True Hann, post-hoc window_sum division. See
        # the note above this block for why periodic=False + pre-
        # normalization was tried and reverted despite being individually
        # well-confirmed against the ONNX graph's actual window constant.
        hann = torch.hann_window(win_length, periodic=True)
        window_sum = hann.sum()

        # Left-align the window within the n_fft-sized frame (zero-padded
        # at the end), rather than letting torch.stft auto-center a
        # window shorter than n_fft within each frame -- torch.stft's
        # default centering is a DIFFERENT "centered" concept from the
        # paper's center=False (frame centering), and conflating the two
        # was an earlier, separately-validated bug fix, independent of
        # the periodic/reorder question above. Passing win_length=n_fft
        # in forward() (matching this already-n_fft-length window)
        # bypasses torch's automatic centering entirely.
        window = torch.nn.functional.pad(hann, (0, n_fft - win_length))
        self.register_buffer("window", window, persistent=False)
        self.register_buffer("window_sum", window_sum, persistent=False)

        # Prefer the exact filterbank matrix extracted directly from the
        # ONNX graph (data/mel_filterbank.npy) over reconstructing it from
        # the HTK formula via torchaudio -- the reconstructed version
        # measured ~0.997-0.998 cosine similarity against ONNX's own
        # spectrogram output, an unresolved gap that plausibly explained
        # why the stem and earliest backbone layers remained the weakest
        # point even after the padding/bias fixes. See
        # scripts/extract_mel_filterbank.py for how this was found and
        # how to regenerate the .npy file.
        filterbank_path = _DATA_DIR / "mel_filterbank.npy"
        expected_shape = (n_fft // 2 + 1, n_mels)
        if use_extracted_filterbank and filterbank_path.exists():
            import numpy as np
            mel_fb = torch.from_numpy(np.load(filterbank_path).astype("float32"))
            if tuple(mel_fb.shape) != expected_shape:
                warnings.warn(
                    f"data/mel_filterbank.npy has shape {tuple(mel_fb.shape)}, expected "
                    f"{expected_shape} for n_fft={n_fft}, n_mels={n_mels} -- falling back "
                    f"to the torchaudio-reconstructed filterbank instead."
                )
                mel_fb = torchaudio.functional.melscale_fbanks(
                    n_freqs=n_fft // 2 + 1, f_min=fmin, f_max=fmax, n_mels=n_mels,
                    sample_rate=sr, norm=None, mel_scale="htk",
                )
        else:
            mel_fb = torchaudio.functional.melscale_fbanks(
                n_freqs=n_fft // 2 + 1,
                f_min=fmin,
                f_max=fmax,
                n_mels=n_mels,
                sample_rate=sr,
                norm=None,
                mel_scale="htk",
            )  # (n_freqs, n_mels)
        self.register_buffer("mel_fb", mel_fb, persistent=False)

        # SpecAugment, applied train-time only
        if freqm == 0:
            self.freqm = nn.Identity()
        else:
            self.freqm = torchaudio.transforms.FrequencyMasking(freqm, iid_masks=True)
        if timem == 0:
            self.timem = nn.Identity()
        else:
            self.timem = torchaudio.transforms.TimeMasking(timem, iid_masks=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, samples)
        # torch.stft with center=False uses n_fft (not win_length) as the
        # frame size for computing frame count -- the window is zero-padded
        # within each n_fft-sized frame. So padding must be computed against
        # n_fft to land on exactly 500 frames for a 160000-sample clip:
        #   n_frames = (padded_len - n_fft) // hop + 1
        #   for n_frames=500: padded_len = 499*320 + 1024 = 160704
        target_frames = -(-x.shape[-1] // self.hopsize)  # ceil(L / hop)
        required_len = (target_frames - 1) * self.hopsize + self.n_fft
        pad_amount = max(required_len - x.shape[-1], 0)
        x = torch.nn.functional.pad(x, (0, pad_amount))

        stft = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hopsize,
            win_length=self.n_fft,  # window is already n_fft-length (left-aligned, zero-padded)
            window=self.window,
            center=False,  # uncentered, per the paper spec
            return_complex=True,
        )  # (batch, freq, time)

        magnitude = torch.sqrt(stft.real**2 + stft.imag**2 + 1e-12)  # magnitude, NOT power
        magnitude = magnitude / self.window_sum  # scaled by reciprocal of window sum
        magnitude = magnitude.transpose(1, 2)  # (batch, time, freq)

        mel = magnitude @ self.mel_fb  # (batch, time, n_mels)

        log_mel = torch.log(mel + self.log_floor)
        out = log_mel * self.log_scalar  # (batch, time, freq)

        if self.training:
            # torchaudio's masking transforms expect (..., freq, time), the
            # opposite of our (batch, time, freq) layout -- transpose in,
            # mask, transpose back.
            out = out.transpose(1, 2)  # (batch, freq, time)
            out = self.freqm(out)
            out = self.timem(out)
            out = out.transpose(1, 2)  # back to (batch, time, freq)

        return out
