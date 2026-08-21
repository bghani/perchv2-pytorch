"""
PyTorch reimplementation of the Perch v2 backbone (converted from Google's
JAX/Flax checkpoint into a timm tf_efficientnet_b3 state dict) plus a
thin classification wrapper supporting three usage modes:

  "frozen"        -- backbone fully frozen; use Perch2Classifier.embed()
                      or Perch2Embedder to pull fixed 1536-dim features
                      for a downstream classifier (sklearn, XGBoost, a
                      separate small net, etc). No head is trained here.
  "linear_probe"  -- backbone frozen (requires_grad=False, forced .eval()
                      so BatchNorm running stats don't drift), only the
                      new linear head is trained. Fast, low-data-friendly.
  "finetune"       -- backbone unfrozen, backbone + head trained jointly.
                      Slowest and most data-hungry, but the highest
                      ceiling if your domain is far from Perch's training
                      distribution.

All three share the same architecture: stock EfficientNet-B3
(in_chans=1, single-channel log-mel input -- not the ImageNet 3-channel
stem) with the classification head stripped (num_classes=0) so the
backbone returns a pooled 1536-dim embedding.
"""

import torch
import torch.nn as nn
import timm
import warnings
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent / "data"


def _fix_stem_padding(backbone: nn.Module) -> None:
    """
    Replace timm's dynamically-"SAME"-padded conv_stem with one using
    true VALID (zero) padding, matching Perch v2's actual stem exactly.

    Discovered by direct inspection of the ONNX graph's stem Conv node
    (kernel_shape=[3,3], strides=[2,2], no `pads`/`auto_pad` attribute --
    which per the ONNX spec means zero padding on all sides, i.e. VALID).
    timm's tf_efficientnet_b3 instead gives conv_stem TF-style dynamic
    "SAME" padding (ceil(input/stride) output size), which produces a
    different output size for odd input dimensions like Perch's 500x128
    spectrogram: 250x64 (timm's SAME) vs 249x63 (Perch's actual VALID).
    That single-pixel spatial misalignment was found (via block-by-block
    bisection against the ONNX model) to propagate through the entire
    network, and was a major contributor to the previously-reported
    ~0.80 whole-network cosine similarity ceiling.

    This ALSO adds a bias term to conv_stem, loaded from
    data/conv_stem_bias.npy if present. This was a second, independent
    discovery: the ONNX graph's stem Conv node has THREE inputs (data,
    weight, bias), not two -- a genuine per-output-channel bias that
    timm's architecture (bias=False, relying on the following BatchNorm)
    doesn't have at all. See scripts/extract_missing_conv_biases.py for
    how this was found and how to regenerate the .npy file.
    """
    old_conv = backbone.conv_stem
    bias_path = _DATA_DIR / "conv_stem_bias.npy"
    has_extracted_bias = bias_path.exists()

    new_conv = nn.Conv2d(
        in_channels=old_conv.in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=0,  # VALID -- matches the ONNX graph's stem Conv node exactly
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=has_extracted_bias or (old_conv.bias is not None),
    )
    new_conv.weight = old_conv.weight

    if has_extracted_bias:
        import numpy as np
        bias_arr = np.load(bias_path)
        with torch.no_grad():
            new_conv.bias.copy_(torch.from_numpy(bias_arr.astype("float32")))
    elif old_conv.bias is not None:
        new_conv.bias = old_conv.bias
    else:
        warnings.warn(
            f"conv_stem_bias.npy not found at {bias_path} -- conv_stem will have no "
            f"bias term, which is known to be architecturally incorrect (the original "
            f"ONNX stem conv has a bias this repo was missing). This file should "
            f"already be present at legacy/data/conv_stem_bias.npy -- if it's "
            f"missing, something is wrong with this checkout (it needs to be "
            f"re-extracted from a real perch_v2.onnx graph's stem Conv node bias input)."
        )

    backbone.conv_stem = new_conv


def _fix_head_bias(backbone: nn.Module) -> None:
    """
    Adds a bias term to conv_head, loaded from data/conv_head_bias.npy if
    present -- the same missing-bias issue found at the stem also exists
    at the head (ONNX's head Conv node also has 3 inputs: data, weight,
    bias). See _fix_stem_padding's docstring and
    scripts/extract_missing_conv_biases.py for the full story.
    """
    old_conv = backbone.conv_head
    bias_path = _DATA_DIR / "conv_head_bias.npy"
    has_extracted_bias = bias_path.exists()

    if not has_extracted_bias and old_conv.bias is not None:
        return  # already has a bias from somewhere, nothing to do

    new_conv = nn.Conv2d(
        in_channels=old_conv.in_channels,
        out_channels=old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        dilation=old_conv.dilation,
        groups=old_conv.groups,
        bias=True,
    )
    new_conv.weight = old_conv.weight

    if has_extracted_bias:
        import numpy as np
        bias_arr = np.load(bias_path)
        with torch.no_grad():
            new_conv.bias.copy_(torch.from_numpy(bias_arr.astype("float32")))
    else:
        warnings.warn(
            f"conv_head_bias.npy not found at {bias_path} -- conv_head will have no "
            f"bias term, which is known to be architecturally incorrect. This file should "
            f"already be present at legacy/data/conv_head_bias.npy -- if it's "
            f"missing, something is wrong with this checkout (it needs to be "
            f"re-extracted from a real perch_v2.onnx graph's head Conv node bias input)."
        )
        with torch.no_grad():
            new_conv.bias.zero_()

    backbone.conv_head = new_conv


def _fix_downsampling_block_padding(backbone: nn.Module) -> None:
    """
    Replaces the depthwise conv in blocks 5, 8, and 18 with an explicit
    asymmetric-padding version matching Perch v2's actual ONNX graph.

    Discovered via scripts/audit_all_conv_nodes.py -- a systematic sweep
    of all 79 conv nodes in the ONNX graph, prompted by finding (via
    onnx2torch's own Conv converter, which splits asymmetric-padding
    convs into an explicit Sequential(Pad, Conv) since plain nn.Conv2d
    can't express asymmetric padding) that blocks 5/8/18's depthwise
    convs specifically have asymmetric `pads` baked into the ONNX graph:

        block 5  (node_Conv_23): onnx pads=[2,1,2,2] (H_begin,W_begin,H_end,W_end)
        block 8  (node_Conv_32): onnx pads=[1,0,1,1]
        block 18 (node_Conv_62): onnx pads=[1,1,2,2]

    timm's tf_efficientnet_b3 gives these convs dynamic "SAME"-style
    padding (Conv2dSame) instead, which is always symmetric by
    construction and can't represent this -- the same class of bug as
    the original stem padding issue, found by extending that
    investigation systematically across every conv in the network rather
    than checking layers one at a time. The full 79-conv audit confirmed
    these are the ONLY three convs in the whole backbone with asymmetric
    padding (block 2's depthwise conv, despite superficially looking
    similar, has symmetric padding and needs no fix).

    ONNX pads [H_begin, W_begin, H_end, W_end] map to PyTorch's
    nn.ZeroPad2d(left, right, top, bottom) = (W_begin, W_end, H_begin, H_end).
    """
    # block flat-index -> (onnx pads in [H_begin,W_begin,H_end,W_end] order)
    _ASYMMETRIC_BLOCKS = {
        5: [2, 1, 2, 2],
        8: [1, 0, 1, 1],
        18: [1, 1, 2, 2],
    }

    flat_idx = 0
    for stage in backbone.blocks:
        for sub_block in stage:
            if flat_idx in _ASYMMETRIC_BLOCKS:
                h_begin, w_begin, h_end, w_end = _ASYMMETRIC_BLOCKS[flat_idx]
                torch_pad = (w_begin, w_end, h_begin, h_end)  # (left, right, top, bottom)

                old_conv = sub_block.conv_dw
                new_conv = nn.Conv2d(
                    in_channels=old_conv.in_channels,
                    out_channels=old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride,
                    padding=0,  # explicit padding applied separately below
                    dilation=old_conv.dilation,
                    groups=old_conv.groups,
                    bias=old_conv.bias is not None,
                )
                new_conv.weight = old_conv.weight
                if old_conv.bias is not None:
                    new_conv.bias = old_conv.bias

                sub_block.conv_dw = nn.Sequential(nn.ZeroPad2d(torch_pad), new_conv)

            flat_idx += 1

    if flat_idx != 26:
        warnings.warn(
            f"_fix_downsampling_block_padding expected 26 flattened blocks, found "
            f"{flat_idx} -- the hardcoded block indices (5, 8, 18) may no longer be "
            f"correct for this timm version/config. Verify against "
            f"scripts/audit_all_conv_nodes.py before trusting this fix."
        )


class Perch2Backbone(nn.Module):
    """
    Stock EfficientNet-B3, single-channel input, no head -- matches
    Perch v2's architecture. Optionally loads converted Perch v2 weights.

    Note on fidelity: this is a community conversion from Google's
    JAX/Flax SavedModel, not an official PyTorch release. Per-block
    validation via direct comparison against the official ONNX release
    (see scripts/compare_pytorch_onnx_blocks.py) found the stem's padding
    convention didn't match Google's original (see _fix_stem_padding
    above) -- that mismatch is now corrected here. Re-run the block
    bisection after any further architecture changes to confirm current
    fidelity rather than trusting this docstring's numbers, which will
    go stale.
    """

    def __init__(self, weights_path: str = None, pretrained_timm: bool = False):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnet_b3",
            pretrained=pretrained_timm,
            in_chans=1,      # Perch's stem is single-channel, not RGB-style
            num_classes=0,   # strip classifier, return pooled embedding
        )
        _fix_stem_padding(self.backbone)
        _fix_head_bias(self.backbone)
        if weights_path:
            state = torch.load(weights_path, map_location="cpu")
            missing, unexpected = self.backbone.load_state_dict(state, strict=False)
            # 'classifier.*' in missing is expected (num_classes=0 strips
            # it) -- and if the checkpoint predates the conv bias fix
            # (see _fix_stem_padding/_fix_head_bias above), 'conv_stem.bias'
            # / 'conv_head.bias' being "missing" is also expected, since
            # those are injected from data/*.npy rather than the
            # checkpoint itself. Anything else showing up here is worth
            # knowing about, so it's surfaced as a warning instead of
            # always printing on every load.
            _expected_missing_prefixes = ("classifier.",)
            _expected_missing_exact = {"conv_stem.bias", "conv_head.bias"}
            unexpected_missing = [
                k for k in missing
                if not k.startswith(_expected_missing_prefixes)
                and k not in _expected_missing_exact
            ]
            if unexpected_missing or unexpected:
                warnings.warn(
                    f"[Perch2Backbone] state_dict mismatch loading {weights_path}: "
                    f"missing={unexpected_missing}, unexpected={unexpected}"
                )

        # IMPORTANT: this must run AFTER the checkpoint is loaded, not
        # before (unlike _fix_stem_padding/_fix_head_bias, which preserve
        # their module's top-level attribute name and so are safe to run
        # before loading -- load_state_dict just overwrites their
        # placeholder values in place). This fix wraps conv_dw in a NEW
        # nn.Sequential, which changes its state_dict key from
        # "conv_dw.weight" to "conv_dw.1.weight". Running this before
        # load_state_dict silently leaves these three blocks' depthwise
        # conv at random initialization forever (the checkpoint's old key
        # name never matches the new wrapped path) -- a real regression
        # caught by testing, not a hypothetical: it collapsed whole-
        # embedding cosine similarity to ~0.08. Running it after loading
        # means old_conv.weight/.bias already hold the real loaded values
        # at the moment they get copied into the new wrapper.
        # NOTE: _fix_downsampling_block_padding (see its definition above)
        # is intentionally NOT called here. It targets a real, confirmed
        # bug (blocks 5/8/18 use asymmetric padding in the actual ONNX
        # graph, not timm's symmetric auto-"SAME"), and an earlier
        # ordering bug in when it ran (before vs after checkpoint
        # loading) was found and fixed -- but even after that fix, it
        # regressed whole-embedding fidelity further (down to ~0.007),
        # not just failed to help. Given repeated regressions on this
        # specific fix, the reliable, validated configuration is WITHOUT
        # it: stem padding + stem/head bias + BN patch, landing at
        # ~0.97 whole-embedding cosine similarity, confirmed stable
        # across multiple real audio files. For fidelity closer to 1.0,
        # use the onnx2torch-converted backbone instead (see
        # scripts/try_onnx2torch.py and scripts/build_block_mapping.py),
        # which reaches 1.00000000 and also supports partial fine-tuning.
        # _fix_downsampling_block_padding(self.backbone)

        self.num_features = self.backbone.num_features  # 1536 for b3

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class Perch2Classifier(nn.Module):
    """
    PerchFrontend -> Perch2Backbone -> linear head, with a `mode` switch
    controlling what gets trained.

    Example:
        from legacy import PerchFrontend, Perch2Classifier

        mel = PerchFrontend()
        model = Perch2Classifier(
            num_classes=42,
            mel=mel,
            weights_path="weights/perch_v2_backbone_timm.pt",
            mode="linear_probe",
        )
    """

    VALID_MODES = ("frozen", "linear_probe", "finetune")

    def __init__(
        self,
        num_classes: int,
        mel: nn.Module,
        weights_path: str = None,
        mode: str = "finetune",
    ):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")

        self.mode = mode
        self.mel = mel
        self.backbone = Perch2Backbone(weights_path=weights_path)
        self.head = nn.Linear(self.backbone.num_features, num_classes)
        self._apply_mode()

    def _apply_mode(self):
        freeze = self.mode in ("frozen", "linear_probe")
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
        if freeze:
            self.backbone.eval()

    def train(self, mode: bool = True):
        """Override so backbone stays in eval() (frozen BN stats, no
        dropout) whenever mode is "frozen" or "linear_probe", even when
        the outer module is switched to train() for the head/optimizer."""
        super().train(mode)
        if self.mode in ("frozen", "linear_probe"):
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.mel(x)
        feats = feats.unsqueeze(1)  # (batch, 1, time, n_mels) -- single channel
        if self.mode in ("frozen", "linear_probe"):
            with torch.no_grad():
                emb = self.backbone(feats)
        else:
            emb = self.backbone(feats)
        return self.head(emb)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw pooled backbone embeddings (no head, no grad) --
        the entry point for pure frozen-feature-extraction workflows."""
        was_training = self.mel.training
        self.mel.eval()  # disable SpecAugment for embedding extraction
        feats = self.mel(x)
        feats = feats.unsqueeze(1)
        emb = self.backbone(feats)
        self.mel.train(was_training)
        return emb


class Perch2Embedder(nn.Module):
    """
    Minimal frozen-feature extractor with no classification head at all --
    for anyone who just wants Perch v2 embeddings (e.g. to feed into
    sklearn/XGBoost, nearest-neighbour search, or clustering) and doesn't
    need Perch2Classifier's train-mode plumbing.

    Example:
        embedder = Perch2Embedder("weights/perch_v2_backbone_timm.pt")
        embedder.eval()
        with torch.no_grad():
            emb = embedder(waveform_batch)  # (batch, 1536)
    """

    def __init__(self, weights_path: str = None):
        super().__init__()
        self.mel = None
        from .frontend import PerchFrontend  # local import avoids a cycle at module load

        self.mel = PerchFrontend()
        self.backbone = Perch2Backbone(weights_path=weights_path)
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.mel(x)
        feats = feats.unsqueeze(1)
        return self.backbone(feats)