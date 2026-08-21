"""
Perch v2 backbone via direct ONNX graph conversion (onnx2torch), instead
of the hand-rebuilt timm-based reconstruction in model.py.

This is the RECOMMENDED backbone for most users. Where model.py's
Perch2Backbone required manually reconstructing the architecture in timm
and finding/fixing individual mismatches against Google's original
(stem padding, missing conv bias -- see model.py's docstrings for that
investigation), this module converts the actual ONNX computation graph
directly into PyTorch modules, node by node. Nothing is reconstructed
from assumption, so there's no architectural guesswork left to get
wrong: confirmed 1.00000000 cosine similarity against ONNX Runtime's own
output on real audio, and fully differentiable (verified: gradients flow
to every parameter).

Trade-off versus model.py's Perch2Backbone: this doesn't produce a clean
`conv_stem` / `blocks[i]` / `bn1`-style module hierarchy -- parameters
are named by their position in the original graph (`node_Conv_23.weight`
etc), which isn't human-readable on its own. Partial/selective
fine-tuning (freezing specific blocks while training others) is
supported here via `PerchONNXBackbone.freeze_up_to_block()`, which
reconstructs block-level structure from the graph's own tensor-naming
metadata -- see build_block_map() below for how.

Requires the `onnx` extras: `pip install -e ".[onnx]"` (onnx2torch,
onnxscript, onnx, onnxruntime). These are NOT required for the rest of
this package -- only import from this module if you're using this
backbone specifically.
"""

import re
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

_MBCONV_PATTERN = re.compile(r"MBConv_(\d+)")
_HEAD_PATTERN = re.compile(r"Head_0")


class _OnnxDFTRfft(nn.Module):
    """
    Custom converter target for ONNX's DFT op @ opset 17 (onesided=1,
    forward only -- this graph's actual usage; see
    _register_custom_converters()'s docstring for the full story).

    IMPORTANT: this class must live at MODULE level, not nested inside a
    function. torch.save() uses pickle, which cannot serialize a class
    defined inside a function's local scope -- it needs a class to be
    resolvable via a fixed `module.ClassName` path to reconstruct it
    later. An earlier version defined this inside
    _register_custom_converters(), which worked fine for building and
    running the model but broke the moment convert_perch_onnx's
    cache_dir tried to torch.save() a model containing an instance of
    it: `AttributeError: Can't get local object
    '_register_custom_converters.<locals>._OnnxDFTRfft'`. Caught on a
    real Perch v2 conversion (which has exactly one DFT node, in the
    frontend) -- none of the synthetic test files used during
    development happened to include a DFT node, so this was invisible
    until tested against the real graph. See _FusedAffine below for the
    identical fix applied to the BatchNorm-fusion path.
    """

    def __init__(self, axis: int):
        super().__init__()
        self.axis = axis

    def forward(self, input_tensor: torch.Tensor, dft_length: torch.Tensor = None) -> torch.Tensor:
        original_ndim = input_tensor.ndim
        positive_axis = self.axis if self.axis >= 0 else original_ndim + self.axis
        x = input_tensor.squeeze(-1)
        n = int(dft_length.item()) if dft_length is not None else x.shape[positive_axis]
        rfft_out = torch.fft.rfft(x, n=n, dim=positive_axis)
        return torch.stack([rfft_out.real, rfft_out.imag], dim=-1)


class _FusedAffine(nn.Module):
    """
    Fused y = x * mul_const + add_const via a single torch.addcmul call,
    used by _fuse_batchnorm_pairs() below to replace each ONNX
    Mul(const)->Add(const) pair (the pattern representing every
    BatchNorm in the Perch v2 graph).

    IMPORTANT: must live at MODULE level for the same pickling reason as
    _OnnxDFTRfft above -- see that class's docstring. This one was
    originally nested inside _fuse_batchnorm_pairs() and had the exact
    same "Can't get local object" failure the first time cache_dir was
    actually exercised against a real model.
    """

    def __init__(self, mul_const: torch.Tensor, add_const: torch.Tensor):
        super().__init__()
        # Keep the ORIGINAL shapes exactly as stored in the graph -- do
        # NOT flatten/reshape. Whatever axis these already broadcast
        # against in the source graph is preserved exactly, regardless
        # of whether that's NCHW or NHWC at this point (see
        # _fuse_batchnorm_pairs()'s docstring for why this matters: an
        # earlier F.batch_norm-based version assumed channels always sit
        # at dim 1, which crashed on real Perch v2 data).
        self.mul_const = nn.Parameter(mul_const.clone())
        self.add_const = nn.Parameter(add_const.clone())

    def forward(self, x):
        return torch.addcmul(self.add_const, x, self.mul_const)


def _require_onnx_extras():
    try:
        import onnx  # noqa: F401
        import onnx2torch  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "The ONNX-converted backbone requires extra dependencies not "
            "installed by default. Run:\n\n"
            '    pip install -e ".[onnx]"\n\n'
            "or: pip install onnx onnx2torch onnxscript onnxruntime"
        ) from e


def _clean_and_reinfer_shapes(onnx_path: str) -> str:
    """
    Clears stale `value_info` shape metadata and re-runs ONNX shape
    inference before conversion. Perch v2's released ONNX graph carries
    leftover shape annotations from an earlier export/optimization pass
    that conflict with what fresh shape inference actually computes --
    onnx2torch's own internal shape-inference step fails outright on the
    raw file without this cleanup (confirmed: "Inferred shape and
    existing shape differ..." error). Returns a path to a cleaned temp
    copy; the original file is never modified.
    """
    import onnx

    model = onnx.load(onnx_path)
    del model.graph.value_info[:]
    try:
        model = onnx.shape_inference.infer_shapes(model, strict_mode=False)
    except Exception as e:  # pragma: no cover -- defensive, see module docstring
        warnings.warn(f"Shape re-inference raised an error even after clearing stale "
                       f"metadata: {e}. Proceeding with the uncleaned model.")

    import tempfile
    tmp_path = tempfile.NamedTemporaryFile(suffix=".onnx", delete=False).name
    onnx.save(model, tmp_path)
    return tmp_path


def _register_custom_converters():
    """
    Registers converters for three op(version) combinations onnx2torch
    doesn't support out of the box, found while converting Perch v2's
    actual graph:

    - Pad @ opset 18: onnx2torch has Pad at v2/11/13 but not 18. Pad's
      core behavior hasn't meaningfully changed since v11, so the
      existing dynamic-pad handler is safely reused. Verified against a
      standalone opset-18 Pad test case.

    - DFT @ opset 17: NOT implemented at any version -- a genuine gap,
      not a version-registration one. This graph's one DFT node
      (axis=-2, onesided=1, inverse unset/forward) maps directly onto
      torch.fft.rfft; implemented and verified against ONNX Runtime's
      own DFT op directly (max abs diff ~1.9e-6, i.e. float32 precision).
      NOTE: this implementation only handles the onesided=1, inverse=0
      case this specific graph uses, not the general DFT op.

    - ReduceL2 / ReduceMax @ opset 18: onnx2torch's existing converters
      only read `axes` from a node ATTRIBUTE (the pre-opset-18
      convention). ONNX moved `axes` to an optional second INPUT for the
      Reduce family at opset 18. Confirmed via direct node inspection
      that this graph's ReduceL2/ReduceMax nodes both pass axes as a
      CONSTANT input, not an attribute -- naively reusing the old
      converter would silently reduce over the wrong axes (or all axes)
      rather than erroring. This extracts the constant axes value and
      reuses the existing, already-trusted reduction math.
    """
    from onnx2torch.node_converters.registry import add_converter
    from onnx2torch.utils.common import (
        OnnxMapping, OperationConverterResult, get_const_value, onnx_mapping_from_node,
    )

    # --- Pad @ v18 ---
    from onnx2torch.node_converters.pad import OnnxPadDynamic, _onnx_to_torch_mode

    @add_converter(operation_type="Pad", version=18)
    def _pad_v18(node, graph):  # pylint: disable=unused-argument
        mode = _onnx_to_torch_mode(node.attributes.get("mode", "constant"))
        return OperationConverterResult(
            torch_module=OnnxPadDynamic(mode=mode),
            onnx_mapping=OnnxMapping(inputs=node.input_values, outputs=node.output_values),
        )

    # --- DFT @ v17 (onesided=1, forward only -- this graph's actual usage) ---
    # _OnnxDFTRfft is defined at module level (not here) -- see its own
    # docstring for why (torch.save/pickle compatibility).

    @add_converter(operation_type="DFT", version=17)
    def _dft_v17(node, graph):  # pylint: disable=unused-argument
        axis = node.attributes.get("axis", 1)
        onesided = node.attributes.get("onesided", 0)
        inverse = node.attributes.get("inverse", 0)
        if onesided != 1 or inverse != 0:
            raise NotImplementedError(
                f"This DFT converter only handles onesided=1, inverse=0 -- "
                f"got onesided={onesided}, inverse={inverse}."
            )
        return OperationConverterResult(
            torch_module=_OnnxDFTRfft(axis=axis),
            onnx_mapping=OnnxMapping(inputs=node.input_values, outputs=node.output_values),
        )

    # --- ReduceL2 / ReduceMax @ v18 (axes as input, not attribute) ---
    from onnx2torch.node_converters.reduce import OnnxReduceStaticAxes

    def _make_reduce_converter(operation_type: str):
        def _converter(node, graph):
            keepdims = node.attributes.get("keepdims", 1)
            axes = node.attributes.get("axes", None)
            if axes is None and len(node.input_values) == 2:
                axes = get_const_value(node.input_values[1], graph).tolist()
                return OperationConverterResult(
                    torch_module=OnnxReduceStaticAxes(
                        operation_type=operation_type, axes=axes, keepdims=keepdims,
                    ),
                    onnx_mapping=OnnxMapping(
                        inputs=(node.input_values[0],), outputs=node.output_values,
                    ),
                )
            return OperationConverterResult(
                torch_module=OnnxReduceStaticAxes(
                    operation_type=operation_type, axes=axes, keepdims=keepdims,
                ),
                onnx_mapping=onnx_mapping_from_node(node=node),
            )
        return _converter

    add_converter(operation_type="ReduceL2", version=18)(_make_reduce_converter("ReduceL2"))
    add_converter(operation_type="ReduceMax", version=18)(_make_reduce_converter("ReduceMax"))


_converters_registered = False


def _ensure_converters_registered():
    global _converters_registered
    if not _converters_registered:
        _register_custom_converters()
        _converters_registered = True


def _fuse_batchnorm_pairs(gm) -> int:
    """
    Finds Mul(const)->Add(const) chains in the converted graph -- the
    pattern onnx2torch produces for every BatchNorm in the Perch v2 graph
    (ONNX represents eval-mode BN as two separate ops, not one fused
    kernel) -- and replaces each pair with a single fused affine call via
    `torch.addcmul` (computes `add_const + x * mul_const` as one kernel
    launch instead of two separate elementwise ops).

    IMPORTANT: this does NOT assume channels sit at any particular
    dimension. An earlier version used `F.batch_norm`, which hard-codes
    channels at dim 1 (NCHW) -- this broke on real Perch v2 data with
    `RuntimeError: running_mean should contain 249 elements not 40`,
    because the tensor at this point in the graph is actually NHWC
    (channel-last), not NCHW: 249 is the stem's spatial height, not a
    channel count, and F.batch_norm silently matched against the wrong
    dimension in a way that only errors when the sizes happen to
    mismatch (an earlier synthetic test used NCHW-shaped data by
    construction and passed by accident, not because the logic was
    actually layout-safe). `addcmul` uses plain broadcasting against the
    constants' ORIGINAL shape exactly as stored in the graph, making no
    assumption about which axis is "channels" -- verified correct for
    both NCHW- and NHWC-style broadcasting.

    This backbone has ~80 BatchNorms (stem + 3/block x 26 blocks + head).
    Verified NOT to touch genuine two-tensor adds (e.g. residual
    connections) -- only pairs where BOTH the Mul's second operand and
    the Add's second operand are graph constants (get_attr nodes, not
    runtime tensors) are fused. Benchmarked on a synthetic 80-pair chain
    at Perch-v2-realistic tensor sizes: ~1.33x speedup on CPU (this
    number belongs to the current addcmul-based implementation
    specifically -- an earlier F.batch_norm-based version measured
    ~1.87x on the same benchmark before being found to crash on real
    Perch v2 data with a shape mismatch, see the class docstring above;
    don't conflate the two numbers). IMPORTANT: on the ACTUAL real Perch
    v2 graph, this synthetic benchmark's speedup did NOT reproduce --
    fused and unfused forward-pass times were statistically
    indistinguishable in repeated real-data testing. This function still
    works correctly (numerically exact) and is left available via
    fuse_batchnorm=True, but is not the default -- see
    convert_perch_onnx()'s docstring for the full story on why a
    synthetic microbenchmark result didn't transfer.

    Mutates `gm` in place and returns the number of pairs fused.
    """
    import torch.fx as fx
    from onnx2torch.node_converters.binary_math_operations import OnnxBinaryMathOperation

    # _FusedAffine is defined at module level (not here) -- see its own
    # docstring for why (torch.save/pickle compatibility).

    def _get_attr_value(gm, target):
        obj = gm
        for part in target.split("."):
            obj = getattr(obj, part)
        return obj

    graph = gm.graph
    fused_count = 0

    for node in list(graph.nodes):
        if node.op != "call_module":
            continue
        module = gm.get_submodule(node.target)
        if not isinstance(module, OnnxBinaryMathOperation) or module.math_op_function is not torch.mul:
            continue

        users = list(node.users)
        if len(users) != 1:
            continue
        add_node = users[0]
        if add_node.op != "call_module":
            continue
        add_module = gm.get_submodule(add_node.target)
        if not isinstance(add_module, OnnxBinaryMathOperation) or add_module.math_op_function is not torch.add:
            continue

        data_arg, const_arg = None, None
        for a in node.args:
            if isinstance(a, fx.Node) and a.op == "get_attr":
                const_arg = a
            else:
                data_arg = a
        if const_arg is None or data_arg is None:
            continue  # not a const-multiply -- e.g. two runtime tensors

        add_const_arg = None
        for a in add_node.args:
            if a is node:
                continue
            if isinstance(a, fx.Node) and a.op == "get_attr":
                add_const_arg = a
        if add_const_arg is None:
            continue  # e.g. a residual add of two runtime tensors, not a BN

        mul_const = _get_attr_value(gm, const_arg.target)
        add_const = _get_attr_value(gm, add_const_arg.target)
        if mul_const is None or add_const is None or mul_const.numel() != add_const.numel():
            continue

        fused = _FusedAffine(mul_const, add_const)
        fused_name = f"fused_bn_{fused_count}"
        gm.add_module(fused_name, fused)

        # Preserve the Mul node's onnx_mapping (its data-path input carries
        # the real scoped tensor name, e.g. ".../MBConv_5/ExpandConv/...")
        # onto the fused replacement -- without this, build_block_map()'s
        # scope-extraction can't find it on the new module (it's a fresh
        # node with no graph history of its own), and would silently
        # misclassify every fused parameter as 'other', breaking
        # freeze_up_to_block() for nearly the whole backbone. Verified
        # this actually matters and actually works, not assumed.
        original_mapping = getattr(module, "onnx_mapping", None)
        if original_mapping is not None:
            fused.onnx_mapping = original_mapping

        with graph.inserting_before(node):
            new_node = graph.call_module(fused_name, args=(data_arg,))
        add_node.replace_all_uses_with(new_node)
        graph.erase_node(add_node)
        graph.erase_node(node)
        fused_count += 1

    graph.eliminate_dead_code()
    gm.recompile()
    return fused_count


def _cache_key(onnx_path: str, fuse_batchnorm: bool) -> str:
    """
    Cheap cache-invalidation key: file size + modification time (not a
    full content hash, to avoid reading a potentially large file just to
    validate a cache hit) plus the fuse_batchnorm setting. If the source
    .onnx file changes in any way that updates its mtime, or the fusion
    setting differs, this produces a different key -- the cache is never
    silently reused across a changed input. Worst case failure mode is
    an unnecessary re-conversion (e.g. if a file's mtime changes without
    its content changing), never a stale/wrong result being loaded.
    """
    st = Path(onnx_path).stat()
    return f"{st.st_size}_{int(st.st_mtime)}_fuse{int(fuse_batchnorm)}.pt"


def convert_perch_onnx(onnx_path: str, fuse_batchnorm: bool = False, cache_dir: str = None):
    """
    Converts perch_v2.onnx directly to a PyTorch model via onnx2torch.
    Returns the raw converted `fx.GraphModule` with all 4 of Perch v2's
    named outputs (embedding, spatial_embedding, spectrogram, label).

    fuse_batchnorm=False (default) uses the raw, unmodified onnx2torch
    conversion. When True, replaces every Mul->Add pair representing a
    BatchNorm with a single fused `torch.addcmul` kernel call instead of
    two separate ops -- a real, verified-correct transformation (exact
    numerical match before/after, see _fuse_batchnorm_pairs() above for
    how it's detected and confirmed safe), but measured on real Perch v2
    data (with cache_dir already removing construction time from the
    picture) to give NO reliable forward-pass speedup: fused and raw
    forward-pass times were statistically indistinguishable across
    repeated runs (~0.19-0.29s either way). Defaulted to False on that
    basis -- it adds real complexity (this whole graph-rewriting pass,
    plus two real bugs it caused and required fixing: an NCHW-axis
    assumption that crashed on the stem's actual NHWC data, and a
    pickling incompatibility that broke cache_dir) for a benefit that
    isn't showing up in practice on this workload. Left available, not
    removed, in case it helps in a different environment (e.g. GPU,
    where kernel-launch overhead may matter more than measured here on
    CPU) -- but not the recommended default.

    cache_dir: if given, caches the fully-converted PyTorch model to
    disk after the first conversion. THIS is what actually addresses
    slow embedding-extraction time, not fuse_batchnorm -- measured
    construction cost on the real Perch v2 graph (1137 nodes) is ~6.5s,
    which appears to be onnx2torch's inherent per-node Python-level
    conversion overhead (roughly linear in node count; pruning the graph
    to just the 'embedding' output was tested and only removes ~1% of
    nodes, so isn't a meaningful lever here -- the backbone itself, not
    the unused classification head, is almost the entire graph). Caching
    turns that cost into a one-time tax across the cache file's lifetime
    instead of once per process/notebook-kernel-restart -- measured
    ~0.5-0.9s to load the real (large) cached model on a kernel restart,
    still a large improvement over the ~6.5-8s uncached cost. Cache key
    incorporates the source file's size+mtime and the fuse_batchnorm
    setting (see _cache_key above) -- changing either automatically
    invalidates the cache rather than silently reusing a stale
    conversion.

    Most users want PerchONNXEmbedder or PerchONNXClassifier below
    instead of calling this directly -- they wrap this with a clean
    embed()-style interface matching the rest of this package.
    """
    _require_onnx_extras()
    _ensure_converters_registered()

    if cache_dir:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir_path / _cache_key(onnx_path, fuse_batchnorm)
        if cache_path.exists():
            try:
                return torch.load(cache_path, weights_only=False)
            except Exception as e:
                # Self-healing: a corrupted cache file (e.g. left over
                # from a crash during a previous save -- this happened
                # for real, not hypothetically: an earlier version of
                # this module used classes that couldn't be pickled,
                # which crashed mid-write and left a truncated .pt file
                # at exactly this path, causing every subsequent run to
                # fail trying to load it) should never permanently break
                # this function. Warn, discard the bad file, and fall
                # through to a fresh conversion below instead of raising.
                warnings.warn(
                    f"Cached model at {cache_path} failed to load ({type(e).__name__}: {e}) "
                    f"-- likely a corrupted/incomplete cache file (e.g. from an interrupted "
                    f"save). Discarding it and re-converting from scratch."
                )
                cache_path.unlink(missing_ok=True)

    from onnx2torch import convert

    cleaned_path = _clean_and_reinfer_shapes(onnx_path)
    gm = convert(cleaned_path, attach_onnx_mapping=True)
    if fuse_batchnorm:
        n_fused = _fuse_batchnorm_pairs(gm)
        if n_fused == 0:
            warnings.warn(
                "fuse_batchnorm=True but 0 Mul->Add pairs were fused -- expected "
                "~80 for the real Perch v2 backbone. Either this isn't the real "
                "graph, or onnx2torch's internal representation of Mul/Add has "
                "changed since this was written. Verify before trusting the "
                "speedup claim for this specific run."
            )

    if cache_dir:
        # Atomic write: save to a temp file in the same directory, then
        # rename into place only after the save fully succeeds. os.replace
        # is atomic on both POSIX and Windows -- a crash or interruption
        # DURING the save can now never leave a corrupted file sitting at
        # the real cache_path (which is exactly what happened before this
        # fix: an un-picklable class crashed mid-save and left a truncated
        # file at cache_path itself, breaking every subsequent load).
        import os
        tmp_fd, tmp_path = tempfile.mkstemp(dir=str(cache_dir_path), suffix=".tmp")
        os.close(tmp_fd)
        try:
            torch.save(gm, tmp_path)
            os.replace(tmp_path, cache_path)
        except Exception:
            Path(tmp_path).unlink(missing_ok=True)
            raise

    return gm


# --- known exceptions to the automatic block-scope extraction below ---
# node_Conv_7 is the frontend's own framing/windowing conv (not part of
# the backbone). node_Conv_8 is the stem -- confirmed via direct
# inspection to carry NO scope info anywhere in its own onnx_mapping
# inputs (unlike every block conv and the head conv, whose Transpose
# predecessor's output tensor name carries full scope, e.g.
# ".../MBConv_5/DepthwiseConv/..." or ".../Head_0/Conv_0/..."). If this
# model is ever re-exported differently, re-verify these two by
# inspecting the ONNX graph's node names/inputs directly (via onnx.load
# and walking model.graph.node) before trusting them.
_KNOWN_FRONTEND_NODE = "node_Conv_7"
_KNOWN_STEM_NODE = "node_Conv_8"


def build_block_map(model) -> dict:
    """
    Maps each parameter name in a converted model to a block label:
    'frontend', 'stem', 'block_0'..'block_25', 'head', or 'other'
    (unrecognized -- e.g. the classification head's prototype-network
    internals, not needed for backbone fine-tuning).

    Needed because the raw converted model's parameters are named purely
    by graph creation order (node_Conv_7, node_Conv_8, ...) with zero
    architectural meaning -- this reconstructs it from each module's
    onnx_mapping metadata (the original ONNX input tensor names, which
    for blocks/head carry full scope info).

    Some convs (confirmed: blocks 5/8/18's depthwise convs, which have
    asymmetric padding in the real ONNX graph -- see model.py's
    _fix_downsampling_block_padding docstring for the same finding on
    the timm-based backbone) get wrapped by onnx2torch in an internal
    nn.Sequential(Pad, Conv), where the mapping lives on the PARENT
    Sequential, not the child holding the actual weight/bias. This walks
    up to the nearest ancestor with a mapping rather than requiring it on
    the exact same module.
    """
    mapping_by_module = {}
    for module_name, module in model.named_modules():
        onnx_mapping = getattr(module, "onnx_mapping", None)
        if onnx_mapping is not None:
            mapping_by_module[module_name] = onnx_mapping

    def _nearest_mapping(module_name: str):
        parts = module_name.split(".")
        for i in range(len(parts), 0, -1):
            candidate = ".".join(parts[:i])
            if candidate in mapping_by_module:
                return mapping_by_module[candidate]
        return None

    block_map = {}
    for module_name, module in model.named_modules():
        params = list(module.named_parameters(recurse=False))
        if not params:
            continue

        onnx_mapping = _nearest_mapping(module_name)
        label = "other"
        if onnx_mapping is not None:
            if module_name == _KNOWN_FRONTEND_NODE or module_name.startswith(_KNOWN_FRONTEND_NODE + "."):
                label = "frontend"
            elif module_name == _KNOWN_STEM_NODE or module_name.startswith(_KNOWN_STEM_NODE + "."):
                label = "stem"
            else:
                input_names = " ".join(onnx_mapping.inputs)
                mbconv_match = _MBCONV_PATTERN.search(input_names)
                if mbconv_match:
                    label = f"block_{mbconv_match.group(1)}"
                elif _HEAD_PATTERN.search(input_names):
                    label = "head"

        for param_name, _ in params:
            full_name = f"{module_name}.{param_name}" if param_name else module_name
            block_map[full_name] = label

    n_blocks = len({l for l in block_map.values() if l.startswith("block_")})
    if n_blocks != 26:
        warnings.warn(
            f"build_block_map found {n_blocks} blocks, expected 26 -- partial "
            f"fine-tuning via freeze_up_to_block() may not behave as expected. "
            f"Verify by inspecting the ONNX graph directly (e.g. via onnx.load "
            f"and walking model.graph.node) before trusting this."
        )

    return block_map


class PerchONNXEmbedder(nn.Module):
    """
    Frozen-feature extractor using the ONNX-converted backbone --
    confirmed 1.00000000 cosine similarity against ONNX Runtime's own
    output. Takes raw waveform directly (the converted graph includes
    its own frontend/frame extraction, unlike model.py's
    PerchFrontend + Perch2Backbone split).

    NOTE ON SPEED: this backbone was initially ~19x slower than the
    native timm-based one for a single frozen-embedding call. That gap
    turned out to be almost entirely ONE-TIME GRAPH CONSTRUCTION cost
    (~6.5-8s to convert the real ~1137-node graph), not forward-pass
    cost (measured ~0.15-0.3s either way, before or after any of the
    fixes below) -- use cache_dir to make that one-time cost a one-time
    cost across the cache file's lifetime instead of paid on every
    process/notebook-kernel-restart; this is the fix that actually
    matters for this backbone's speed. A decomposed-BatchNorm theory
    (see fuse_batchnorm on convert_perch_onnx) was tested directly and
    found to give no reliable forward-pass speedup on real data --
    defaulted off on that basis, see its own docstring for the full
    story. `compile=True` below is a separate lever, useful for a real
    training run with many repeated calls, not for a one-off call.

    Example:
        embedder = PerchONNXEmbedder("weights/perch_v2.onnx", cache_dir="weights/onnx_cache")
        embedder.eval()
        with torch.no_grad():
            emb = embedder(waveform_batch)  # (batch, 1536)

        # for many repeated calls within a training loop (a separate
        # lever from cache_dir, which is about construction cost):
        embedder = PerchONNXEmbedder("weights/perch_v2.onnx", compile=True)
    """

    def __init__(self, onnx_path: str, compile: bool = False, compile_kwargs: dict = None,
                 fuse_batchnorm: bool = False, cache_dir: str = None):
        super().__init__()
        self._model = convert_perch_onnx(onnx_path, fuse_batchnorm=fuse_batchnorm, cache_dir=cache_dir)
        if compile:
            self._model = torch.compile(self._model, **(compile_kwargs or {}))
        self.eval()

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._model(x)
        # onnx2torch preserves the graph's 4 named outputs
        # (embedding, spatial_embedding, spectrogram, label) as a tuple,
        # in that order.
        return out[0] if isinstance(out, (tuple, list)) else out


class PerchONNXBackbone(nn.Module):
    """
    Trainable ONNX-converted backbone with partial fine-tuning support --
    the ONNX-graph-conversion counterpart to model.py's Perch2Backbone,
    with genuinely 1.00000000-fidelity weights and gradients confirmed
    flowing to every parameter (unlike a frozen inference-only ONNX/
    TFLite export).

    NOTE ON SPEED: see PerchONNXEmbedder's docstring above -- the ~19x
    original slowdown was almost entirely one-time construction cost,
    not forward-pass cost; use cache_dir. `compile=True` has real
    one-time compilation cost on the first forward/backward pass --
    worth it for a real training run (many steps), actively
    counterproductive for a single one-off call.

    Example:
        backbone = PerchONNXBackbone("weights/perch_v2.onnx", cache_dir="weights/onnx_cache")
        backbone.freeze_up_to_block(13)  # freeze stem + blocks 0-12
        emb = backbone(waveform_batch)   # (batch, 1536)

        # for a real training run (amortizes the one-time compile cost
        # over many steps -- a separate lever from cache_dir):
        backbone = PerchONNXBackbone("weights/perch_v2.onnx", compile=True)
    """

    def __init__(self, onnx_path: str, compile: bool = False, compile_kwargs: dict = None,
                 fuse_batchnorm: bool = False, cache_dir: str = None):
        super().__init__()
        self._model = convert_perch_onnx(onnx_path, fuse_batchnorm=fuse_batchnorm, cache_dir=cache_dir)
        self.block_map = build_block_map(self._model)
        if compile:
            self._model = torch.compile(self._model, **(compile_kwargs or {}))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._model(x)
        return out[0] if isinstance(out, (tuple, list)) else out

    def freeze_up_to_block(self, n: int, freeze_stem: bool = True, freeze_frontend: bool = True):
        """
        Freezes stem (if freeze_stem) and blocks 0..n-1, leaves blocks
        n..25 and head trainable. Anything labeled 'other' is left
        as-is. freeze_frontend is mostly a no-op here -- the frontend has
        no learnable parameters in this model (windowing/filterbank are
        fixed).

        Returns (frozen_count, trainable_count).
        """
        frozen, trainable = 0, 0
        for name, param in self._model.named_parameters():
            label = self.block_map.get(name, "other")

            should_freeze = False
            if label == "stem" and freeze_stem:
                should_freeze = True
            elif label == "frontend" and freeze_frontend:
                should_freeze = True
            elif label.startswith("block_"):
                should_freeze = int(label.split("_")[1]) < n

            param.requires_grad = not should_freeze
            if should_freeze:
                frozen += 1
            else:
                trainable += 1

        return frozen, trainable

    def unfreeze_all(self):
        """Sets requires_grad=True on every parameter (full fine-tuning)."""
        for param in self._model.parameters():
            param.requires_grad = True

    def freeze_all(self):
        """Sets requires_grad=False on every parameter (frozen-feature use)."""
        for param in self._model.parameters():
            param.requires_grad = False


class PerchONNXClassifier(nn.Module):
    """
    PerchONNXBackbone + a linear head, mirroring model.py's
    Perch2Classifier interface (mode="frozen"/"linear_probe"/"finetune")
    but built on the ONNX-converted backbone.

    NOTE ON SPEED: see PerchONNXBackbone's docstring -- pass
    compile=True for a real training run, leave it off for one-off use,
    and pass cache_dir to avoid repeating the one-time graph-conversion
    cost across runs (the fix that matters most for this backbone's
    speed -- see PerchONNXEmbedder's docstring for the full story).

    Example:
        model = PerchONNXClassifier(
            num_classes=42,
            onnx_path="weights/perch_v2.onnx",
            mode="linear_probe",
            cache_dir="weights/onnx_cache",
        )
    """

    VALID_MODES = ("frozen", "linear_probe", "finetune")

    def __init__(self, num_classes: int, onnx_path: str, mode: str = "finetune",
                 compile: bool = False, compile_kwargs: dict = None,
                 fuse_batchnorm: bool = False, cache_dir: str = None):
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")

        self.mode = mode
        self.backbone = PerchONNXBackbone(onnx_path, compile=compile, compile_kwargs=compile_kwargs,
                                           fuse_batchnorm=fuse_batchnorm, cache_dir=cache_dir)
        self.head = nn.Linear(1536, num_classes)

        if mode in ("frozen", "linear_probe"):
            self.backbone.freeze_all()
        else:
            self.backbone.unfreeze_all()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode in ("frozen", "linear_probe"):
            with torch.no_grad():
                emb = self.backbone(x)
        else:
            emb = self.backbone(x)
        return self.head(emb)

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Raw pooled embeddings (no head, no grad)."""
        return self.backbone(x)
