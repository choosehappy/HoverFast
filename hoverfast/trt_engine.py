#!/usr/bin/env python3
"""TensorRT engine lifecycle for HoverFast.

This module owns every TensorRT-specific concern so that the rest of the
pipeline never crashes when the compiled engine is missing or was built for a
different machine:

* :func:`tensorrt_available` probes whether ``torch_tensorrt`` + the TensorRT
  runtime can be imported.
* :func:`load_eager_model` builds the plain PyTorch network from a
  ``.safetensors``/``.pth`` checkpoint. This is the portable fallback that works
  on any machine.
* :func:`load_engine` loads a compiled ``.ts`` engine.
* :func:`build_engine` compiles an engine for the *current* GPU and is exposed
  through the ``HoverFast build`` sub-command.
* :func:`resolve_model` is the single entry point used by inference: it prefers
  the compiled engine, and degrades gracefully to eager PyTorch with an explicit
  message telling the user how to build a compatible engine.
"""

from __future__ import annotations

import json
import os
from importlib.util import find_spec
from typing import Any

import torch

#: Default file name of the compiled TensorRT engine (matches historical name).
DEFAULT_ENGINE_NAME = "unet_trt.ts"
#: Environment variable that overrides the engine location.
ENGINE_ENV_VAR = "HOVERFAST_TRT_ENGINE"

DEFAULT_MIN_BATCH = 1
DEFAULT_OPT_BATCH = 7
DEFAULT_MAX_BATCH = 16
DEFAULT_WORKSPACE_BYTES = 8 << 30

#: Model tensor shape expected by the exported graph (N, C, H, W).
_INPUT_CHANNELS = 3
_INPUT_SIZE = 1024


def _engine_path_from_env(default: str | None = None) -> str:
    """Return the engine path, honouring ``HOVERFAST_TRT_ENGINE`` if set."""
    return os.environ.get(ENGINE_ENV_VAR) or default or DEFAULT_ENGINE_NAME


def tensorrt_available() -> bool:
    """Return ``True`` when ``torch_tensorrt`` and TensorRT can be imported."""
    if find_spec("torch_tensorrt") is None or find_spec("tensorrt") is None:
        return False
    try:
        import tensorrt  # noqa: F401
        import torch_tensorrt  # noqa: F401
    except Exception:
        return False
    return True


def _config_from_safetensors(model_path: str) -> dict[str, Any]:
    """Read the HoverFast constructor config stored in safetensors metadata."""
    from safetensors import safe_open

    with safe_open(model_path, framework="pt") as f:
        metadata = f.metadata() or {}
    if "config" not in metadata:
        raise ValueError(f"No 'config' metadata found in {model_path!r}.")
    config: dict[str, Any] = json.loads(metadata["config"])
    return config


def load_eager_model(model_path: str, device: torch.device) -> torch.nn.Module:
    """Load HoverFast as a plain PyTorch module (no TensorRT required).

    Supports both the modern ``.safetensors`` checkpoints and legacy ``.pth``
    training checkpoints that embed a ``model_dict`` plus the model config.
    """
    from .hoverfast import HoverFast

    if model_path.endswith(".safetensors"):
        import safetensors.torch

        config = _config_from_safetensors(model_path)
        model = HoverFast(**config).to(device, memory_format=torch.channels_last)  # type: ignore[call-overload]
        safetensors.torch.load_model(model, model_path)
    else:
        try:
            state = torch.load(model_path, map_location="cpu", weights_only=False)
        except TypeError:  # pragma: no cover - older torch without weights_only
            state = torch.load(model_path, map_location="cpu")
        if not isinstance(state, dict) or "model_dict" not in state:
            raise ValueError(
                f"Unsupported checkpoint format in {model_path!r}; expected a training "
                "checkpoint containing 'model_dict' or a .safetensors file."
            )
        config_keys = ("n_classes", "in_channels", "padding", "depth", "wf", "up_mode", "batch_norm", "conv_block")
        config = {key: state[key] for key in config_keys if key in state}
        model = HoverFast(**config).to(device, memory_format=torch.channels_last)  # type: ignore[call-overload]
        model.load_state_dict(state["model_dict"])

    return model.half().eval()  # type: ignore[no-any-return]


def load_engine(engine_path: str) -> Any:
    """Load a compiled TensorRT torchscript engine.

    Raises whatever the underlying loader raises (``OSError`` for missing
    shared libraries, ``RuntimeError`` for an engine built on another machine)
    so :func:`resolve_model` can decide how to recover.
    """
    import ctypes

    from .wsi_model import _find_tensorrt_libs

    nvinfer_path, trt_runtime_path = _find_tensorrt_libs()
    ctypes.CDLL(nvinfer_path, mode=ctypes.RTLD_GLOBAL)
    torch.ops.load_library(trt_runtime_path)  # type: ignore[no-untyped-call]
    return torch.jit.load(engine_path)  # type: ignore[no-untyped-call]


def _compile_dynamic_engine(
    model: torch.nn.Module,
    device: torch.device,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
    workspace_bytes: int,
) -> tuple[Any, torch.Tensor]:
    """Export ``model`` and compile it with a dynamic batch dimension.

    Returns the compiled engine together with the example input used for the
    export (required when serialising the engine).
    """
    import torch_tensorrt

    batch = torch.export.Dim("batch", min=min_batch, max=max_batch)
    example_input = torch.randn(
        opt_batch, _INPUT_CHANNELS, _INPUT_SIZE, _INPUT_SIZE, device=device, dtype=torch.float16
    )
    exp_program = torch.export.export(model, (example_input,), dynamic_shapes={"x": {0: batch}})

    trt_model = torch_tensorrt.dynamo.compile(
        exp_program,
        inputs=[
            torch_tensorrt.Input(
                min_shape=(min_batch, _INPUT_CHANNELS, _INPUT_SIZE, _INPUT_SIZE),
                opt_shape=(opt_batch, _INPUT_CHANNELS, _INPUT_SIZE, _INPUT_SIZE),
                max_shape=(max_batch, _INPUT_CHANNELS, _INPUT_SIZE, _INPUT_SIZE),
                dtype=torch.half,
            )
        ],
        enabled_precisions={torch.half},
        optimization_level=5,
        workspace_size=workspace_bytes,
        use_python_runtime=False,
    )
    return trt_model, example_input


def build_engine(
    model_path: str,
    engine_path: str | None = None,
    device: torch.device | None = None,
    min_batch: int = DEFAULT_MIN_BATCH,
    opt_batch: int = DEFAULT_OPT_BATCH,
    max_batch: int = DEFAULT_MAX_BATCH,
    workspace_bytes: int = DEFAULT_WORKSPACE_BYTES,
) -> str:
    """Compile a TensorRT engine for the current GPU and save it to disk.

    Returns the path of the written engine. Requires ``torch_tensorrt``.

    TensorRT must pick a convolution tactic for the *largest* dynamic profile.
    On GPUs with limited VRAM the tactic for a large ``max_batch`` can exceed
    the available memory, causing ``compile`` to abort with an obscure
    "Could not find any implementation" internal error. When that happens the
    maximum batch size is halved and compilation is retried, so the command
    still produces a usable engine instead of failing outright.
    """
    if not tensorrt_available():
        raise RuntimeError(
            "TensorRT is not available. Install it first, e.g. "
            "`pip install tensorrt torch-tensorrt` in an environment matching your "
            "PyTorch/CUDA build, then re-run `HoverFast build`."
        )

    engine_path = _engine_path_from_env(engine_path)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if opt_batch > max_batch:
        raise ValueError(f"opt_batch ({opt_batch}) must not exceed max_batch ({max_batch}).")

    import torch_tensorrt

    model = load_eager_model(model_path, device)

    effective_max = max_batch
    while True:
        try:
            trt_model, example_input = _compile_dynamic_engine(
                model, device, min_batch, opt_batch, effective_max, workspace_bytes
            )
            break
        except Exception as exc:  # noqa: BLE001 - retry with a smaller profile
            if effective_max <= opt_batch:
                raise RuntimeError(
                    f"Failed to compile a TensorRT engine (min={min_batch}, "
                    f"opt={opt_batch}, max={effective_max}). Last error: {exc}"
                ) from exc
            effective_max = max(opt_batch, effective_max // 2)
            print(
                f"[HoverFast] TensorRT compilation failed for max_batch={effective_max * 2}; "
                f"retrying with max_batch={effective_max}."
            )
            torch.cuda.empty_cache()

    torch_tensorrt.save(trt_model, engine_path, inputs=[example_input], output_format="torchscript")
    del example_input
    torch.cuda.empty_cache()
    return engine_path


def _build_hint(model_path: str, engine_path: str) -> str:
    return f"To build an engine for this machine run:\n    HoverFast build -m {model_path} -o {engine_path}"


def resolve_model(
    model_path: str,
    device: torch.device,
    engine_path: str | None = None,
    allow_eager_fallback: bool = True,
) -> Any:
    """Return a runnable model, preferring the compiled TensorRT engine.

    Resolution order:

    1. If a compiled engine exists and loads, it is returned.
    2. If it exists but cannot be loaded (e.g. built for another GPU / driver /
       TensorRT version), a clear message is printed and, when
       ``allow_eager_fallback`` is set, eager PyTorch is used instead.
    3. If no engine exists, a build hint is printed and eager PyTorch is used.

    ``allow_eager_fallback=False`` turns the mismatch into a ``RuntimeError``
    so callers can enforce TensorRT explicitly.
    """
    engine_path = _engine_path_from_env(engine_path)

    if os.path.exists(engine_path):
        try:
            model = load_engine(engine_path)
            print(f"[HoverFast] Loaded TensorRT engine: {engine_path}")
            return model
        except Exception as exc:  # noqa: BLE001 - any loader failure means "incompatible"
            message = (
                f"[HoverFast] TensorRT engine {engine_path!r} could not be loaded on this "
                f"machine ({type(exc).__name__}: {exc}). It was probably compiled on a "
                f"different GPU/driver/TensorRT version."
            )
            if not allow_eager_fallback:
                raise RuntimeError(f"{message}\n{_build_hint(model_path, engine_path)}") from exc
            print(message)
            print(f"[HoverFast] Falling back to eager PyTorch (slower). {_build_hint(model_path, engine_path)}")
    elif not allow_eager_fallback:
        raise RuntimeError(f"No TensorRT engine found at {engine_path!r}.\n{_build_hint(model_path, engine_path)}")
    else:
        print(
            f"[HoverFast] No compiled TensorRT engine at {engine_path!r}; using eager PyTorch "
            f"(slower).\n{_build_hint(model_path, engine_path)}"
        )

    return load_eager_model(model_path, device)


def build_main(args: Any) -> None:
    """Entry point for the ``HoverFast build`` sub-command."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not tensorrt_available():
        raise SystemExit(
            "TensorRT is not available in this environment. Install `tensorrt` and "
            "`torch-tensorrt` matching your PyTorch/CUDA build before running `HoverFast build`."
        )
    print(f"[HoverFast] Compiling TensorRT engine for device: {device}")
    engine_path = build_engine(
        model_path=args.model_path,
        engine_path=args.engine_path,
        device=device,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        workspace_bytes=args.workspace_gb << 30,
    )
    print(f"[HoverFast] Engine written to: {engine_path}")
