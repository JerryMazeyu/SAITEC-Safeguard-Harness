from __future__ import annotations

from typing import Any


def resolve_torch_device(device: str | None = "auto") -> str:
    """Resolve an explicit or automatic torch device string."""
    if device is not None and str(device).strip().casefold() != "auto":
        return str(device)

    torch = import_torch()
    if _npu_available(torch):
        return "npu:0"
    if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        return "cuda:0"
    if _mps_available(torch):
        return "mps"
    return "cpu"


def import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "local model inference requires PyTorch. Install the local-model dependencies for this runtime."
        ) from exc
    return torch


def coerce_torch_dtype(torch: Any, value: str | None) -> Any:
    if value is None:
        return None
    normalized = value.strip().casefold()
    if normalized == "auto":
        return "auto"
    aliases = {
        "bf16": "bfloat16",
        "fp16": "float16",
        "fp32": "float32",
    }
    dtype_name = aliases.get(normalized, normalized)
    if not hasattr(torch, dtype_name):
        raise ValueError(f"unknown torch dtype {value!r}")
    return getattr(torch, dtype_name)


def default_model_dtype(torch: Any, device: str) -> Any:
    if device.startswith(("cuda", "npu")) and hasattr(torch, "bfloat16"):
        return torch.bfloat16
    return torch.float32


def patch_broken_triton_namespace() -> None:
    try:
        import triton  # type: ignore
    except Exception:
        return

    if hasattr(triton, "language"):
        return

    import types

    class _DummyDType:
        pass

    triton.language = types.SimpleNamespace(dtype=_DummyDType)


def _npu_available(torch: Any) -> bool:
    if not hasattr(torch, "npu"):
        try:
            import torch_npu  # noqa: F401
        except Exception:
            return False
    npu = getattr(torch, "npu", None)
    return bool(npu is not None and hasattr(npu, "is_available") and npu.is_available())


def _mps_available(torch: Any) -> bool:
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    return bool(mps is not None and hasattr(mps, "is_available") and mps.is_available())

