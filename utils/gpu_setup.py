"""
Shared GPU helpers: pick a sensible default device, enable cuDNN autotune / TF32
on CUDA, and move tensors to device with non-blocking copies when using CUDA.

Set environment variable SAE_DETERMINISTIC=1 to disable cuDNN benchmark (more
reproducible, often slower).
"""

from __future__ import annotations

import os
from typing import Optional, Union

import torch


def default_torch_device_str() -> str:
    """Prefer CUDA when available; otherwise CPU."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def resolve_device_str(explicit: Optional[str]) -> str:
    """
    Resolve CLI device string. None / empty -> auto.
    If user asks for CUDA but it is unavailable, fall back to CPU with a warning.
    """
    if explicit is None or str(explicit).strip() == "":
        return default_torch_device_str()
    s = str(explicit).strip()
    if s.lower().startswith("cuda") and not torch.cuda.is_available():
        import warnings

        warnings.warn(
            f'Device "{explicit}" was requested but CUDA is not available; using CPU.',
            UserWarning,
            stacklevel=2,
        )
        return "cpu"
    return s


def configure_cuda_performance() -> None:
    """
    When CUDA is active: enable cuDNN autotune (good for steady-shape inference)
    and TF32 for matmul on Ampere+ (faster, small numeric difference vs FP32).
    """
    if not torch.cuda.is_available():
        return
    det = os.environ.get("SAE_DETERMINISTIC", "").lower() in (
        "1",
        "true",
        "yes",
    )
    torch.backends.cudnn.benchmark = not det
    if hasattr(torch.backends, "cuda") and hasattr(
        torch.backends.cuda, "matmul"
    ):
        try:
            torch.backends.cuda.matmul.allow_tf32 = not det
        except Exception:
            pass
    if hasattr(torch.backends, "cudnn"):
        try:
            torch.backends.cudnn.allow_tf32 = not det
        except Exception:
            pass


def tensor_to_device_fast(
    x: torch.Tensor,
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Move tensor to device; use non-blocking H2D when target is CUDA."""
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type != "cuda":
        return x.to(dev)
    return x.to(dev, non_blocking=True)
