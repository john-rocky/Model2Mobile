"""Recipe: Force contiguous memory layout to fix channels_last tracing issues.

Some models use ``channels_last`` memory format or call ``.contiguous()``
in ways that break ``torch.jit.trace`` or coremltools conversion.
This recipe forces all parameters to contiguous layout and wraps the
model forward to ensure inputs are contiguous before processing.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from model2mobile.convert.recipes.base import Recipe, RecipeResult

logger = logging.getLogger(__name__)


class _ContiguousWrapper(nn.Module):
    """Wraps a model so that inputs are forced to contiguous layout."""

    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        x = x.contiguous()
        return self.inner(x, *args, **kwargs)


def _force_params_contiguous(model: nn.Module) -> int:
    """Make every parameter and buffer contiguous. Return count of fixed tensors."""
    fixed = 0
    for name, param in model.named_parameters():
        if not param.is_contiguous():
            param.data = param.data.contiguous()
            fixed += 1
    for name, buf in model.named_buffers():
        if not buf.is_contiguous():
            buf.data = buf.data.contiguous()
            fixed += 1
    return fixed


def _has_channels_last_params(model: nn.Module) -> bool:
    """Check if any parameter uses a non-contiguous / channels_last layout."""
    for p in model.parameters():
        if not p.is_contiguous():
            return True
    for b in model.buffers():
        if not b.is_contiguous():
            return True
    return False


class ChannelLastFixRecipe(Recipe):
    name = "channel_last_fix"

    def match(self, model: nn.Module, architecture: str, error: str | None = None) -> bool:
        if error:
            err_lower = error.lower()
            if "contiguous" in err_lower or "memory_format" in err_lower or "channels_last" in err_lower:
                return True

        return _has_channels_last_params(model)

    def apply(self, model: nn.Module) -> RecipeResult:
        modifications: list[str] = []

        # Force all parameters / buffers to contiguous
        fixed = _force_params_contiguous(model)
        if fixed:
            modifications.append(f"Forced {fixed} tensors to contiguous layout")

        # Convert model away from channels_last format if applied globally
        try:
            model.to(memory_format=torch.contiguous_format)  # type: ignore[call-overload]
            modifications.append("Called model.to(memory_format=contiguous_format)")
        except Exception:
            pass

        # Wrap the forward method to ensure input is contiguous
        original_forward = model.forward

        def _contiguous_forward(x: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            x = x.contiguous()
            return original_forward(x, *args, **kwargs)

        model.forward = _contiguous_forward  # type: ignore[assignment]
        modifications.append("Wrapped forward to force contiguous input")

        return RecipeResult(
            applied=len(modifications) > 0,
            recipe_name=self.name,
            description="Fixed channels_last / contiguous memory layout issues",
            modifications=modifications,
        )
