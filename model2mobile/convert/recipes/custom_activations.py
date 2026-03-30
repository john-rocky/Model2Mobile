"""Recipe: Replace unsupported custom activations.

Some models use activation functions not supported by older coremltools
(Mish, GELU with tanh approximation, etc.).  This recipe replaces them
with equivalent formulations built from elementary ops.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from model2mobile.convert.recipes.base import Recipe, RecipeResult

logger = logging.getLogger(__name__)


class _TraceFriendlyMish(nn.Module):
    """Mish(x) = x * tanh(softplus(x))."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.tanh(torch.nn.functional.softplus(x))


class _TraceFriendlyGELU(nn.Module):
    """GELU approximated as x * sigmoid(1.702 * x).

    This is the sigmoid-based approximation that avoids the erf / tanh
    ops which some coremltools versions cannot convert.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


def _is_mish(module: nn.Module) -> bool:
    cls_name = type(module).__name__
    return "Mish" in cls_name


def _is_gelu(module: nn.Module) -> bool:
    cls_name = type(module).__name__
    return cls_name == "GELU"


class CustomActivationsRecipe(Recipe):
    name = "custom_activations"

    def match(self, model: nn.Module, architecture: str, error: str | None = None) -> bool:
        if error:
            err_lower = error.lower()
            if "mish" in err_lower or "gelu" in err_lower:
                return True

        for m in model.modules():
            if _is_mish(m) or _is_gelu(m):
                return True
        return False

    def apply(self, model: nn.Module) -> RecipeResult:
        replaced_mish = 0
        replaced_gelu = 0

        named = dict(model.named_modules())

        for name, module in list(named.items()):
            replacement: nn.Module | None = None

            if _is_mish(module):
                replacement = _TraceFriendlyMish()
                replaced_mish += 1
            elif _is_gelu(module):
                replacement = _TraceFriendlyGELU()
                replaced_gelu += 1

            if replacement is not None:
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = named[parts[0]]
                    setattr(parent, parts[1], replacement)
                elif len(parts) == 1:
                    setattr(model, parts[0], replacement)

        total = replaced_mish + replaced_gelu
        mods: list[str] = []
        if replaced_mish:
            mods.append(f"Mish -> x*tanh(softplus(x)) ({replaced_mish} instances)")
        if replaced_gelu:
            mods.append(f"GELU -> x*sigmoid(1.702*x) ({replaced_gelu} instances)")

        return RecipeResult(
            applied=total > 0,
            recipe_name=self.name,
            description=f"Replaced {total} custom activation modules",
            modifications=mods,
        )
