"""Recipe: Strip NMS and dynamic postprocessing from detection models.

Many detection models include NMS (Non-Maximum Suppression) inside
their forward pass.  NMS produces variable-length output and uses
ops that Core ML cannot convert.  This recipe detects NMS-containing
modules and replaces them with an identity passthrough, so the raw
detection tensor is exported and NMS can run natively on-device.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn

from model2mobile.convert.recipes.base import Recipe, RecipeResult

_NMS_PATTERNS = re.compile(
    r"nms|non_max|NonMaxSuppression|batched_nms|multiclass_nms",
    re.IGNORECASE,
)


class _Identity(nn.Module):
    def forward(self, *args: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if len(args) == 1:
            return args[0]
        return args


class NMSStripRecipe(Recipe):
    name = "nms_strip"

    def match(self, model: nn.Module, architecture: str, error: str | None = None) -> bool:
        # Match on conversion error mentioning NMS-related ops
        if error and _NMS_PATTERNS.search(error):
            return True
        # Match pre-emptively if model has NMS-like submodules
        for name, _ in model.named_modules():
            if _NMS_PATTERNS.search(name):
                return True
        return False

    def apply(self, model: nn.Module) -> RecipeResult:
        stripped: list[str] = []

        for name, module in list(model.named_modules()):
            if not _NMS_PATTERNS.search(name) and not _NMS_PATTERNS.search(type(module).__name__):
                continue

            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = dict(model.named_modules())[parts[0]]
                setattr(parent, parts[1], _Identity())
            elif len(parts) == 1 and name:
                setattr(model, parts[0], _Identity())
            stripped.append(name or type(module).__name__)

        return RecipeResult(
            applied=len(stripped) > 0,
            recipe_name=self.name,
            description=f"Stripped NMS from model: {stripped}",
            modifications=[f"Removed NMS module: {s}" for s in stripped],
        )
