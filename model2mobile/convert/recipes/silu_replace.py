"""Recipe: Replace SiLU variants that fail in older coremltools.

Some models use custom SiLU implementations (x * sigmoid(x)) or
torch.nn.SiLU which can fail on older coremltools versions.
This recipe replaces them with a trace-friendly wrapper.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model2mobile.convert.recipes.base import Recipe, RecipeResult


class _TraceFriendlySiLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SiLUReplaceRecipe(Recipe):
    name = "silu_replace"

    def match(self, model: nn.Module, architecture: str, error: str | None = None) -> bool:
        if error and "silu" in error.lower():
            return True
        # Also match pre-emptively if model contains SiLU modules
        for m in model.modules():
            if type(m).__name__ == "SiLU":
                return True
        return False

    def apply(self, model: nn.Module) -> RecipeResult:
        replaced = 0
        for name, module in model.named_modules():
            if type(module).__name__ == "SiLU":
                # Walk parent to replace child
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = dict(model.named_modules())[parts[0]]
                    setattr(parent, parts[1], _TraceFriendlySiLU())
                elif len(parts) == 1:
                    setattr(model, parts[0], _TraceFriendlySiLU())
                replaced += 1

        return RecipeResult(
            applied=replaced > 0,
            recipe_name=self.name,
            description=f"Replaced {replaced} SiLU modules with trace-friendly version",
            modifications=[f"SiLU -> x*sigmoid(x) ({replaced} instances)"],
        )
