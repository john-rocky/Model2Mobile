"""Recipe: Patch GroupNorm with a manual implementation.

GroupNorm can cause issues when ``num_groups`` does not divide the
channel count evenly after model modifications, or on certain
coremltools versions.  This recipe replaces every GroupNorm with an
equivalent manual implementation (reshape -> per-group normalisation
-> reshape).  When ``groups == 1`` it falls back to LayerNorm which
is broadly supported.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from model2mobile.convert.recipes.base import Recipe, RecipeResult

logger = logging.getLogger(__name__)


class _ManualGroupNorm(nn.Module):
    """GroupNorm expressed with reshape + instance-norm style ops."""

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C = x.shape[:2]
        spatial = x.shape[2:]

        if self.num_groups == 1:
            # Equivalent to LayerNorm over C and spatial dims
            dims = list(range(1, x.dim()))
            mean = x.mean(dim=dims, keepdim=True)
            var = x.var(dim=dims, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + self.eps)
        else:
            G = self.num_groups
            # (N, G, C//G, *spatial)
            x = x.reshape(N, G, C // G, *spatial)
            dims = list(range(2, x.dim()))
            mean = x.mean(dim=dims, keepdim=True)
            var = x.var(dim=dims, keepdim=True, unbiased=False)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = x.reshape(N, C, *spatial)

        if self.affine and self.weight is not None and self.bias is not None:
            shape = [1, C] + [1] * len(spatial)
            x = x * self.weight.reshape(shape) + self.bias.reshape(shape)

        return x


def _from_group_norm(gn: nn.GroupNorm) -> _ManualGroupNorm:
    """Create a manual replacement from an existing GroupNorm."""
    manual = _ManualGroupNorm(
        num_groups=gn.num_groups,
        num_channels=gn.num_channels,
        eps=gn.eps,
        affine=gn.affine,
    )
    if gn.affine and gn.weight is not None and gn.bias is not None:
        manual.weight = nn.Parameter(gn.weight.data.clone())
        manual.bias = nn.Parameter(gn.bias.data.clone())
    return manual


class GroupNormPatchRecipe(Recipe):
    name = "group_norm_patch"

    def match(self, model: nn.Module, architecture: str, error: str | None = None) -> bool:
        if error:
            err_lower = error.lower()
            if "group_norm" in err_lower or "groupnorm" in err_lower or "divisible" in err_lower:
                return True

        for m in model.modules():
            if isinstance(m, nn.GroupNorm):
                return True
        return False

    def apply(self, model: nn.Module) -> RecipeResult:
        replaced = 0
        named = dict(model.named_modules())

        for name, module in list(named.items()):
            if not isinstance(module, nn.GroupNorm):
                continue

            replacement = _from_group_norm(module)

            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = named[parts[0]]
                setattr(parent, parts[1], replacement)
            elif len(parts) == 1:
                setattr(model, parts[0], replacement)

            replaced += 1

        return RecipeResult(
            applied=replaced > 0,
            recipe_name=self.name,
            description=f"Replaced {replaced} GroupNorm modules with manual implementation",
            modifications=[f"GroupNorm -> manual reshape+norm ({replaced} instances)"],
        )
