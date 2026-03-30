"""Recipe: Replace deformable convolutions with standard Conv2d.

Deformable convolutions (DCN / DCNv2) are not supported by Core ML.
This recipe replaces them with a standard ``nn.Conv2d`` using the same
weight shape.  This is an *approximation* that discards the learned
offsets, so accuracy may differ from the original model.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from model2mobile.convert.recipes.base import Recipe, RecipeResult

logger = logging.getLogger(__name__)

_DEFORM_CLASS_NAMES = (
    "DeformConv2d",
    "DeformConv",
    "ModulatedDeformConv",
    "ModulatedDeformConv2d",
    "DCN",
    "DCNv2",
)


def _is_deformable(module: nn.Module) -> bool:
    cls_name = type(module).__name__
    return any(dc in cls_name for dc in _DEFORM_CLASS_NAMES)


def _conv2d_from_deform(module: nn.Module) -> nn.Conv2d:
    """Build a standard Conv2d that mirrors the deformable module's weight shape."""
    weight: torch.Tensor = module.weight  # type: ignore[union-attr]
    out_channels, in_channels_per_group = weight.shape[0], weight.shape[1]
    kernel_h, kernel_w = weight.shape[2], weight.shape[3]

    groups = getattr(module, "groups", 1)
    in_channels = in_channels_per_group * groups
    stride = getattr(module, "stride", (1, 1))
    padding = getattr(module, "padding", (0, 0))
    dilation = getattr(module, "dilation", (1, 1))
    bias = getattr(module, "bias", None) is not None

    # Normalise stride / padding / dilation that may be ints
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)

    conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=(kernel_h, kernel_w),
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )

    conv.weight = nn.Parameter(weight.data.clone())
    if bias and module.bias is not None:
        conv.bias = nn.Parameter(module.bias.data.clone())

    return conv


class DeformableConvRecipe(Recipe):
    name = "deformable_conv"

    def match(self, model: nn.Module, architecture: str, error: str | None = None) -> bool:
        if error and "deform" in error.lower():
            return True

        for m in model.modules():
            if _is_deformable(m):
                return True
        return False

    def apply(self, model: nn.Module) -> RecipeResult:
        replaced = 0
        named = dict(model.named_modules())

        for name, module in list(named.items()):
            if not _is_deformable(module):
                continue
            if not hasattr(module, "weight"):
                logger.warning("Deformable module '%s' has no weight; skipping", name)
                continue

            conv = _conv2d_from_deform(module)

            parts = name.rsplit(".", 1)
            if len(parts) == 2:
                parent = named[parts[0]]
                setattr(parent, parts[1], conv)
            elif len(parts) == 1:
                setattr(model, parts[0], conv)

            replaced += 1
            logger.warning(
                "Replaced deformable conv '%s' with standard Conv2d — accuracy may differ",
                name,
            )

        return RecipeResult(
            applied=replaced > 0,
            recipe_name=self.name,
            description=f"Replaced {replaced} deformable conv(s) with standard Conv2d (approximation)",
            modifications=[
                f"DeformConv -> Conv2d ({replaced} instances, accuracy may differ)"
            ],
        )
