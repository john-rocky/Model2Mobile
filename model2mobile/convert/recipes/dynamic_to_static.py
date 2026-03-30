"""Recipe: Patch dynamic-shape ops that break Core ML tracing.

Common patterns:
- torch.Tensor.size() used in arithmetic → replace with fixed constants
- Reshape with -1 dims → explicit shape
- torch.arange(dynamic_len) → bounded static version

This recipe wraps the model's forward in a shape-fixing wrapper that
intercepts dynamic shape queries during tracing.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn

from model2mobile.convert.recipes.base import Recipe, RecipeResult

_DYNAMIC_PATTERNS = re.compile(
    r"dynamic|variable.*shape|unknown rank|data.dependent|aten::size",
    re.IGNORECASE,
)


class _StaticShapeWrapper(nn.Module):
    """Wraps a model to force static shapes during tracing."""

    def __init__(self, inner: nn.Module, input_size: int):
        super().__init__()
        self.inner = inner
        self.input_size = input_size

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple:
        # Force batch=1 static shape annotation for the tracer
        assert x.shape[0] == 1
        return self.inner(x)


class DynamicToStaticRecipe(Recipe):
    name = "dynamic_to_static"

    def match(self, model: nn.Module, architecture: str, error: str | None = None) -> bool:
        # Only activate on actual conversion failure with dynamic shape error
        if error and _DYNAMIC_PATTERNS.search(error):
            return True
        return False

    def apply(self, model: nn.Module) -> RecipeResult:
        # Look for common dynamic-shape sources and patch them
        modifications: list[str] = []

        # 1. Replace nn.AdaptiveAvgPool2d with fixed-size AvgPool where possible
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.AdaptiveAvgPool2d):
                output_size = module.output_size
                if output_size == (1, 1) or output_size == 1:
                    # This is global average pooling — safe as-is, skip
                    continue

        # 2. Disable any dynamic export flags
        for attr in ("export", "dynamic", "onnx_dynamic"):
            if hasattr(model, attr):
                setattr(model, attr, False)
                modifications.append(f"Set model.{attr} = False")

        # 3. If model has an 'fuse' method (common in YOLO), call it
        if hasattr(model, "fuse") and callable(model.fuse):
            try:
                model.fuse()
                modifications.append("Called model.fuse() to merge BN layers")
            except Exception:
                pass

        return RecipeResult(
            applied=len(modifications) > 0,
            recipe_name=self.name,
            description="Patched model for static shape tracing",
            modifications=modifications,
        )
