"""Recipe: Strip YOLO Detect head and return raw feature maps.

YOLO models (v5/v8/v9/v10/v11) have a Detect head that performs
anchor decoding and produces variable-length output unsuitable for
Core ML static graphs.  This recipe replaces the Detect forward to
return the raw convolution outputs (feature maps) so that decoding
and NMS can run on-device with dedicated Swift/Obj-C code.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from model2mobile.convert.recipes.base import Recipe, RecipeResult

logger = logging.getLogger(__name__)

_DETECT_KEYWORDS = ("Detect", "DetectionModel", "YOLO")


def _has_detect_module(model: nn.Module) -> bool:
    """Return True if any sub-module looks like a YOLO Detect head."""
    for m in model.modules():
        cls_name = type(m).__name__
        if any(kw in cls_name for kw in _DETECT_KEYWORDS):
            return True
    return False


class _RawDetectForwardV5:
    """Replacement forward for YOLOv5-style Detect (self.m convolutions)."""

    def __init__(self, original: nn.Module) -> None:
        self.module = original

    def __call__(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for i, feat in enumerate(x):
            outputs.append(self.module.m[i](feat))
        return outputs


class _RawDetectForwardV8:
    """Replacement forward for YOLOv8-style Detect (self.cv2, self.cv3)."""

    def __init__(self, original: nn.Module) -> None:
        self.module = original

    def __call__(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for i, feat in enumerate(x):
            box = self.module.cv2[i](feat)
            cls = self.module.cv3[i](feat)
            outputs.append(torch.cat([box, cls], dim=1))
        return outputs


def _patch_detect_module(model: nn.Module) -> list[str]:
    """Find and patch Detect modules, returning a list of modifications."""
    modifications: list[str] = []

    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "Detect" not in cls_name:
            continue

        # YOLOv8+ style: has cv2 and cv3 sequential blocks
        if hasattr(module, "cv2") and hasattr(module, "cv3"):
            wrapper = _RawDetectForwardV8(module)
            module.forward = wrapper  # type: ignore[assignment]
            modifications.append(f"{name} ({cls_name}): replaced forward with raw cv2+cv3 outputs")
            logger.info("Patched YOLOv8-style Detect at '%s'", name)

        # YOLOv5 style: has self.m ModuleList of convolutions
        elif hasattr(module, "m") and isinstance(module.m, nn.ModuleList):
            wrapper_v5 = _RawDetectForwardV5(module)
            module.forward = wrapper_v5  # type: ignore[assignment]
            modifications.append(f"{name} ({cls_name}): replaced forward with raw conv outputs")
            logger.info("Patched YOLOv5-style Detect at '%s'", name)

        # Disable any in-place dynamic flag that guards export paths
        if hasattr(module, "dynamic"):
            module.dynamic = False
        if hasattr(module, "export"):
            module.export = True
        if hasattr(module, "end2end"):
            module.end2end = False

    return modifications


class YOLODetectHeadRecipe(Recipe):
    name = "yolo_detect_head"

    def match(self, model: nn.Module, architecture: str, error: str | None = None) -> bool:
        # Match by architecture name
        if any(kw.lower() in architecture.lower() for kw in _DETECT_KEYWORDS):
            return True
        # Match by error message hinting at list / variable output
        if error and ("list" in error.lower() and "output" in error.lower()):
            return True
        # Match by presence of a Detect-like module
        return _has_detect_module(model)

    def apply(self, model: nn.Module) -> RecipeResult:
        modifications = _patch_detect_module(model)
        return RecipeResult(
            applied=len(modifications) > 0,
            recipe_name=self.name,
            description="Replaced YOLO Detect head forward with raw feature-map outputs",
            modifications=modifications,
        )
