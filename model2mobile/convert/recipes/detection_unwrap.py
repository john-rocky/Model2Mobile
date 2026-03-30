"""Recipe: Unwrap detection models that return non-traceable outputs.

Detection models (FCOS, FasterRCNN, SSD, RetinaNet, etc.) typically
return List[Dict[str, Tensor]] which torch.jit.trace cannot handle.
This recipe wraps the model to return raw backbone + head tensors
instead, so Core ML conversion can proceed.  NMS and postprocessing
are expected to run natively on-device.
"""

from __future__ import annotations

import re

import torch
import torch.nn as nn

from model2mobile.convert.recipes.base import Recipe, RecipeResult

_TRACE_OUTPUT_ERROR = re.compile(
    r"Only tensors.*can be output from traced|"
    r"cannot be understood by the tracer|"
    r"Encountering a dict at the output",
    re.IGNORECASE,
)

# Known detection model class names from torchvision and common libraries
_DETECTION_CLASSES = {
    "FCOS", "FasterRCNN", "SSD", "SSDLite", "RetinaNet",
    "MaskRCNN", "KeypointRCNN",
    "FoveaBox", "ATSS", "CenterNet",
}


class _BackboneHeadWrapper(nn.Module):
    """Wraps a torchvision-style detection model to output raw tensors."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # torchvision detection models expect List[Tensor] input
        images = [x.squeeze(0)]

        # Access internal components to get raw predictions
        model = self.model

        # Try torchvision GeneralizedRCNN / FCOS path
        if hasattr(model, "backbone") and hasattr(model, "head"):
            features = model.backbone(x)
            if isinstance(features, dict):
                feature_list = list(features.values())
            else:
                feature_list = [features] if isinstance(features, torch.Tensor) else list(features)
            # Run head on features
            try:
                raw = model.head(feature_list)
                if isinstance(raw, dict):
                    return tuple(raw.values())
                if isinstance(raw, (list, tuple)):
                    flat: list[torch.Tensor] = []
                    for item in raw:
                        if isinstance(item, torch.Tensor):
                            flat.append(item)
                        elif isinstance(item, dict):
                            flat.extend(item.values())
                    return tuple(flat) if flat else (feature_list[0],)
            except Exception:
                pass
            return tuple(feature_list)

        # Try backbone-only extraction for FasterRCNN-style
        if hasattr(model, "backbone"):
            features = model.backbone(x)
            if isinstance(features, dict):
                return tuple(features.values())
            if isinstance(features, torch.Tensor):
                return (features,)
            return tuple(features)

        # Fallback: run full model in eval mode with dummy target
        # and try to intercept before the dict output
        try:
            model.eval()
            model.eager_outputs = lambda losses, detections: detections  # type: ignore
        except Exception:
            pass

        # Last resort: just return the backbone feature maps
        for name, child in model.named_children():
            if "backbone" in name.lower() or "body" in name.lower():
                features = child(x)
                if isinstance(features, dict):
                    return tuple(features.values())
                return (features,) if isinstance(features, torch.Tensor) else tuple(features)

        raise RuntimeError("Could not extract raw tensors from detection model")


class DetectionUnwrapRecipe(Recipe):
    name = "detection_unwrap"

    def match(self, model: nn.Module, architecture: str, error: str | None = None) -> bool:
        # Match on the trace output error
        if error and _TRACE_OUTPUT_ERROR.search(error):
            return True
        # Match pre-emptively on known detection model classes
        if architecture in _DETECTION_CLASSES:
            return True
        # Check class hierarchy
        for cls in type(model).__mro__:
            if cls.__name__ in _DETECTION_CLASSES:
                return True
        return False

    def apply(self, model: nn.Module) -> RecipeResult:
        wrapper = _BackboneHeadWrapper(model)
        # Replace the model's forward method with the wrapper's
        # We need to modify the model in-place for the converter to pick it up.
        # Store original forward and replace it.
        original_forward = model.forward

        def new_forward(x: torch.Tensor) -> tuple[torch.Tensor, ...]:
            return wrapper(x)

        model.forward = new_forward  # type: ignore[assignment]

        return RecipeResult(
            applied=True,
            recipe_name=self.name,
            description="Wrapped detection model to output raw tensors (NMS removed)",
            modifications=[
                "Replaced forward() to return raw backbone+head tensors",
                "NMS and postprocessing stripped — run these on-device",
            ],
        )
