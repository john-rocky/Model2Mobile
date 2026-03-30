"""Recipe: Prepare D-FINE models for CoreML conversion.

D-FINE models require several modifications before ``torch.jit.trace``:

1. **Deploy mode** — ``model.deploy()`` switches from training to inference
   graph (removes denoising, auxiliary heads, etc.).
2. **Project tensor fix** — The decoder's ``project`` parameter may be 1-D,
   but CoreML's linear op requires 2-D weights.  Unsqueeze fixes this.
3. **Output wrapping** — The raw model returns a dict with ``pred_logits``
   and ``pred_boxes``.  CoreML needs flat tensor outputs, so we wrap in a
   module that applies sigmoid (focal loss) or softmax (CE loss) and
   squeezes the batch dimension.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from model2mobile.convert.recipes.base import Recipe, RecipeResult

logger = logging.getLogger(__name__)

_DFINE_CLASS_NAMES = (
    "DFINE",
    "DFINEModel",
    "HybridEncoder",
    "DFINEDecoder",
)


def _is_dfine(model: nn.Module) -> bool:
    cls_name = type(model).__name__
    if any(tag in cls_name for tag in _DFINE_CLASS_NAMES):
        return True
    # Check submodules for D-FINE specific components
    for m in model.modules():
        if any(tag in type(m).__name__ for tag in ("DFINEDecoder", "HybridEncoder")):
            return True
    return False


def _has_deploy(model: nn.Module) -> bool:
    return hasattr(model, "deploy") and callable(model.deploy)


def _fix_project_tensor(model: nn.Module) -> bool:
    """Unsqueeze 1-D decoder.project to 2-D for CoreML linear op."""
    fixed = False
    for name, mod in model.named_modules():
        if hasattr(mod, "project") and isinstance(mod.project, (nn.Parameter, torch.Tensor)):
            if mod.project.dim() == 1:
                mod.project = nn.Parameter(mod.project.unsqueeze(0), requires_grad=False)
                logger.info("Fixed project tensor at %s: unsqueezed 1D → 2D", name)
                fixed = True
        # Also check nested decoder.decoder.project
        if hasattr(mod, "decoder") and hasattr(mod.decoder, "project"):
            proj = mod.decoder.project
            if isinstance(proj, (nn.Parameter, torch.Tensor)) and proj.dim() == 1:
                mod.decoder.project = nn.Parameter(proj.unsqueeze(0), requires_grad=False)
                logger.info("Fixed decoder.project: unsqueezed 1D → 2D")
                fixed = True
    return fixed


def _detect_use_focal_loss(model: nn.Module) -> bool:
    """Check if the D-FINE model uses focal loss (sigmoid) or CE (softmax)."""
    for m in model.modules():
        if hasattr(m, "use_focal_loss"):
            return bool(m.use_focal_loss)
    # Default to focal loss (most D-FINE COCO models use it)
    return True


class _DFINECoreMLWrapper(nn.Module):
    """Wraps D-FINE to output flat (confidence, coordinates) tensors."""

    def __init__(self, model: nn.Module, use_focal_loss: bool):
        super().__init__()
        self.model = model
        self.use_focal_loss = use_focal_loss

    def forward(self, images: torch.Tensor):
        outputs = self.model(images)
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        if self.use_focal_loss:
            confidence = F.sigmoid(logits)
        else:
            confidence = F.softmax(logits, dim=-1)[:, :, :-1]
        return confidence.squeeze(0), boxes.squeeze(0)


class DFINEExportRecipe(Recipe):
    name = "dfine_export"

    def match(self, model: nn.Module, architecture: str, error: str | None = None) -> bool:
        # Error-triggered: dict output or project tensor issues
        if error:
            err_lower = error.lower()
            if "pred_logits" in err_lower or "pred_boxes" in err_lower:
                return True
            if "project" in err_lower and "linear" in err_lower:
                return True

        # Pre-emptive: detect D-FINE architecture
        if _is_dfine(model):
            return True

        arch_lower = architecture.lower()
        return "dfine" in arch_lower or "d-fine" in arch_lower or "d_fine" in arch_lower

    def apply(self, model: nn.Module) -> RecipeResult:
        mods: list[str] = []

        # 1. Deploy mode
        if _has_deploy(model):
            model.deploy()
            mods.append("Called model.deploy() to switch to inference graph")

        # 2. Fix project tensor
        if _fix_project_tensor(model):
            mods.append("Unsqueezed 1D decoder.project to 2D for CoreML linear op")

        # 3. Detect focal loss setting
        use_focal_loss = _detect_use_focal_loss(model)
        activation = "sigmoid (focal loss)" if use_focal_loss else "softmax (CE loss)"
        mods.append(f"Output activation: {activation}")

        # Note: The actual output wrapping (_DFINECoreMLWrapper) cannot be
        # applied here because the recipe modifies the model in-place and
        # the converter does `torch.jit.trace(model, ...)` on the modified model.
        # The dict output issue will be caught by the detection_unwrap recipe
        # or needs the converter to handle dict-output models.
        # For now, we prepare the model so that if the user loads via peaceofcake
        # (which applies its own wrapper), the conversion works.

        return RecipeResult(
            applied=len(mods) > 0,
            recipe_name=self.name,
            description=f"Prepared D-FINE for CoreML export ({len(mods)} modifications)",
            modifications=mods,
        )
