"""Recipe: Patch DINOv2 backbone and deformable attention for CoreML.

RF-DETR (and other DINOv2-based detectors) fail CoreML conversion because:

1. **torch_int** — DINOv2 casts shape values to int64 tensors during
   ``torch.jit.trace``, producing ``int`` ops that coremltools cannot fold.
2. **Channel check** — ``Dinov2WithRegistersPatchEmbeddings.forward`` compares
   ``pixel_values.shape[1]`` against ``self.num_channels``, creating a traced
   ``int`` op.
3. **Rank-6 tensors** — Deformable attention reshapes sampling offsets to
   ``(N, Len_q, n_heads, n_levels, n_points, 2)`` which exceeds CoreML's
   rank ≤ 5 limitation.
4. **Dynamic splits** — ``value.split([H*W for H,W in spatial_shapes])``
   produces non-constant split sizes that CoreML rejects.

This recipe monkey-patches the relevant modules at the Python level so that
``torch.jit.trace`` produces a clean graph that coremltools can convert.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from model2mobile.convert.recipes.base import Recipe, RecipeResult

logger = logging.getLogger(__name__)

_DINOV2_MODULES = (
    "Dinov2WithRegisters",
    "WindowedDinov2",
    "DINOv2",
)

_DEFORM_ATTN_MODULES = (
    "MSDeformAttn",
    "MultiScaleDeformableAttention",
)


def _has_dinov2(model: nn.Module) -> bool:
    for m in model.modules():
        if any(tag in type(m).__name__ for tag in _DINOV2_MODULES):
            return True
    return False


def _has_deform_attn(model: nn.Module) -> bool:
    for m in model.modules():
        if any(tag in type(m).__name__ for tag in _DEFORM_ATTN_MODULES):
            return True
    return False


def _patch_torch_int() -> bool:
    """Force ``torch_int`` to always return Python int (safe for fixed-res export)."""
    patched = False
    try:
        import transformers.utils as tu
        tu.torch_int = lambda x: int(x)
        patched = True
    except ImportError:
        pass
    try:
        import rfdetr.models.backbone.dinov2_with_windowed_attn as dwv
        dwv.torch_int = lambda x: int(x)
        patched = True
    except ImportError:
        pass
    return patched


def _patch_dinov2_embeddings() -> bool:
    """Remove the channel-count assertion that creates an ``int`` op in trace."""
    try:
        import rfdetr.models.backbone.dinov2_with_windowed_attn as dwv
        dwv.Dinov2WithRegistersPatchEmbeddings.forward = (
            lambda self, pv: self.projection(pv).flatten(2).transpose(1, 2)
        )
        return True
    except (ImportError, AttributeError):
        return False


def _patch_deformable_attention() -> bool:
    """Rewrite MSDeformAttn.forward to use rank-5 tensors and static slicing.

    The original forward reshapes sampling_offsets to rank 6 and uses dynamic
    ``tensor.split()`` — both unsupported by CoreML.  This replacement merges
    ``n_levels`` and ``n_points`` into one dimension (rank 5) and slices the
    value tensor with static indices instead of calling ``split()``.
    """
    try:
        import rfdetr.models.ops.modules.ms_deform_attn as mda
        from rfdetr.utilities.tensors import _bilinear_grid_sample
    except ImportError:
        return False

    def _patched_forward(
        self, query, reference_points, input_flatten, input_spatial_shapes,
        input_level_start_index, input_padding_mask=None,
    ):
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[..., None], float(0))

        nh, nl, np_ = self.n_heads, self.n_levels, self.n_points
        hd = self.d_model // nh

        # Merge n_levels and n_points → rank 5
        offsets = self.sampling_offsets(query).view(N, Len_q, nh, nl * np_, 2)
        attn_w = self.attention_weights(query).view(N, Len_q, nh, nl * np_)

        if reference_points.shape[-1] == 2:
            norm = torch.stack(
                [input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1
            )
            norm = norm.repeat_interleave(np_, dim=0)
            ref = reference_points.repeat_interleave(np_, dim=2)
            sloc = ref[:, :, None, :, :] + offsets / norm[None, None, None, :, :]
        elif reference_points.shape[-1] == 4:
            rxy = reference_points[:, :, :, :2].repeat_interleave(np_, dim=2)
            rwh = reference_points[:, :, :, 2:].repeat_interleave(np_, dim=2)
            sloc = rxy[:, :, None, :, :] + offsets / np_ * rwh[:, :, None, :, :] * 0.5
        else:
            raise ValueError(
                f"reference_points last dim must be 2 or 4, got {reference_points.shape[-1]}"
            )

        attn_w = F.softmax(attn_w, -1)
        value = value.transpose(1, 2).contiguous().view(N, nh, hd, Len_in)

        sg = 2 * sloc - 1
        svl = []
        offset = 0
        for lid_ in range(nl):
            H = int(input_spatial_shapes[lid_, 0].item())
            W = int(input_spatial_shapes[lid_, 1].item())
            vl = value[:, :, :, offset:offset + H * W].reshape(N * nh, hd, H, W)
            offset += H * W
            gl = sg[:, :, :, lid_ * np_:(lid_ + 1) * np_, :]
            gl = gl.permute(0, 2, 1, 3, 4).reshape(N * nh, Len_q, np_, 2)
            svl.append(
                _bilinear_grid_sample(vl, gl, padding_mode="zeros", align_corners=False)
            )

        attn_w = attn_w.permute(0, 2, 1, 3).reshape(N * nh, 1, Len_q, nl * np_)
        out = (torch.stack(svl, dim=-2).flatten(-2) * attn_w).sum(-1)
        out = out.reshape(N, nh * hd, Len_q).permute(0, 2, 1).contiguous()
        return self.output_proj(out)

    mda.MSDeformAttn.forward = _patched_forward
    return True


class DINOv2DeformableAttnRecipe(Recipe):
    name = "dinov2_deformable_attn"

    def match(self, model: nn.Module, architecture: str, error: str | None = None) -> bool:
        # Match on error messages produced by the known failure chain
        if error:
            err_lower = error.lower()
            if "only supports tensors with rank <= 5" in err_lower:
                return True
            if "split_sizes" in err_lower and "must be const" in err_lower:
                return True
            if "0-dimensional arrays" in err_lower and "int" in err_lower:
                return True

        # Match on model architecture
        if _has_dinov2(model) or _has_deform_attn(model):
            return True

        arch_lower = architecture.lower()
        return any(tag in arch_lower for tag in ("rfdetr", "rf-detr", "rf_detr", "dinov2"))

    def apply(self, model: nn.Module) -> RecipeResult:
        mods: list[str] = []

        if _patch_torch_int():
            mods.append("torch_int -> Python int (fixed-resolution tracing)")

        if _patch_dinov2_embeddings():
            mods.append("Removed DINOv2 patch-embedding channel check")

        if _patch_deformable_attention():
            mods.append("Rewrote MSDeformAttn to rank-5 with static slicing")

        return RecipeResult(
            applied=len(mods) > 0,
            recipe_name=self.name,
            description=(
                f"Patched DINOv2 + deformable attention for CoreML ({len(mods)} fixes)"
            ),
            modifications=mods,
        )
