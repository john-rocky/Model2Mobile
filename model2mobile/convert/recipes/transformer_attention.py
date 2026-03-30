"""Recipe: Replace scaled_dot_product_attention with basic ops.

Models that call ``F.scaled_dot_product_attention`` can fail on older
coremltools versions that do not map the op.  This recipe monkey-patches
``torch.nn.functional`` so the attention is computed with elementary
matmul / softmax operations that coremltools can always convert.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model2mobile.convert.recipes.base import Recipe, RecipeResult

logger = logging.getLogger(__name__)


def _manual_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Drop-in replacement using Q*K^T / sqrt(d) + softmax + V."""
    d_k = query.size(-1)
    scale_factor = scale if scale is not None else 1.0 / math.sqrt(d_k)

    attn_weight = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    if is_causal:
        seq_len_q = query.size(-2)
        seq_len_k = key.size(-2)
        causal_mask = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device).tril()
        attn_weight = attn_weight.masked_fill(~causal_mask, float("-inf"))

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weight = attn_weight.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_weight = attn_weight + attn_mask

    attn_weight = torch.softmax(attn_weight, dim=-1)

    if dropout_p > 0.0 and query.requires_grad:
        attn_weight = F.dropout(attn_weight, p=dropout_p)

    return torch.matmul(attn_weight, value)


def _has_sdpa_modules(model: nn.Module) -> bool:
    """Check if the model contains MultiheadAttention or similar attention modules."""
    for m in model.modules():
        cls_name = type(m).__name__
        if "MultiheadAttention" in cls_name or "Attention" in cls_name:
            return True
    return False


class TransformerAttentionRecipe(Recipe):
    name = "transformer_attention"

    _original_sdpa = None

    def match(self, model: nn.Module, architecture: str, error: str | None = None) -> bool:
        if error and "scaled_dot_product" in error.lower():
            return True
        return _has_sdpa_modules(model)

    def apply(self, model: nn.Module) -> RecipeResult:
        # Monkey-patch torch.nn.functional
        if hasattr(F, "scaled_dot_product_attention"):
            TransformerAttentionRecipe._original_sdpa = F.scaled_dot_product_attention
            F.scaled_dot_product_attention = _manual_scaled_dot_product_attention  # type: ignore[assignment]
            logger.info("Replaced F.scaled_dot_product_attention with manual implementation")

            return RecipeResult(
                applied=True,
                recipe_name=self.name,
                description="Replaced scaled_dot_product_attention with manual Q*K^T/sqrt(d) implementation",
                modifications=["F.scaled_dot_product_attention -> manual matmul+softmax"],
            )

        return RecipeResult(
            applied=False,
            recipe_name=self.name,
            description="F.scaled_dot_product_attention not found; nothing to patch",
        )
