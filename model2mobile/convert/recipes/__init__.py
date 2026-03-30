"""Conversion recipes — model-specific fixes applied before Core ML conversion.

Each recipe knows how to detect a specific model family and apply the
op-level modifications needed to make the conversion succeed.

To add a new recipe:
  1. Create a new file in this directory (e.g. my_model.py)
  2. Subclass `Recipe` and implement `match()` and `apply()`
  3. Register it in `_ALL_RECIPES` below

The recipe system is tried automatically by the converter.  When a
conversion fails, the converter checks if any recipe matches and
retries with the patched model.
"""

from model2mobile.convert.recipes.base import Recipe, RecipeResult
from model2mobile.convert.recipes.registry import get_recipes, match_recipe

__all__ = ["Recipe", "RecipeResult", "get_recipes", "match_recipe"]
