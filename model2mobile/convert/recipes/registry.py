"""Recipe registry — discovers and matches recipes to models."""

from __future__ import annotations

import logging

import torch.nn as nn

from model2mobile.convert.recipes.base import Recipe, RecipeResult

logger = logging.getLogger(__name__)


def _load_builtin_recipes() -> list[Recipe]:
    """Import and instantiate all built-in recipes."""
    recipes: list[Recipe] = []

    # Each recipe module exposes a top-level RECIPE instance or a list.
    # Import them here to keep the registry as the single discovery point.
    try:
        from model2mobile.convert.recipes.silu_replace import SiLUReplaceRecipe
        recipes.append(SiLUReplaceRecipe())
    except ImportError:
        pass

    try:
        from model2mobile.convert.recipes.nms_strip import NMSStripRecipe
        recipes.append(NMSStripRecipe())
    except ImportError:
        pass

    try:
        from model2mobile.convert.recipes.dynamic_to_static import DynamicToStaticRecipe
        recipes.append(DynamicToStaticRecipe())
    except ImportError:
        pass

    try:
        from model2mobile.convert.recipes.detection_unwrap import DetectionUnwrapRecipe
        recipes.append(DetectionUnwrapRecipe())
    except ImportError:
        pass

    return recipes


_recipes: list[Recipe] | None = None


def get_recipes() -> list[Recipe]:
    """Return all registered recipes (loaded lazily)."""
    global _recipes
    if _recipes is None:
        _recipes = _load_builtin_recipes()
    return _recipes


def match_recipe(
    model: nn.Module,
    architecture: str,
    error: str | None = None,
) -> Recipe | None:
    """Find the first recipe that matches the model (optionally after a failure)."""
    for recipe in get_recipes():
        try:
            if recipe.match(model, architecture, error):
                logger.info("Recipe matched: %s", recipe.name)
                return recipe
        except Exception:
            continue
    return None


def apply_recipes(
    model: nn.Module,
    architecture: str,
    error: str | None = None,
) -> list[RecipeResult]:
    """Apply ALL matching recipes to the model. Returns list of results."""
    results: list[RecipeResult] = []
    for recipe in get_recipes():
        try:
            if recipe.match(model, architecture, error):
                result = recipe.apply(model)
                if result.applied:
                    logger.info("Recipe applied: %s — %s", recipe.name, result.description)
                    results.append(result)
        except Exception as exc:
            logger.warning("Recipe %s failed: %s", recipe.name, exc)
    return results
