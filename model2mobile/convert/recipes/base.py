"""Base class for conversion recipes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class RecipeResult:
    """What the recipe did to the model."""

    applied: bool
    recipe_name: str = ""
    description: str = ""
    modifications: list[str] = field(default_factory=list)


class Recipe(ABC):
    """A conversion recipe that patches a model before Core ML conversion.

    Subclass this and implement `match()` and `apply()`.
    """

    # Human-readable name shown in reports
    name: str = "unnamed"

    @abstractmethod
    def match(self, model: nn.Module, architecture: str, error: str | None = None) -> bool:
        """Return True if this recipe applies to the given model.

        Args:
            model: The loaded PyTorch model.
            architecture: Detected architecture name (class name or TorchScript original_name).
            error: If this is a retry after a failed conversion, the error message.
                   None on the first (pre-emptive) pass.
        """
        ...

    @abstractmethod
    def apply(self, model: nn.Module) -> RecipeResult:
        """Modify the model in-place and return what was changed.

        The model is already in eval mode.  Modifications typically include:
        - Replacing unsupported ops with supported alternatives
        - Removing NMS or other dynamic postprocessing
        - Fixing output shapes
        - Wrapping layers with coreml-friendly versions
        """
        ...
