"""Validation module."""

from model2mobile.validate.validator import run_validation
from model2mobile.validate.task_classify import validate_classification
from model2mobile.validate.task_depth import validate_depth
from model2mobile.validate.task_detect import validate_detection
from model2mobile.validate.task_segment import validate_segmentation

__all__ = [
    "run_validation",
    "validate_classification",
    "validate_depth",
    "validate_detection",
    "validate_segmentation",
]
