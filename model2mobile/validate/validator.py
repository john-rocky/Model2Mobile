"""Validation module: compare PyTorch and Core ML model outputs."""

from __future__ import annotations

import traceback
from typing import Any

import coremltools as ct
import numpy as np
import torch
from PIL import Image

from model2mobile.config import RunConfig
from model2mobile.models import ValidationCheck, ValidationResult, ValidationStatus


def _load_pytorch_model(path: str) -> torch.nn.Module:
    """Load a PyTorch model from a .pt / .pth / TorchScript file."""
    try:
        model = torch.jit.load(path, map_location="cpu")
    except Exception:
        # Fall back to loading as a state dict wrapper -- caller likely
        # saved the full model via torch.save(model, path).
        model = torch.load(path, map_location="cpu", weights_only=False)
    model.eval()
    return model


def _create_test_input(input_size: int) -> torch.Tensor:
    """Create a deterministic test tensor (NCHW, float32, [0..1])."""
    gen = torch.Generator()
    gen.manual_seed(12345)
    return torch.rand(1, 3, input_size, input_size, generator=gen)


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a NCHW [0..1] tensor to a PIL Image."""
    arr = (tensor.squeeze(0).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def _prepare_coreml_input(
    model: ct.models.MLModel, image: Image.Image, tensor: torch.Tensor
) -> dict[str, Any]:
    """Build the input dict for the Core ML model."""
    spec = model.get_spec()
    input_dict: dict[str, Any] = {}

    for inp in spec.description.input:
        name = inp.name
        input_type = inp.type.WhichOneof("Type")

        if input_type == "imageType":
            input_dict[name] = image
        elif input_type == "multiArrayType":
            # Use the same tensor data, converted to numpy
            input_dict[name] = tensor.numpy()
        else:
            input_dict[name] = tensor.numpy()

    return input_dict


def _flatten_outputs(raw: Any) -> dict[str, np.ndarray]:
    """Normalize model outputs to a dict of numpy arrays."""
    if isinstance(raw, dict):
        return {k: np.asarray(v) for k, v in raw.items()}
    if isinstance(raw, (list, tuple)):
        return {f"output_{i}": np.asarray(v) for i, v in enumerate(raw)}
    return {"output_0": np.asarray(raw)}


def _torch_outputs(model: torch.nn.Module, tensor: torch.Tensor) -> dict[str, np.ndarray]:
    """Run PyTorch inference and return flattened numpy outputs."""
    with torch.no_grad():
        raw = model(tensor)
    # Handle common YOLO-style returns (tuple of tensors, list of dicts, etc.)
    if isinstance(raw, torch.Tensor):
        return {"output_0": raw.cpu().numpy()}
    if isinstance(raw, (tuple, list)):
        out: dict[str, np.ndarray] = {}
        for i, item in enumerate(raw):
            if isinstance(item, torch.Tensor):
                out[f"output_{i}"] = item.cpu().numpy()
            elif isinstance(item, dict):
                for k, v in item.items():
                    arr = v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
                    out[f"output_{i}_{k}"] = arr
            else:
                out[f"output_{i}"] = np.asarray(item)
        return out
    if isinstance(raw, dict):
        return {k: (v.cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v))
                for k, v in raw.items()}
    return {"output_0": np.asarray(raw)}


def _coreml_outputs(model: ct.models.MLModel, input_dict: dict[str, Any]) -> dict[str, np.ndarray]:
    """Run Core ML inference and return flattened numpy outputs."""
    prediction = model.predict(input_dict)
    return _flatten_outputs(prediction)


# ---------------------------------------------------------------------------
# Individual check helpers
# ---------------------------------------------------------------------------

def _check_output_presence(
    pt_outs: dict[str, np.ndarray],
    cm_outs: dict[str, np.ndarray],
) -> ValidationCheck:
    if not pt_outs and not cm_outs:
        return ValidationCheck(
            name="output_presence",
            status=ValidationStatus.FAIL,
            detail="Both models produced no output.",
        )
    if not pt_outs or not cm_outs:
        missing = "PyTorch" if not pt_outs else "CoreML"
        return ValidationCheck(
            name="output_presence",
            status=ValidationStatus.FAIL,
            detail=f"{missing} model produced no output.",
        )
    return ValidationCheck(
        name="output_presence",
        status=ValidationStatus.PASS,
        detail=f"PyTorch outputs: {len(pt_outs)}, CoreML outputs: {len(cm_outs)}.",
    )


def _check_output_shape(
    pt_outs: dict[str, np.ndarray],
    cm_outs: dict[str, np.ndarray],
) -> ValidationCheck:
    pt_shapes = {k: v.shape for k, v in pt_outs.items()}
    cm_shapes = {k: v.shape for k, v in cm_outs.items()}

    # If both have the same keys and shapes, PASS
    if set(pt_shapes.keys()) == set(cm_shapes.keys()):
        mismatches = {k: (pt_shapes[k], cm_shapes[k])
                      for k in pt_shapes if pt_shapes[k] != cm_shapes[k]}
        if not mismatches:
            return ValidationCheck(
                name="output_shape",
                status=ValidationStatus.PASS,
                detail="All output shapes match.",
                expected=str(pt_shapes),
                actual=str(cm_shapes),
            )
        return ValidationCheck(
            name="output_shape",
            status=ValidationStatus.WARNING,
            detail=f"Shape mismatches in: {list(mismatches.keys())}.",
            expected=str(pt_shapes),
            actual=str(cm_shapes),
        )

    # Different keys -- try element-count comparison
    pt_total = sum(v.size for v in pt_outs.values())
    cm_total = sum(v.size for v in cm_outs.values())
    if pt_total == cm_total:
        return ValidationCheck(
            name="output_shape",
            status=ValidationStatus.WARNING,
            detail="Output names differ but total element counts match.",
            expected=str(pt_shapes),
            actual=str(cm_shapes),
        )
    return ValidationCheck(
        name="output_shape",
        status=ValidationStatus.WARNING,
        detail="Output names and shapes differ between models.",
        expected=str(pt_shapes),
        actual=str(cm_shapes),
    )


def _try_pair_outputs(
    pt_outs: dict[str, np.ndarray],
    cm_outs: dict[str, np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Best-effort pairing of PyTorch and CoreML outputs by shape similarity."""
    pairs: list[tuple[np.ndarray, np.ndarray]] = []

    # First try matching by key
    matched_cm: set[str] = set()
    for pk, pv in pt_outs.items():
        if pk in cm_outs:
            pairs.append((pv, cm_outs[pk]))
            matched_cm.add(pk)

    if pairs:
        return pairs

    # Fall back to shape-based matching
    cm_remaining = [(k, v) for k, v in cm_outs.items() if k not in matched_cm]
    for _, pv in pt_outs.items():
        for j, (ck, cv) in enumerate(cm_remaining):
            if pv.shape == cv.shape:
                pairs.append((pv, cv))
                cm_remaining.pop(j)
                break

    if pairs:
        return pairs

    # Last resort: flatten and compare by size
    pt_sorted = sorted(pt_outs.values(), key=lambda a: a.size, reverse=True)
    cm_sorted = sorted(cm_outs.values(), key=lambda a: a.size, reverse=True)
    for pv, cv in zip(pt_sorted, cm_sorted):
        pairs.append((pv, cv))

    return pairs


def _check_detection_count(
    pt_outs: dict[str, np.ndarray],
    cm_outs: dict[str, np.ndarray],
    tolerance_ratio: float = 0.3,
) -> ValidationCheck:
    """Compare detection counts (heuristic: look for variable-length dims)."""
    # Heuristic: detection count is the largest non-batch dim product
    def _det_count(outputs: dict[str, np.ndarray]) -> int:
        total = 0
        for v in outputs.values():
            if v.ndim >= 2:
                total += int(np.prod(v.shape[1:]))
            else:
                total += v.size
        return total

    pt_count = _det_count(pt_outs)
    cm_count = _det_count(cm_outs)

    if pt_count == 0 and cm_count == 0:
        return ValidationCheck(
            name="detection_count",
            status=ValidationStatus.PASS,
            detail="Both models produced zero detections (or no detection-like output).",
            expected=pt_count,
            actual=cm_count,
        )

    denom = max(pt_count, cm_count, 1)
    ratio = abs(pt_count - cm_count) / denom

    if ratio <= tolerance_ratio:
        return ValidationCheck(
            name="detection_count",
            status=ValidationStatus.PASS,
            detail=f"Detection element counts within {tolerance_ratio*100:.0f}% tolerance.",
            expected=pt_count,
            actual=cm_count,
        )
    return ValidationCheck(
        name="detection_count",
        status=ValidationStatus.WARNING,
        detail=f"Detection element counts differ by {ratio*100:.1f}%.",
        expected=pt_count,
        actual=cm_count,
    )


def _check_confidence_consistency(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    tolerance: float,
) -> ValidationCheck:
    """Check that confidence-like values (0-1 range) are consistent."""
    for pt_arr, cm_arr in pairs:
        # Only consider arrays whose values are in [0, 1]
        if pt_arr.size == 0 or cm_arr.size == 0:
            continue

        pt_flat = pt_arr.flatten().astype(np.float64)
        cm_flat = cm_arr.flatten().astype(np.float64)

        pt_in_range = np.all((pt_flat >= -0.01) & (pt_flat <= 1.01))
        cm_in_range = np.all((cm_flat >= -0.01) & (cm_flat <= 1.01))

        if not (pt_in_range and cm_in_range):
            continue

        # Compare element-wise if shapes match
        if pt_flat.shape == cm_flat.shape:
            max_diff = float(np.max(np.abs(pt_flat - cm_flat)))
            mean_diff = float(np.mean(np.abs(pt_flat - cm_flat)))

            if max_diff <= tolerance:
                return ValidationCheck(
                    name="confidence_consistency",
                    status=ValidationStatus.PASS,
                    detail=f"Max confidence diff: {max_diff:.6f} (mean: {mean_diff:.6f}).",
                    tolerance=tolerance,
                )
            if max_diff <= tolerance * 5:
                return ValidationCheck(
                    name="confidence_consistency",
                    status=ValidationStatus.WARNING,
                    detail=f"Max confidence diff: {max_diff:.6f} exceeds tolerance {tolerance}.",
                    tolerance=tolerance,
                )
            return ValidationCheck(
                name="confidence_consistency",
                status=ValidationStatus.FAIL,
                detail=f"Max confidence diff: {max_diff:.6f} far exceeds tolerance {tolerance}.",
                tolerance=tolerance,
            )

        # Shapes differ -- compare distributions
        if pt_flat.size > 0 and cm_flat.size > 0:
            pt_mean, cm_mean = float(np.mean(pt_flat)), float(np.mean(cm_flat))
            diff = abs(pt_mean - cm_mean)
            if diff <= tolerance:
                return ValidationCheck(
                    name="confidence_consistency",
                    status=ValidationStatus.PASS,
                    detail=f"Mean confidence diff: {diff:.6f} (shapes differ).",
                    tolerance=tolerance,
                )
            return ValidationCheck(
                name="confidence_consistency",
                status=ValidationStatus.WARNING,
                detail=f"Mean confidence diff: {diff:.6f}, shapes differ.",
                tolerance=tolerance,
            )

    # No confidence-like arrays found
    return ValidationCheck(
        name="confidence_consistency",
        status=ValidationStatus.WARNING,
        detail="No confidence-like output arrays found for comparison.",
        tolerance=tolerance,
    )


def _check_bbox_consistency(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    tolerance_px: float,
) -> ValidationCheck:
    """Check bounding-box consistency (arrays not confined to [0,1])."""
    for pt_arr, cm_arr in pairs:
        if pt_arr.size == 0 or cm_arr.size == 0:
            continue

        pt_flat = pt_arr.flatten().astype(np.float64)
        cm_flat = cm_arr.flatten().astype(np.float64)

        # Heuristic: bbox values are typically > 1 (pixel coordinates)
        has_large = np.any(np.abs(pt_flat) > 1.5) or np.any(np.abs(cm_flat) > 1.5)
        if not has_large:
            continue

        if pt_flat.shape == cm_flat.shape:
            max_diff = float(np.max(np.abs(pt_flat - cm_flat)))
            mean_diff = float(np.mean(np.abs(pt_flat - cm_flat)))

            if max_diff <= tolerance_px:
                return ValidationCheck(
                    name="bbox_consistency",
                    status=ValidationStatus.PASS,
                    detail=f"Max bbox diff: {max_diff:.2f}px (mean: {mean_diff:.2f}px).",
                    tolerance=tolerance_px,
                )
            if max_diff <= tolerance_px * 5:
                return ValidationCheck(
                    name="bbox_consistency",
                    status=ValidationStatus.WARNING,
                    detail=f"Max bbox diff: {max_diff:.2f}px exceeds {tolerance_px}px.",
                    tolerance=tolerance_px,
                )
            return ValidationCheck(
                name="bbox_consistency",
                status=ValidationStatus.FAIL,
                detail=f"Max bbox diff: {max_diff:.2f}px far exceeds {tolerance_px}px.",
                tolerance=tolerance_px,
            )

    return ValidationCheck(
        name="bbox_consistency",
        status=ValidationStatus.WARNING,
        detail="No bbox-like output arrays found for element-wise comparison.",
        tolerance=tolerance_px,
    )


def _check_class_consistency(
    pairs: list[tuple[np.ndarray, np.ndarray]],
) -> ValidationCheck:
    """Check class prediction consistency by comparing argmax indices."""
    for pt_arr, cm_arr in pairs:
        if pt_arr.ndim < 2 or cm_arr.ndim < 2:
            continue
        if pt_arr.shape != cm_arr.shape:
            continue

        # Assume last dim or second dim holds class logits/probabilities
        # Try argmax along the last axis
        pt_classes = np.argmax(pt_arr, axis=-1).flatten()
        cm_classes = np.argmax(cm_arr, axis=-1).flatten()

        if pt_classes.shape != cm_classes.shape:
            continue

        if pt_classes.size == 0:
            continue

        match_ratio = float(np.mean(pt_classes == cm_classes))

        if match_ratio >= 0.95:
            return ValidationCheck(
                name="class_consistency",
                status=ValidationStatus.PASS,
                detail=f"Class match ratio: {match_ratio*100:.1f}%.",
            )
        if match_ratio >= 0.8:
            return ValidationCheck(
                name="class_consistency",
                status=ValidationStatus.WARNING,
                detail=f"Class match ratio: {match_ratio*100:.1f}%.",
            )
        return ValidationCheck(
            name="class_consistency",
            status=ValidationStatus.FAIL,
            detail=f"Class match ratio: {match_ratio*100:.1f}%.",
        )

    return ValidationCheck(
        name="class_consistency",
        status=ValidationStatus.WARNING,
        detail="No suitable output arrays found for class comparison.",
    )


def _general_tensor_comparison(
    pairs: list[tuple[np.ndarray, np.ndarray]],
    atol: float = 1e-3,
) -> ValidationCheck:
    """Fallback: general element-wise numeric comparison."""
    for pt_arr, cm_arr in pairs:
        if pt_arr.shape != cm_arr.shape:
            continue
        pt_flat = pt_arr.flatten().astype(np.float64)
        cm_flat = cm_arr.flatten().astype(np.float64)
        max_diff = float(np.max(np.abs(pt_flat - cm_flat)))
        mean_diff = float(np.mean(np.abs(pt_flat - cm_flat)))
        cos_sim = 0.0
        norm_pt = np.linalg.norm(pt_flat)
        norm_cm = np.linalg.norm(cm_flat)
        if norm_pt > 0 and norm_cm > 0:
            cos_sim = float(np.dot(pt_flat, cm_flat) / (norm_pt * norm_cm))

        detail = (
            f"Max abs diff: {max_diff:.6f}, mean abs diff: {mean_diff:.6f}, "
            f"cosine similarity: {cos_sim:.6f}."
        )
        if max_diff <= atol and cos_sim >= 0.99:
            return ValidationCheck(name="general_tensor_match", status=ValidationStatus.PASS, detail=detail)
        if cos_sim >= 0.95:
            return ValidationCheck(name="general_tensor_match", status=ValidationStatus.WARNING, detail=detail)
        return ValidationCheck(name="general_tensor_match", status=ValidationStatus.FAIL, detail=detail)

    return ValidationCheck(
        name="general_tensor_match",
        status=ValidationStatus.WARNING,
        detail="Could not perform element-wise tensor comparison (shapes differ).",
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _determine_overall_status(checks: list[ValidationCheck]) -> ValidationStatus:
    """Determine overall validation status from individual checks."""
    if any(c.status == ValidationStatus.FAIL for c in checks):
        return ValidationStatus.FAIL
    if any(c.status == ValidationStatus.WARNING for c in checks):
        return ValidationStatus.WARNING
    return ValidationStatus.PASS


def run_validation(
    pytorch_model_path: str,
    coreml_path: str,
    config: RunConfig,
) -> ValidationResult:
    """Compare PyTorch and Core ML model outputs for consistency."""
    try:
        # Load models
        pt_model = _load_pytorch_model(pytorch_model_path)
        cm_model = ct.models.MLModel(coreml_path)

        # Create deterministic test input
        test_tensor = _create_test_input(config.input_size)
        test_image = _tensor_to_pil(test_tensor)

        # Run inference on both models
        pt_outs = _torch_outputs(pt_model, test_tensor)
        cm_input = _prepare_coreml_input(cm_model, test_image, test_tensor)
        cm_outs = _coreml_outputs(cm_model, cm_input)

        # Pair outputs for element-wise comparisons
        pairs = _try_pair_outputs(pt_outs, cm_outs)

        # Run all checks
        checks: list[ValidationCheck] = [
            _check_output_presence(pt_outs, cm_outs),
            _check_output_shape(pt_outs, cm_outs),
            _check_detection_count(pt_outs, cm_outs),
            _check_confidence_consistency(pairs, config.confidence_tolerance),
            _check_bbox_consistency(pairs, config.bbox_tolerance),
            _check_class_consistency(pairs),
        ]

        # Add a general tensor comparison as extra signal
        general_check = _general_tensor_comparison(pairs)
        checks.append(general_check)

        status = _determine_overall_status(checks)

        return ValidationResult(
            status=status,
            checks=checks,
        )

    except Exception as exc:
        return ValidationResult(
            status=ValidationStatus.FAIL,
            checks=[],
            error_message=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
