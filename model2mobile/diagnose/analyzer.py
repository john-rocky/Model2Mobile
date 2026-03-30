"""Rule-based failure diagnosis for Core ML conversion and deployment."""

from __future__ import annotations

import re
from dataclasses import field

from model2mobile.models import (
    BenchmarkResult,
    ConversionResult,
    Diagnosis,
    DiagnosisCategory,
    DiagnosisResult,
    ValidationResult,
    ValidationStatus,
)

# Each rule: (compiled regex pattern, DiagnosisCategory, likely cause description)
_ERROR_RULES: list[tuple[re.Pattern[str], DiagnosisCategory, str]] = [
    # Unsupported ops
    (
        re.compile(r"not supported|unsupported op|no conversion function", re.IGNORECASE),
        DiagnosisCategory.UNSUPPORTED_OP,
        "The model contains an operator that coremltools cannot convert.",
    ),
    (
        re.compile(r"could not fold|failed to fold", re.IGNORECASE),
        DiagnosisCategory.UNSUPPORTED_OP,
        "A constant-folding pass failed, likely due to an unsupported op.",
    ),
    # Dynamic shapes
    (
        re.compile(r"dynamic|variable.*(shape|length|size)|unknown rank", re.IGNORECASE),
        DiagnosisCategory.DYNAMIC_SHAPE,
        "The model uses dynamic shapes which Core ML does not fully support.",
    ),
    # Shape mismatch
    (
        re.compile(r"shape mismatch|dimension.*mismatch|incompatible.*shape", re.IGNORECASE),
        DiagnosisCategory.OUTPUT_SHAPE_MISMATCH,
        "Output tensor shapes do not match expectations after conversion.",
    ),
    (
        re.compile(r"size.*expected.*got|expected shape|shape.*differ", re.IGNORECASE),
        DiagnosisCategory.OUTPUT_SHAPE_MISMATCH,
        "Tensor shape diverged between PyTorch and Core ML.",
    ),
    # Numeric instability
    (
        re.compile(r"\bnan\b|\binf\b|numeric.*instability|overflow|underflow", re.IGNORECASE),
        DiagnosisCategory.NUMERIC_INSTABILITY,
        "Numerical issues detected (NaN/Inf), possibly from float16 precision.",
    ),
    # Memory issues
    (
        re.compile(r"out of memory|memory.*error|alloc.*fail|oom\b", re.IGNORECASE),
        DiagnosisCategory.MEMORY_ISSUE,
        "The conversion or model execution exceeded available memory.",
    ),
    (
        re.compile(r"killed|segfault|signal 9", re.IGNORECASE),
        DiagnosisCategory.MEMORY_ISSUE,
        "Process was killed, likely due to excessive memory usage.",
    ),
    # Non-traceable output (detection models returning List[Dict])
    (
        re.compile(r"Only tensors.*can be output|cannot be understood by the tracer", re.IGNORECASE),
        DiagnosisCategory.UNSUPPORTED_OP,
        "Model returns non-tensor output (e.g. List[Dict]). The detection_unwrap recipe handles this.",
    ),
    # Model loading issues
    (
        re.compile(r"state_dict|not a runnable model|Unexpected.*type", re.IGNORECASE),
        DiagnosisCategory.UNSUPPORTED_OP,
        "The file contains weights only, not a runnable model. Save with torch.save(model, path).",
    ),
    # Runtime failure (broad catch for execution errors)
    (
        re.compile(r"runtime error|execution.*fail|predict.*fail|inference.*error", re.IGNORECASE),
        DiagnosisCategory.RUNTIME_FAILURE,
        "The Core ML model failed at runtime during prediction.",
    ),
]

_SUGGESTED_STEPS: dict[DiagnosisCategory, list[str]] = {
    DiagnosisCategory.UNSUPPORTED_OP: [
        "Identify the unsupported op in the error message.",
        "Replace the op with an equivalent supported op or custom implementation.",
        "Consider simplifying the model head or using torch.onnx.export as intermediate.",
        "Update coremltools to the latest version.",
    ],
    DiagnosisCategory.DYNAMIC_SHAPE: [
        "Use a fixed input shape (e.g., ct.TensorType(shape=(1,3,640,640))).",
        "Replace data-dependent control flow with static alternatives.",
        "Avoid using torch.Tensor.size() inside forward() for branching.",
    ],
    DiagnosisCategory.OUTPUT_SHAPE_MISMATCH: [
        "Verify the expected output shape from the PyTorch model.",
        "Simplify the output head to return a fixed-shape tensor.",
        "Move NMS or dynamic postprocessing outside the model.",
    ],
    DiagnosisCategory.RUNTIME_FAILURE: [
        "Check the model for data-dependent operations.",
        "Try converting with a different compute unit (CPU_ONLY).",
        "Verify that the input shape matches the model expectation.",
    ],
    DiagnosisCategory.NUMERIC_INSTABILITY: [
        "Convert using float32 precision instead of float16.",
        "Check for large activation values or very small denominators.",
        "Add numerical clamps (e.g., torch.clamp) in sensitive areas.",
    ],
    DiagnosisCategory.POSTPROCESS_BOTTLENECK: [
        "Move NMS outside the model and run it natively on-device.",
        "Reduce the number of candidate boxes before postprocessing.",
        "Pre-filter low-confidence detections before NMS.",
    ],
    DiagnosisCategory.MEMORY_ISSUE: [
        "Reduce input resolution (e.g., 640 -> 320).",
        "Use a smaller model variant (e.g., nano / small).",
        "Close other applications to free memory.",
        "Try CPU_ONLY compute unit to reduce GPU memory pressure.",
    ],
    DiagnosisCategory.UNKNOWN: [
        "Review the raw error message for clues.",
        "Try a simpler model to verify the conversion pipeline works.",
        "Check coremltools and PyTorch version compatibility.",
    ],
}


def _match_error(error_text: str) -> list[Diagnosis]:
    diagnoses: list[Diagnosis] = []
    seen_categories: set[DiagnosisCategory] = set()

    for pattern, category, likely_cause in _ERROR_RULES:
        if pattern.search(error_text) and category not in seen_categories:
            seen_categories.add(category)
            diagnoses.append(
                Diagnosis(
                    category=category,
                    raw_error=error_text,
                    likely_cause=likely_cause,
                    suggested_steps=list(_SUGGESTED_STEPS.get(category, [])),
                )
            )

    return diagnoses


def _check_benchmark(benchmark: BenchmarkResult) -> list[Diagnosis]:
    diagnoses: list[Diagnosis] = []

    # Postprocess bottleneck: postprocess latency exceeds inference latency
    if (
        benchmark.postprocess.mean_ms > 0
        and benchmark.inference.mean_ms > 0
        and benchmark.postprocess.mean_ms > benchmark.inference.mean_ms
    ):
        diagnoses.append(
            Diagnosis(
                category=DiagnosisCategory.POSTPROCESS_BOTTLENECK,
                raw_error=(
                    f"Postprocess latency ({benchmark.postprocess.mean_ms:.1f}ms) "
                    f"exceeds inference latency ({benchmark.inference.mean_ms:.1f}ms)"
                ),
                likely_cause="Postprocessing is the primary bottleneck, not model inference.",
                suggested_steps=list(
                    _SUGGESTED_STEPS.get(DiagnosisCategory.POSTPROCESS_BOTTLENECK, [])
                ),
            )
        )

    # Memory issue from benchmark
    if benchmark.peak_memory_mb is not None and benchmark.peak_memory_mb > 1500:
        diagnoses.append(
            Diagnosis(
                category=DiagnosisCategory.MEMORY_ISSUE,
                raw_error=f"Peak memory usage: {benchmark.peak_memory_mb:.0f} MB",
                likely_cause="The model consumes excessive memory at runtime.",
                suggested_steps=list(
                    _SUGGESTED_STEPS.get(DiagnosisCategory.MEMORY_ISSUE, [])
                ),
            )
        )

    # Benchmark-level error
    if not benchmark.success and benchmark.error_message:
        diagnoses.extend(_match_error(benchmark.error_message))

    return diagnoses


def _check_validation(validation: ValidationResult) -> list[Diagnosis]:
    diagnoses: list[Diagnosis] = []

    if validation.status == ValidationStatus.FAIL:
        # Look for numeric issues in check details
        for check in validation.checks:
            if check.status == ValidationStatus.FAIL:
                detail_lower = check.detail.lower()
                if "nan" in detail_lower or "inf" in detail_lower:
                    diagnoses.append(
                        Diagnosis(
                            category=DiagnosisCategory.NUMERIC_INSTABILITY,
                            raw_error=check.detail,
                            likely_cause="Validation detected NaN or Inf in model outputs.",
                            suggested_steps=list(
                                _SUGGESTED_STEPS.get(DiagnosisCategory.NUMERIC_INSTABILITY, [])
                            ),
                        )
                    )
                elif "shape" in detail_lower or "dimension" in detail_lower:
                    diagnoses.append(
                        Diagnosis(
                            category=DiagnosisCategory.OUTPUT_SHAPE_MISMATCH,
                            raw_error=check.detail,
                            likely_cause="Validation found mismatched output shapes.",
                            suggested_steps=list(
                                _SUGGESTED_STEPS.get(DiagnosisCategory.OUTPUT_SHAPE_MISMATCH, [])
                            ),
                        )
                    )

    if validation.error_message:
        diagnoses.extend(_match_error(validation.error_message))

    return diagnoses


def _determine_primary(diagnoses: list[Diagnosis]) -> DiagnosisCategory:
    if not diagnoses:
        return DiagnosisCategory.UNKNOWN

    # Priority order for primary category
    priority = [
        DiagnosisCategory.UNSUPPORTED_OP,
        DiagnosisCategory.DYNAMIC_SHAPE,
        DiagnosisCategory.OUTPUT_SHAPE_MISMATCH,
        DiagnosisCategory.NUMERIC_INSTABILITY,
        DiagnosisCategory.MEMORY_ISSUE,
        DiagnosisCategory.RUNTIME_FAILURE,
        DiagnosisCategory.POSTPROCESS_BOTTLENECK,
        DiagnosisCategory.UNKNOWN,
    ]
    categories = {d.category for d in diagnoses}
    for cat in priority:
        if cat in categories:
            return cat
    return diagnoses[0].category


def diagnose(
    conversion: ConversionResult,
    benchmark: BenchmarkResult | None = None,
    validation: ValidationResult | None = None,
) -> DiagnosisResult:
    all_diagnoses: list[Diagnosis] = []

    # Analyze conversion errors
    if not conversion.success:
        error_text = conversion.raw_error or conversion.error_message or ""
        matched = _match_error(error_text)
        if matched:
            all_diagnoses.extend(matched)
        else:
            # No pattern matched -> unknown
            all_diagnoses.append(
                Diagnosis(
                    category=DiagnosisCategory.UNKNOWN,
                    raw_error=error_text,
                    likely_cause="The error did not match any known pattern.",
                    suggested_steps=list(_SUGGESTED_STEPS[DiagnosisCategory.UNKNOWN]),
                )
            )

    # Analyze conversion warnings
    for warning in conversion.warnings:
        all_diagnoses.extend(_match_error(warning))

    # Analyze benchmark results
    if benchmark is not None:
        all_diagnoses.extend(_check_benchmark(benchmark))

    # Analyze validation results
    if validation is not None:
        all_diagnoses.extend(_check_validation(validation))

    primary = _determine_primary(all_diagnoses)

    return DiagnosisResult(
        diagnoses=all_diagnoses,
        primary_category=primary,
    )
