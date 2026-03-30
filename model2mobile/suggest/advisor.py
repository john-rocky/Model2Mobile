"""Suggestion system for improving model deployment readiness."""

from __future__ import annotations

from model2mobile.models import (
    BenchmarkResult,
    DiagnosisCategory,
    DiagnosisResult,
    Suggestion,
    ValidationResult,
    ValidationStatus,
)

# Mapping from diagnosis category to prioritized suggestions
_CATEGORY_SUGGESTIONS: dict[DiagnosisCategory, list[Suggestion]] = {
    DiagnosisCategory.UNSUPPORTED_OP: [
        Suggestion(
            title="Replace export-unfriendly block",
            description=(
                "Identify the unsupported operator in the error message and replace it "
                "with a CoreML-compatible alternative. Check the coremltools op support "
                "matrix for compatible ops."
            ),
            priority=1,
        ),
        Suggestion(
            title="Simplify the output head",
            description=(
                "Complex output heads (e.g., multi-branch detection heads) often contain "
                "unsupported ops. Simplify by returning raw tensors and handling "
                "postprocessing on the host side."
            ),
            priority=2,
        ),
        Suggestion(
            title="Update coremltools",
            description=(
                "Newer versions of coremltools add support for additional ops. "
                "Run: pip install --upgrade coremltools"
            ),
            priority=3,
        ),
    ],
    DiagnosisCategory.DYNAMIC_SHAPE: [
        Suggestion(
            title="Use fixed input shape",
            description=(
                "Core ML works best with static shapes. Specify a fixed input shape "
                "(e.g., (1, 3, 640, 640)) during conversion and avoid data-dependent "
                "tensor sizes inside the model."
            ),
            priority=1,
        ),
        Suggestion(
            title="Remove data-dependent control flow",
            description=(
                "Replace if/else branches that depend on tensor values or shapes with "
                "static alternatives. Avoid calling .size() for branching inside forward()."
            ),
            priority=2,
        ),
    ],
    DiagnosisCategory.OUTPUT_SHAPE_MISMATCH: [
        Suggestion(
            title="Simplify output head",
            description=(
                "Return fixed-shape tensors from the model. Variable-length outputs "
                "(e.g., filtered detections) should be handled outside the model."
            ),
            priority=1,
        ),
        Suggestion(
            title="Move NMS outside the model",
            description=(
                "Non-Maximum Suppression produces variable-length output. Export the "
                "model without NMS and run it natively on-device using the Vision "
                "framework or a custom Swift implementation."
            ),
            priority=2,
        ),
    ],
    DiagnosisCategory.RUNTIME_FAILURE: [
        Suggestion(
            title="Retest on a different compute unit",
            description=(
                "Some ops behave differently on GPU vs Neural Engine. Try converting "
                "with CPU_ONLY or CPU_AND_GPU to isolate the issue."
            ),
            priority=1,
        ),
        Suggestion(
            title="Verify input preprocessing",
            description=(
                "Ensure the input tensor format, normalization, and data type match "
                "what the model expects. Mismatches cause silent runtime failures."
            ),
            priority=2,
        ),
    ],
    DiagnosisCategory.NUMERIC_INSTABILITY: [
        Suggestion(
            title="Use float32 precision",
            description=(
                "Float16 conversion can introduce NaN/Inf for models with large "
                "activations. Convert with compute_precision=ct.precision.FLOAT32 "
                "to verify if precision is the root cause."
            ),
            priority=1,
        ),
        Suggestion(
            title="Add numerical safeguards",
            description=(
                "Insert torch.clamp() around divisions and exponentials in the model "
                "to prevent overflow/underflow. Pay special attention to softmax, "
                "log, and normalization layers."
            ),
            priority=2,
        ),
    ],
    DiagnosisCategory.POSTPROCESS_BOTTLENECK: [
        Suggestion(
            title="Move NMS outside the model",
            description=(
                "Non-Maximum Suppression is a major postprocessing bottleneck. "
                "Remove it from the model graph and run it using Apple's Vision "
                "framework (VNDetectedObjectObservation) for hardware-optimized NMS."
            ),
            priority=1,
        ),
        Suggestion(
            title="Reduce candidate boxes before postprocess",
            description=(
                "Apply a confidence threshold early to reduce the number of boxes "
                "entering NMS. Fewer candidates means faster postprocessing."
            ),
            priority=2,
        ),
        Suggestion(
            title="Simplify postprocessing logic",
            description=(
                "Complex decode/filter steps inside the model slow down the pipeline. "
                "Move these to the host side where they can run in parallel with the "
                "next inference call."
            ),
            priority=3,
        ),
    ],
    DiagnosisCategory.MEMORY_ISSUE: [
        Suggestion(
            title="Reduce input size",
            description=(
                "Lowering the input resolution (e.g., 640 -> 320) reduces memory "
                "consumption quadratically. This is the most effective single change "
                "for memory issues."
            ),
            priority=1,
        ),
        Suggestion(
            title="Use a smaller model variant",
            description=(
                "Switch to a lighter model (e.g., YOLOv8n instead of YOLOv8x). "
                "Smaller models use less memory and often still meet accuracy targets "
                "for on-device use cases."
            ),
            priority=2,
        ),
        Suggestion(
            title="Try CPU_ONLY compute unit",
            description=(
                "Running on CPU avoids duplicating model weights between CPU and "
                "GPU/Neural Engine memory. This can reduce peak memory usage."
            ),
            priority=3,
        ),
    ],
    DiagnosisCategory.UNKNOWN: [
        Suggestion(
            title="Review raw error details",
            description=(
                "Check the raw error message and traceback in the report JSON for "
                "specific clues. Search for the error text in coremltools GitHub issues."
            ),
            priority=1,
        ),
        Suggestion(
            title="Test with a minimal model",
            description=(
                "Create a minimal nn.Module that reproduces the issue to isolate "
                "whether the problem is in the model architecture or the conversion "
                "pipeline."
            ),
            priority=2,
        ),
    ],
}


def _suggestions_from_diagnosis(diagnosis_result: DiagnosisResult) -> list[Suggestion]:
    suggestions: list[Suggestion] = []
    seen_titles: set[str] = set()

    for diag in diagnosis_result.diagnoses:
        category_suggestions = _CATEGORY_SUGGESTIONS.get(diag.category, [])
        for s in category_suggestions:
            if s.title not in seen_titles:
                seen_titles.add(s.title)
                suggestions.append(
                    Suggestion(
                        title=s.title,
                        description=s.description,
                        priority=s.priority,
                    )
                )

    return suggestions


def _suggestions_from_benchmark(benchmark: BenchmarkResult) -> list[Suggestion]:
    suggestions: list[Suggestion] = []

    # Slow inference
    if benchmark.inference.mean_ms > 100:
        suggestions.append(
            Suggestion(
                title="Retest on a different compute unit",
                description=(
                    f"Inference latency is {benchmark.inference.mean_ms:.1f}ms. "
                    "Try CPU_AND_NE (Neural Engine) which is optimized for "
                    "convolutional models and may significantly reduce latency."
                ),
                priority=2,
            )
        )

    if benchmark.inference.mean_ms > 200:
        suggestions.append(
            Suggestion(
                title="Reduce input size",
                description=(
                    f"Inference latency is {benchmark.inference.mean_ms:.1f}ms. "
                    "Reducing input resolution (e.g., 640 -> 416 or 320) can cut "
                    "latency substantially with modest accuracy trade-off."
                ),
                priority=1,
            )
        )

    # Low FPS
    if 0 < benchmark.estimated_fps < 15:
        suggestions.append(
            Suggestion(
                title="Optimize for real-time performance",
                description=(
                    f"Estimated FPS is {benchmark.estimated_fps:.1f}, below the "
                    "15 FPS threshold for smooth real-time processing. Consider a "
                    "smaller model variant or lower input resolution."
                ),
                priority=1,
            )
        )

    # High memory
    if benchmark.peak_memory_mb is not None and benchmark.peak_memory_mb > 500:
        suggestions.append(
            Suggestion(
                title="Reduce model memory footprint",
                description=(
                    f"Peak memory usage is {benchmark.peak_memory_mb:.0f} MB. "
                    "Consider quantizing the model (e.g., int8 weight quantization "
                    "via coremltools) or using a smaller variant."
                ),
                priority=2,
            )
        )

    # High variance in latency
    if benchmark.inference.std_ms > 0 and benchmark.inference.mean_ms > 0:
        cv = benchmark.inference.std_ms / benchmark.inference.mean_ms
        if cv > 0.3:
            suggestions.append(
                Suggestion(
                    title="Investigate latency variance",
                    description=(
                        f"Inference latency has high variance (CV={cv:.2f}). "
                        "This may indicate thermal throttling or contention with "
                        "other processes. Run benchmarks in a controlled environment."
                    ),
                    priority=3,
                )
            )

    return suggestions


def _suggestions_from_validation(validation: ValidationResult) -> list[Suggestion]:
    suggestions: list[Suggestion] = []

    if validation.status == ValidationStatus.FAIL:
        # Check for numeric issues
        has_numeric_issue = any(
            "nan" in c.detail.lower() or "inf" in c.detail.lower()
            for c in validation.checks
            if c.status == ValidationStatus.FAIL
        )
        if has_numeric_issue:
            suggestions.append(
                Suggestion(
                    title="Switch to float32 precision",
                    description=(
                        "Validation detected NaN or Inf in converted model outputs. "
                        "Try converting with float32 precision to eliminate "
                        "half-precision rounding errors."
                    ),
                    priority=1,
                )
            )

        # Check for accuracy drift
        has_accuracy_drift = any(
            "tolerance" in c.detail.lower() or "exceed" in c.detail.lower()
            for c in validation.checks
            if c.status == ValidationStatus.FAIL
        )
        if has_accuracy_drift:
            suggestions.append(
                Suggestion(
                    title="Increase validation tolerance or check preprocessing",
                    description=(
                        "Output values exceed the tolerance threshold. This may be "
                        "acceptable for the use case (relax tolerance) or may indicate "
                        "a preprocessing mismatch between PyTorch and Core ML."
                    ),
                    priority=2,
                )
            )

    if validation.status == ValidationStatus.WARNING:
        suggestions.append(
            Suggestion(
                title="Review validation warnings",
                description=(
                    "Some validation checks produced warnings. Review the detailed "
                    "check results to determine if the differences are acceptable "
                    "for your deployment scenario."
                ),
                priority=3,
            )
        )

    return suggestions


def generate_suggestions(
    diagnosis: DiagnosisResult,
    benchmark: BenchmarkResult | None = None,
    validation: ValidationResult | None = None,
) -> list[Suggestion]:
    all_suggestions: list[Suggestion] = []
    seen_titles: set[str] = set()

    # Collect from all sources
    sources: list[list[Suggestion]] = [_suggestions_from_diagnosis(diagnosis)]
    if benchmark is not None:
        sources.append(_suggestions_from_benchmark(benchmark))
    if validation is not None:
        sources.append(_suggestions_from_validation(validation))

    # Merge, deduplicating by title (keep higher priority version)
    for source in sources:
        for s in source:
            if s.title not in seen_titles:
                seen_titles.add(s.title)
                all_suggestions.append(s)

    # Sort by priority (lower number = higher priority)
    all_suggestions.sort(key=lambda s: s.priority)

    return all_suggestions
