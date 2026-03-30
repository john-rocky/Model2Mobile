"""Data models for pipeline results and states."""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class ReadinessState(str, enum.Enum):
    READY = "READY"
    PARTIAL = "PARTIAL"
    NOT_READY = "NOT_READY"


class ValidationStatus(str, enum.Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"


class DiagnosisCategory(str, enum.Enum):
    UNSUPPORTED_OP = "unsupported_op"
    DYNAMIC_SHAPE = "dynamic_shape"
    OUTPUT_SHAPE_MISMATCH = "output_shape_mismatch"
    RUNTIME_FAILURE = "runtime_failure"
    NUMERIC_INSTABILITY = "numeric_instability"
    POSTPROCESS_BOTTLENECK = "postprocess_bottleneck"
    MEMORY_ISSUE = "memory_issue"
    UNKNOWN = "unknown"


@dataclass
class ModelInfo:
    """Metadata about the input model."""

    path: str
    parameter_count: int = 0
    input_shape: tuple[int, ...] = ()
    estimated_size_mb: float = 0.0
    architecture: str = "unknown"
    has_dynamic_shapes: bool = False
    op_summary: dict[str, int] = field(default_factory=dict)
    task: str = "unknown"


@dataclass
class ConversionResult:
    """Result of Core ML conversion attempt."""

    success: bool
    coreml_path: str | None = None
    coreml_size_mb: float = 0.0
    compute_unit: str = "ALL"
    conversion_time_s: float = 0.0
    error_message: str | None = None
    raw_error: str | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class Diagnosis:
    """A single normalized diagnosis entry."""

    category: DiagnosisCategory
    raw_error: str
    likely_cause: str
    suggested_steps: list[str] = field(default_factory=list)


@dataclass
class DiagnosisResult:
    """Aggregated diagnosis output."""

    diagnoses: list[Diagnosis] = field(default_factory=list)
    primary_category: DiagnosisCategory = DiagnosisCategory.UNKNOWN

    @property
    def has_issues(self) -> bool:
        return len(self.diagnoses) > 0


@dataclass
class LatencyStats:
    """Latency statistics for a single stage."""

    mean_ms: float = 0.0
    median_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    p95_ms: float = 0.0
    std_ms: float = 0.0
    samples: int = 0


@dataclass
class BenchmarkResult:
    """Result of on-device or local benchmark."""

    success: bool
    device_name: str = "unknown"
    compute_unit: str = "ALL"
    preprocess: LatencyStats = field(default_factory=LatencyStats)
    inference: LatencyStats = field(default_factory=LatencyStats)
    postprocess: LatencyStats = field(default_factory=LatencyStats)
    end_to_end: LatencyStats = field(default_factory=LatencyStats)
    estimated_fps: float = 0.0
    peak_memory_mb: float | None = None
    warmup_iterations: int = 0
    measurement_iterations: int = 0
    error_message: str | None = None

    # Improvement: multi-compute-unit comparison
    compute_unit_comparison: dict[str, dict[str, float]] | None = None


@dataclass
class ValidationCheck:
    """A single validation check result."""

    name: str
    status: ValidationStatus
    detail: str
    expected: Any = None
    actual: Any = None
    tolerance: float | None = None


@dataclass
class ValidationResult:
    """Aggregated validation output."""

    status: ValidationStatus
    checks: list[ValidationCheck] = field(default_factory=list)
    error_message: str | None = None

    @property
    def pass_count(self) -> int:
        return sum(1 for c in self.checks if c.status == ValidationStatus.PASS)

    @property
    def warning_count(self) -> int:
        return sum(1 for c in self.checks if c.status == ValidationStatus.WARNING)

    @property
    def fail_count(self) -> int:
        return sum(1 for c in self.checks if c.status == ValidationStatus.FAIL)


@dataclass
class Suggestion:
    """A suggested next action."""

    title: str
    description: str
    priority: int = 0  # lower = higher priority


@dataclass
class RunResult:
    """Aggregate result of a full pipeline run."""

    readiness: ReadinessState
    model_info: ModelInfo
    conversion: ConversionResult
    diagnosis: DiagnosisResult
    benchmark: BenchmarkResult | None = None
    validation: ValidationResult | None = None
    optimization: OptimizationResult | None = None
    suggestions: list[Suggestion] = field(default_factory=list)
    run_id: str = ""
    timestamp: str = ""
    output_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        d = asdict(self)
        d["readiness"] = self.readiness.value
        if self.validation:
            d["validation"]["status"] = self.validation.status.value
            for i, c in enumerate(d["validation"]["checks"]):
                c["status"] = self.validation.checks[i].status.value
        d["diagnosis"]["primary_category"] = self.diagnosis.primary_category.value
        for i, diag in enumerate(d["diagnosis"]["diagnoses"]):
            diag["category"] = self.diagnosis.diagnoses[i].category.value
        if self.optimization:
            d["optimization"] = self.optimization.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunResult:
        """Reconstruct a RunResult from a JSON-compatible dict."""
        readiness = ReadinessState(d.get("readiness", "NOT_READY"))

        # Model info
        mi = d.get("model_info", {})
        model_info = ModelInfo(
            path=mi.get("path", ""),
            parameter_count=mi.get("parameter_count", 0),
            input_shape=tuple(mi.get("input_shape", ())),
            estimated_size_mb=mi.get("estimated_size_mb", 0.0),
            architecture=mi.get("architecture", "unknown"),
            has_dynamic_shapes=mi.get("has_dynamic_shapes", False),
            op_summary=mi.get("op_summary", {}),
            task=mi.get("task", "unknown"),
        )

        # Conversion
        cv = d.get("conversion", {})
        conversion = ConversionResult(
            success=cv.get("success", False),
            coreml_path=cv.get("coreml_path"),
            coreml_size_mb=cv.get("coreml_size_mb", 0.0),
            compute_unit=cv.get("compute_unit", "ALL"),
            conversion_time_s=cv.get("conversion_time_s", 0.0),
            error_message=cv.get("error_message"),
            raw_error=cv.get("raw_error"),
            warnings=cv.get("warnings", []),
        )

        # Diagnosis
        dg = d.get("diagnosis", {})
        diagnoses = []
        for dd in dg.get("diagnoses", []):
            diagnoses.append(Diagnosis(
                category=DiagnosisCategory(dd.get("category", "unknown")),
                raw_error=dd.get("raw_error", ""),
                likely_cause=dd.get("likely_cause", ""),
                suggested_steps=dd.get("suggested_steps", []),
            ))
        diagnosis = DiagnosisResult(
            diagnoses=diagnoses,
            primary_category=DiagnosisCategory(dg.get("primary_category", "unknown")),
        )

        # Benchmark
        benchmark: BenchmarkResult | None = None
        bm = d.get("benchmark")
        if bm and isinstance(bm, dict):
            benchmark = BenchmarkResult(
                success=bm.get("success", False),
                device_name=bm.get("device_name", "unknown"),
                compute_unit=bm.get("compute_unit", "ALL"),
                preprocess=LatencyStats(**bm["preprocess"]) if "preprocess" in bm else LatencyStats(),
                inference=LatencyStats(**bm["inference"]) if "inference" in bm else LatencyStats(),
                postprocess=LatencyStats(**bm["postprocess"]) if "postprocess" in bm else LatencyStats(),
                end_to_end=LatencyStats(**bm["end_to_end"]) if "end_to_end" in bm else LatencyStats(),
                estimated_fps=bm.get("estimated_fps", 0.0),
                peak_memory_mb=bm.get("peak_memory_mb"),
                warmup_iterations=bm.get("warmup_iterations", 0),
                measurement_iterations=bm.get("measurement_iterations", 0),
                error_message=bm.get("error_message"),
                compute_unit_comparison=bm.get("compute_unit_comparison"),
            )

        # Validation
        validation: ValidationResult | None = None
        vl = d.get("validation")
        if vl and isinstance(vl, dict) and "status" in vl:
            checks = []
            for vc in vl.get("checks", []):
                checks.append(ValidationCheck(
                    name=vc.get("name", ""),
                    status=ValidationStatus(vc.get("status", "FAIL")),
                    detail=vc.get("detail", ""),
                    expected=vc.get("expected"),
                    actual=vc.get("actual"),
                    tolerance=vc.get("tolerance"),
                ))
            validation = ValidationResult(
                status=ValidationStatus(vl["status"]),
                checks=checks,
                error_message=vl.get("error_message"),
            )

        # Optimization
        optimization: OptimizationResult | None = None
        opt = d.get("optimization")
        if opt and isinstance(opt, dict):
            variants = []
            for ov in opt.get("variants", []):
                variants.append(OptimizationVariant(
                    name=ov.get("name", ""),
                    strategy=ov.get("strategy", ""),
                    model_size_mb=ov.get("model_size_mb", 0.0),
                    inference_mean_ms=ov.get("inference_mean_ms", 0.0),
                    inference_p95_ms=ov.get("inference_p95_ms", 0.0),
                    estimated_fps=ov.get("estimated_fps", 0.0),
                    size_reduction_pct=ov.get("size_reduction_pct", 0.0),
                    speedup_pct=ov.get("speedup_pct", 0.0),
                    error=ov.get("error"),
                ))
            optimization = OptimizationResult(
                original_size_mb=opt.get("original_size_mb", 0.0),
                original_inference_ms=opt.get("original_inference_ms", 0.0),
                variants=variants,
                recommended=opt.get("recommended", ""),
                recommendation_reason=opt.get("recommendation_reason", ""),
            )

        # Suggestions
        suggestions = []
        for s in d.get("suggestions", []):
            suggestions.append(Suggestion(
                title=s.get("title", ""),
                description=s.get("description", ""),
                priority=s.get("priority", 0),
            ))

        return cls(
            readiness=readiness,
            model_info=model_info,
            conversion=conversion,
            diagnosis=diagnosis,
            benchmark=benchmark,
            validation=validation,
            optimization=optimization,
            suggestions=suggestions,
            run_id=d.get("run_id", ""),
            timestamp=d.get("timestamp", ""),
            output_dir=d.get("output_dir", ""),
        )

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_json(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")


@dataclass
class OptimizationVariant:
    """Result of a single optimization strategy trial."""

    name: str                    # e.g. "int8", "palettize_4bit"
    strategy: str                # human-readable description
    model_size_mb: float
    inference_mean_ms: float
    inference_p95_ms: float
    estimated_fps: float
    size_reduction_pct: float    # vs original
    speedup_pct: float           # vs original
    error: str | None = None


@dataclass
class OptimizationResult:
    """Aggregated results from trying multiple optimization strategies."""

    original_size_mb: float
    original_inference_ms: float
    variants: list[OptimizationVariant] = field(default_factory=list)
    recommended: str = ""            # name of best variant
    recommendation_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_json(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")
