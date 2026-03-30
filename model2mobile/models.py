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
        return d

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save_json(self, path: Path) -> None:
        path.write_text(self.to_json(), encoding="utf-8")
