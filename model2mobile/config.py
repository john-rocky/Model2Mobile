"""Run configuration for Model2Mobile."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class RunConfig:
    """Configuration for a single pipeline run."""

    model_path: str = ""
    task: str = "auto"
    input_size: int = 640
    input_size_auto: bool = True  # auto-detect input size from model if --input-size not given
    output_dir: str = "outputs"
    device: str = "local"  # "local" (Mac) or "iphone"

    # Pipeline stage toggles
    benchmark_enabled: bool = True
    validation_enabled: bool = True
    codegen_enabled: bool = True
    optimize_enabled: bool = False

    # Benchmark settings
    warmup_iterations: int = 5
    measurement_iterations: int = 20

    # Conversion settings
    compute_unit: str = "ALL"  # ALL, CPU_ONLY, CPU_AND_GPU, CPU_AND_NE

    # Improvement: auto-compare all compute units
    compare_compute_units: bool = False

    # Validation settings
    confidence_tolerance: float = 0.05
    bbox_tolerance: float = 5.0  # pixels

    # Improvement: configurable performance thresholds
    latency_threshold_ms: float = 100.0  # max acceptable inference latency
    fps_threshold: float = 15.0  # min acceptable FPS
    memory_threshold_mb: float = 500.0  # max acceptable peak memory

    # Output control
    verbose: bool = False
    quiet: bool = False  # suppress intermediate output, show only final result

    @classmethod
    def from_yaml(cls, path: str | Path) -> RunConfig:
        """Load config from a YAML file."""
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str | Path) -> None:
        """Save config to a YAML file."""
        from dataclasses import asdict

        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    def resolve_output_dir(self, run_id: str) -> Path:
        """Return the run-specific output directory."""
        p = Path(self.output_dir) / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p
