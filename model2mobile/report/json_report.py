"""JSON report generator for Model2Mobile pipeline results."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from model2mobile.models import RunResult


def _serialize(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


def _extract_metrics(result: RunResult) -> dict[str, Any]:
    data: dict[str, Any] = {
        "conversion": {
            "success": result.conversion.success,
            "conversion_time_s": result.conversion.conversion_time_s,
            "coreml_size_mb": result.conversion.coreml_size_mb,
            "compute_unit": result.conversion.compute_unit,
        },
        "model": {
            "parameter_count": result.model_info.parameter_count,
            "estimated_size_mb": result.model_info.estimated_size_mb,
            "input_shape": list(result.model_info.input_shape),
        },
    }
    if result.benchmark:
        bm = result.benchmark
        data["benchmark"] = {
            "device_name": bm.device_name,
            "compute_unit": bm.compute_unit,
            "estimated_fps": bm.estimated_fps,
            "peak_memory_mb": bm.peak_memory_mb,
            "warmup_iterations": bm.warmup_iterations,
            "measurement_iterations": bm.measurement_iterations,
            "preprocess": asdict(bm.preprocess),
            "inference": asdict(bm.inference),
            "postprocess": asdict(bm.postprocess),
            "end_to_end": asdict(bm.end_to_end),
        }
        if bm.compute_unit_comparison:
            data["benchmark"]["compute_unit_comparison"] = bm.compute_unit_comparison
    return data


def _extract_diagnosis(result: RunResult) -> dict[str, Any]:
    diag = result.diagnosis
    return {
        "has_issues": diag.has_issues,
        "primary_category": diag.primary_category.value,
        "diagnoses": [
            {
                "category": d.category.value,
                "raw_error": d.raw_error,
                "likely_cause": d.likely_cause,
                "suggested_steps": d.suggested_steps,
            }
            for d in diag.diagnoses
        ],
    }


def _extract_validation(result: RunResult) -> dict[str, Any]:
    if result.validation is None:
        return {"evaluated": False}
    val = result.validation
    return {
        "evaluated": True,
        "status": val.status.value,
        "pass_count": val.pass_count,
        "warning_count": val.warning_count,
        "fail_count": val.fail_count,
        "error_message": val.error_message,
        "checks": [
            {
                "name": c.name,
                "status": c.status.value,
                "detail": c.detail,
                "expected": c.expected,
                "actual": c.actual,
                "tolerance": c.tolerance,
            }
            for c in val.checks
        ],
    }


def save_json_reports(result: RunResult, output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    # metrics.json
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(_serialize(_extract_metrics(result)), encoding="utf-8")
    paths["metrics"] = metrics_path

    # diagnosis.json
    diagnosis_path = output_dir / "diagnosis.json"
    diagnosis_path.write_text(_serialize(_extract_diagnosis(result)), encoding="utf-8")
    paths["diagnosis"] = diagnosis_path

    # validation.json
    validation_path = output_dir / "validation.json"
    validation_path.write_text(_serialize(_extract_validation(result)), encoding="utf-8")
    paths["validation"] = validation_path

    # summary.json - full run result
    summary_path = output_dir / "summary.json"
    summary_path.write_text(_serialize(result.to_dict()), encoding="utf-8")
    paths["summary"] = summary_path

    return paths
