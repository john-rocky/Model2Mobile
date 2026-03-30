"""Core ML model benchmark runner for local Mac execution."""

from __future__ import annotations

import platform
import resource
import time
import traceback
from typing import Any

import coremltools as ct
import numpy as np
from PIL import Image

from model2mobile.config import RunConfig
from model2mobile.models import BenchmarkResult, LatencyStats

# Compute unit name -> coremltools enum mapping
_COMPUTE_UNITS = {
    "ALL": ct.ComputeUnit.ALL,
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
}


def _compute_stats(timings: list[float]) -> LatencyStats:
    """Compute latency statistics from a list of timing values in seconds."""
    if not timings:
        return LatencyStats()

    arr = np.array(timings) * 1000.0  # convert to ms
    return LatencyStats(
        mean_ms=float(np.mean(arr)),
        median_ms=float(np.median(arr)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        p95_ms=float(np.percentile(arr, 95)),
        std_ms=float(np.std(arr)),
        samples=len(timings),
    )


def _create_dummy_input(input_size: int) -> Image.Image:
    """Create a deterministic dummy RGB image for benchmarking."""
    rng = np.random.RandomState(42)
    pixels = rng.randint(0, 256, (input_size, input_size, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def _prepare_input_dict(
    model: ct.models.MLModel, image: Image.Image
) -> dict[str, Any]:
    """Build the input dictionary expected by the Core ML model."""
    spec = model.get_spec()
    input_dict: dict[str, Any] = {}

    for inp in spec.description.input:
        name = inp.name
        input_type = inp.type.WhichOneof("Type")

        if input_type == "imageType":
            input_dict[name] = image
        elif input_type == "multiArrayType":
            shape = list(inp.type.multiArrayType.shape)
            if not shape:
                # Fall back to a sensible default
                shape = [1, 3, image.size[1], image.size[0]]
            input_dict[name] = np.array(image.resize(
                (shape[-1], shape[-2])
            )).astype(np.float32).transpose(2, 0, 1)[np.newaxis]
        else:
            # Unknown input type -- provide a placeholder array
            input_dict[name] = np.zeros((1,), dtype=np.float32)

    return input_dict


def _extract_outputs(prediction: Any) -> dict[str, np.ndarray]:
    """Extract output arrays from a Core ML prediction result."""
    outputs: dict[str, np.ndarray] = {}
    if isinstance(prediction, dict):
        for k, v in prediction.items():
            outputs[k] = np.asarray(v)
    else:
        outputs["output"] = np.asarray(prediction)
    return outputs


def _get_peak_memory_mb() -> float:
    """Return peak resident memory in MB using the resource module."""
    # On macOS, ru_maxrss is in bytes
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_maxrss / (1024 * 1024)


def _benchmark_single_unit(
    coreml_path: str,
    compute_unit_name: str,
    config: RunConfig,
    image: Image.Image,
) -> BenchmarkResult:
    """Run benchmark for a single compute unit configuration."""
    cu = _COMPUTE_UNITS.get(compute_unit_name, ct.ComputeUnit.ALL)
    model = ct.models.MLModel(coreml_path, compute_units=cu)

    preprocess_times: list[float] = []
    inference_times: list[float] = []
    postprocess_times: list[float] = []
    e2e_times: list[float] = []

    total_iters = config.warmup_iterations + config.measurement_iterations

    mem_before = _get_peak_memory_mb()

    for i in range(total_iters):
        is_measurement = i >= config.warmup_iterations

        # --- end-to-end start ---
        t_e2e_start = time.perf_counter()

        # Preprocess
        t_pre_start = time.perf_counter()
        input_dict = _prepare_input_dict(model, image)
        t_pre_end = time.perf_counter()

        # Inference
        t_inf_start = time.perf_counter()
        prediction = model.predict(input_dict)
        t_inf_end = time.perf_counter()

        # Postprocess
        t_post_start = time.perf_counter()
        _ = _extract_outputs(prediction)
        t_post_end = time.perf_counter()

        t_e2e_end = time.perf_counter()
        # --- end-to-end end ---

        if is_measurement:
            preprocess_times.append(t_pre_end - t_pre_start)
            inference_times.append(t_inf_end - t_inf_start)
            postprocess_times.append(t_post_end - t_post_start)
            e2e_times.append(t_e2e_end - t_e2e_start)

    mem_after = _get_peak_memory_mb()
    peak_memory = max(mem_before, mem_after)

    e2e_stats = _compute_stats(e2e_times)
    fps = 1000.0 / e2e_stats.mean_ms if e2e_stats.mean_ms > 0 else 0.0

    return BenchmarkResult(
        success=True,
        device_name=platform.node(),
        compute_unit=compute_unit_name,
        preprocess=_compute_stats(preprocess_times),
        inference=_compute_stats(inference_times),
        postprocess=_compute_stats(postprocess_times),
        end_to_end=e2e_stats,
        estimated_fps=round(fps, 2),
        peak_memory_mb=round(peak_memory, 2),
        warmup_iterations=config.warmup_iterations,
        measurement_iterations=config.measurement_iterations,
    )


def run_benchmark(coreml_path: str, config: RunConfig) -> BenchmarkResult:
    """Run Core ML model benchmark on the local Mac.

    Measures preprocess, inference, postprocess, and end-to-end latency.
    Optionally compares across multiple compute units.
    """
    try:
        image = _create_dummy_input(config.input_size)

        # Primary benchmark with the configured compute unit
        result = _benchmark_single_unit(
            coreml_path, config.compute_unit, config, image
        )

        # Optionally compare across all compute units
        if config.compare_compute_units:
            comparison: dict[str, dict[str, float]] = {}
            for unit_name in _COMPUTE_UNITS:
                try:
                    unit_result = _benchmark_single_unit(
                        coreml_path, unit_name, config, image
                    )
                    comparison[unit_name] = {
                        "inference_mean_ms": unit_result.inference.mean_ms,
                        "inference_median_ms": unit_result.inference.median_ms,
                        "inference_p95_ms": unit_result.inference.p95_ms,
                        "e2e_mean_ms": unit_result.end_to_end.mean_ms,
                        "estimated_fps": unit_result.estimated_fps,
                        "peak_memory_mb": unit_result.peak_memory_mb or 0.0,
                    }
                except Exception:
                    # Some compute units may not be available on this machine
                    comparison[unit_name] = {"error": -1.0}
            result.compute_unit_comparison = comparison

        return result

    except Exception as exc:
        return BenchmarkResult(
            success=False,
            device_name=platform.node(),
            compute_unit=config.compute_unit,
            warmup_iterations=config.warmup_iterations,
            measurement_iterations=config.measurement_iterations,
            error_message=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}",
        )
