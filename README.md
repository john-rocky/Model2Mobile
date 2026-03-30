# Model2Mobile

**Evaluate whether your PyTorch vision model is actually ready for on-device deployment.**

Model2Mobile is a CLI tool that answers a simple question: *Can my model run on iPhone?*

It goes beyond conversion — it evaluates deployment readiness by running conversion, benchmark, validation, and diagnosis in a single pipeline, then tells you whether your model is `READY`, `PARTIAL`, or `NOT_READY`, and what to do next.

## What This Solves

You have a PyTorch object detection model. You want to try it on iPhone. But you don't want to manually learn Core ML conversion details, debug cryptic errors, or wonder if the converted model actually behaves the same.

Model2Mobile handles the full evaluation:

- **Conversion** — Attempts Core ML conversion and classifies failures
- **Benchmark** — Measures preprocess, inference, and postprocess latency separately
- **Validation** — Compares PyTorch vs Core ML outputs for consistency
- **Diagnosis** — Normalizes errors into actionable categories
- **Report** — Generates human-readable summaries and machine-readable JSON

## Quick Start

```bash
pip install -e .
```

```bash
# Local file - input size auto-detected
model2mobile run --model ./model.pt

# URL - downloads automatically
model2mobile run --model https://example.com/yolov8n.pt

# Explicit input size
model2mobile run --model ./model.pt --input-size 640
```

One command gives you:

- A readiness verdict (`READY` / `PARTIAL` / `NOT_READY`)
- Benchmark breakdown by stage
- Validation results
- Diagnosed issues with suggested fixes
- Full report (Markdown, HTML, JSON) in the output directory

## Readiness States

| State | Meaning |
|-------|---------|
| `READY` | Conversion, runtime, and validation all passed within acceptable thresholds |
| `PARTIAL` | Some stages passed but there are warnings, performance issues, or validation concerns |
| `NOT_READY` | Conversion failed, runtime failed, or validation failed critically |

Results are **scenario-specific** — they depend on your model, input size, device, and compute unit.

## CLI Modes

### Primary: `run`

Full pipeline in one command.

```bash
model2mobile run \
  --model ./model.pt \
  --task detect \
  --input-size 640 \
  --compute-unit ALL \
  --compare-units
```

### Guided: `init`

Interactive setup that walks you through configuration.

```bash
model2mobile init
```

### Expert: Individual Stages

Run each stage independently.

```bash
# Convert only
model2mobile convert --model ./model.pt --input-size 640

# Benchmark an existing .mlpackage
model2mobile benchmark --coreml ./model.mlpackage --input-size 640

# Validate PyTorch vs Core ML
model2mobile validate --model ./model.pt --coreml ./model.mlpackage
```

## Output Structure

Each run produces a dedicated directory under `outputs/`:

```
outputs/20240315_143022_a1b2c3/
├── report.md          # Human-readable summary
├── report.html        # Visual report (open in browser)
├── summary.json       # Full structured result
├── metrics.json       # Benchmark data
├── diagnosis.json     # Diagnosed issues
├── validation.json    # Validation checks
├── run.log            # Run metadata
└── model.mlpackage    # Converted Core ML model (if successful)
```

## Diagnosis Categories

When something fails, Model2Mobile classifies the issue:

| Category | Example |
|----------|---------|
| `unsupported_op` | Model uses an operator Core ML can't convert |
| `dynamic_shape` | Model has data-dependent tensor shapes |
| `output_shape_mismatch` | Output shapes differ between PyTorch and Core ML |
| `runtime_failure` | Core ML model fails during prediction |
| `numeric_instability` | NaN/Inf values in converted model |
| `postprocess_bottleneck` | Postprocessing slower than inference |
| `memory_issue` | Excessive memory usage |

Each diagnosis includes the raw error, likely cause, and suggested next steps.

## Supported Scope (v1)

- **Input**: PyTorch models (`.pt`, `.pth`, TorchScript)
- **Target**: Core ML (`.mlpackage`)
- **Device**: Mac (local benchmark), iPhone (future)
- **Task**: Object detection

## Configuration

CLI flags, or a YAML config file:

```yaml
model_path: ./model.pt
task: detect
input_size: 640
compute_unit: ALL
benchmark_enabled: true
validation_enabled: true
warmup_iterations: 5
measurement_iterations: 20
latency_threshold_ms: 100.0
fps_threshold: 15.0
```

```bash
model2mobile run --model ./model.pt --config config.yaml
```

## Requirements

- Python >= 3.10
- macOS (required for Core ML)
- PyTorch >= 2.0
- coremltools >= 7.0

## Project Structure

```
model2mobile/
├── convert/     # Core ML conversion
├── benchmark/   # Performance measurement
├── validate/    # PyTorch vs Core ML comparison
├── diagnose/    # Error classification
├── suggest/     # Next-action recommendations
├── report/      # Markdown, JSON, HTML generation
├── cli.py       # CLI entry point
├── pipeline.py  # Run orchestration
├── config.py    # Configuration model
└── models.py    # Data models
```

## License

MIT
