# Model2Mobile

**Evaluate whether your PyTorch vision model is actually ready for on-device deployment.**

Model2Mobile answers a simple question: ***Can my model run on iPhone?***

Give it a `.pt` file — local or URL — and get a deployment readiness verdict with benchmark numbers, validation results, and actionable next steps. No Core ML knowledge required.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b96c1aa0-890f-44df-9098-5044a169e155" alt="Model2Mobile HTML Report" width="700">
</p>

---

## Why Model2Mobile?

Converting a model is not the same as deploying a model.

A model can convert successfully but still be unusable — too slow, too much memory, wrong outputs, broken ops.  Model2Mobile evaluates the **full deployment path** in one command:

| Step | What it checks |
|------|----------------|
| **Conversion** | Can PyTorch → Core ML succeed? If not, what op failed? |
| **Benchmark** | How fast is inference? Where is the bottleneck? |
| **Validation** | Does the converted model produce the same results? |
| **Diagnosis** | What went wrong, classified into actionable categories |
| **Suggestion** | What should you try next to fix it? |

The output is not just a `.mlpackage` — it's a **readiness report** that tells you whether to ship, iterate, or rethink.

---

## Quick Start

```bash
pip install -e .
```

```bash
# Just point it at a model
model2mobile run --model ./model.pt

# Or a URL — it downloads automatically
model2mobile run --model https://example.com/model.pt
```

That's it. Input size is auto-detected. Missing packages are auto-installed.

---

## What You Get

### Readiness Verdict

Every run ends with one of three states:

| State | Meaning |
|-------|---------|
| `READY` | Conversion, runtime, and validation all passed within thresholds |
| `PARTIAL` | Mostly works, but has warnings or performance concerns |
| `NOT_READY` | Conversion failed, runtime crashed, or validation failed |

### Benchmark Breakdown

Latency is split by stage so you know **where** the bottleneck is:

```
| Stage       | Mean (ms) | Median | P95   |
|-------------|-----------|--------|-------|
| Preprocess  | 0.88      | 0.87   | 0.98  |
| Inference   | 29.6      | 29.1   | 31.2  |
| Postprocess | 0.01      | 0.01   | 0.01  |
| End-to-End  | 30.5      | 30.0   | 32.1  |

Estimated FPS: 31.9
```

### Reports

Each run generates a full output directory:

```
outputs/20260330_040703_368840/
├── report.html        # Visual report (open in browser)
├── report.md          # Markdown summary
├── summary.json       # Full structured result
├── metrics.json       # Benchmark data
├── diagnosis.json     # Diagnosed issues
├── validation.json    # Validation checks
├── run.log            # Run metadata
└── model.mlpackage    # Converted model (if successful)
```

---

## Auto-Fix with Recipes

Model2Mobile doesn't just diagnose failures — it **fixes known problems automatically**.

The recipe system contains accumulated knowledge about conversion pitfalls. When a model fails to convert, matching recipes patch the model and retry:

```
Stage 1/4: Converting to Core ML...
  Applied recipes: detection_unwrap        ← auto-fixed before conversion
  Conversion succeeded (7.2s, 61.9 MB)
```

Built-in recipes:

| Recipe | What it fixes |
|--------|--------------|
| `detection_unwrap` | Detection models returning `List[Dict]` (FCOS, FasterRCNN, SSD, etc.) |
| `nms_strip` | Removes NMS for on-device postprocessing |
| `silu_replace` | Replaces SiLU with trace-friendly `x * sigmoid(x)` |
| `dynamic_to_static` | Patches dynamic shape operations |

Adding a new recipe is one file — the system gets smarter over time.

---

## CLI Modes

### `run` — Full pipeline (primary)

```bash
model2mobile run --model ./model.pt
model2mobile run --model ./model.pt --input-size 320 --compute-unit CPU_AND_NE
model2mobile run --model ./model.pt --compare-units    # benchmark all compute units
```

### `init` — Interactive guided setup

```bash
model2mobile init
```

### Expert — Individual stages

```bash
model2mobile convert   --model ./model.pt
model2mobile benchmark --coreml ./model.mlpackage
model2mobile validate  --model ./model.pt --coreml ./model.mlpackage
```

---

## Diagnosis

When something fails, the error is classified — not just dumped:

| Category | Meaning |
|----------|---------|
| `unsupported_op` | Core ML can't convert an operator |
| `dynamic_shape` | Data-dependent tensor shapes |
| `output_shape_mismatch` | Shape divergence after conversion |
| `runtime_failure` | Prediction crashes |
| `numeric_instability` | NaN / Inf in outputs |
| `postprocess_bottleneck` | Postprocessing slower than inference |
| `memory_issue` | Excessive memory usage |

Each diagnosis includes a likely cause and suggested next steps.

---

## How It Handles Any `.pt` File

Model2Mobile is designed for users who just downloaded a model and want to try it:

1. **URL support** — pass a URL instead of a file path
2. **Auto-install** — if the `.pt` file needs `ultralytics`, `timm`, or any package, it's installed automatically
3. **Auto-detect input size** — inferred from model attributes or trial forward passes
4. **Auto-fix** — recipe system patches known issues before you even see an error

---

## Configuration

All options can be passed as CLI flags or in a YAML file:

```yaml
model_path: ./model.pt
task: detect
input_size: 640
compute_unit: ALL
benchmark_enabled: true
validation_enabled: true
latency_threshold_ms: 100.0
fps_threshold: 15.0
```

```bash
model2mobile run --model ./model.pt --config config.yaml
```

---

## Requirements

- Python 3.10+
- macOS (required for Core ML runtime)
- PyTorch >= 2.0
- coremltools >= 7.0

---

## Scope (v1)

- **Input**: PyTorch (`.pt`, `.pth`, TorchScript)
- **Target**: Core ML (`.mlpackage`)
- **Task**: Object detection
- **Benchmark**: Local Mac (on-device iPhone support planned)

---

## License

MIT
