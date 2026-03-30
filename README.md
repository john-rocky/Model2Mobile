# Model2Mobile

**Take your PyTorch model from training to iPhone in one command.**

Model2Mobile answers a simple question: ***Can my model run on iPhone?***

Give it a `.pt` file -- local or URL -- and get a deployment readiness verdict with benchmark numbers, validation results, Swift integration code, and actionable next steps. No Core ML knowledge required.

<p align="center">
  <img src="https://github.com/user-attachments/assets/b96c1aa0-890f-44df-9098-5044a169e155" alt="Model2Mobile HTML Report" width="700">
</p>

---

## Why Model2Mobile?

Converting a model is not the same as deploying a model.

A model can convert successfully but still be unusable -- too slow, too much memory, wrong outputs, broken ops. Model2Mobile evaluates the **full deployment path** in one command:

| Step | What it checks |
|------|----------------|
| **Conversion** | Can PyTorch to Core ML succeed? If not, what op failed? |
| **Auto-Fix** | 10 built-in recipes patch known issues before you even see an error |
| **Benchmark** | How fast is inference? Where is the bottleneck? (Mac or iPhone) |
| **Validation** | Does the converted model produce the same results? |
| **Optimization** | Which quantization variant gives the best size/speed tradeoff? |
| **Code Generation** | Drop-in Swift files for Xcode -- Predictor and PostProcessor |
| **Diagnosis** | What went wrong, classified into actionable categories |

The output is not just a `.mlpackage` -- it is a **deployment-ready package** with Swift code, benchmark data, and a readiness report that tells you whether to ship, iterate, or rethink.

---

## Quick Start

```bash
pip install -e .
```

```bash
# Just point it at a model
model2mobile run --model ./model.pt

# Or a URL -- it downloads automatically
model2mobile run --model https://example.com/model.pt
```

That's it. Input size is auto-detected. Missing packages are auto-installed. Swift code is generated. A full HTML report is produced.

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

### Reports and Generated Files

Each run produces a complete output directory:

```
outputs/run_20260330/
├── ModelPredictor.swift        <- ready-to-use inference class
├── ModelPostProcessor.swift    <- NMS + decode logic (when applicable)
├── model.mlpackage             <- converted Core ML model
├── report.html                 <- visual report (open in browser)
├── report.md                   <- markdown summary
├── summary.json                <- full structured result
├── metrics.json                <- benchmark data
├── diagnosis.json              <- diagnosed issues
├── validation.json             <- validation checks
└── run.log                     <- run metadata
```

---

## Auto-Fix Recipes

Model2Mobile does not just diagnose failures -- it **fixes known problems automatically**.

The recipe system contains accumulated knowledge about conversion pitfalls. When a model fails to convert, matching recipes patch the model and retry:

```
Stage 1/4: Converting to Core ML...
  Applied recipes: detection_unwrap, silu_replace     <- auto-fixed before conversion
  Conversion succeeded (7.2s, 61.9 MB)
```

### All 10 Built-in Recipes

| Recipe | What it fixes |
|--------|--------------|
| `detection_unwrap` | Detection models returning `List[Dict]` (FCOS, FasterRCNN, SSD, RetinaNet, etc.) -- unwraps to raw tensors |
| `nms_strip` | Strips NMS modules that produce variable-length output incompatible with Core ML |
| `silu_replace` | Replaces `nn.SiLU` with trace-friendly `x * sigmoid(x)` for older coremltools |
| `dynamic_to_static` | Patches dynamic shape operations, disables export flags, fuses BN layers |
| `yolo_detect_head` | Strips YOLO v5/v8/v9/v10/v11 Detect heads to return raw feature maps |
| `transformer_attention` | Replaces `F.scaled_dot_product_attention` with manual matmul+softmax |
| `custom_activations` | Replaces Mish and GELU with trace-friendly elementary ops |
| `group_norm_patch` | Replaces `nn.GroupNorm` with a manual reshape+normalize implementation |
| `deformable_conv` | Replaces deformable convolutions (DCN/DCNv2) with standard `nn.Conv2d` |
| `channel_last_fix` | Forces contiguous memory layout to fix `channels_last` tracing issues |

Recipes are applied automatically when they match the model architecture or the conversion error. Adding a new recipe is one file -- the system gets smarter over time.

---

## Optimize

Find the best quantization variant for your model with a single command. Model2Mobile converts the model at multiple precision levels, benchmarks each one, and recommends the best tradeoff:

```bash
model2mobile optimize --model ./model.pt
model2mobile optimize --model ./model.mlpackage --target-fps 60
```

Example output:

```
┃ Variant     ┃ Size (MB) ┃ Reduction ┃ Inference ┃ FPS   ┃
┃ original    ┃      5.0  ┃         - ┃     4.63  ┃ 216.0 ┃
┃ int8_linear ┃      2.6  ┃    +47.8% ┃     4.65  ┃ 181.1 ┃ <- RECOMMENDED
```

The optimizer accepts either a `.pt` file (converts first) or a `.mlpackage` directly. When `--target-fps` is specified, the recommendation is tuned to meet that target.

---

## Swift Code Generation

Every successful run generates **production-ready Swift files** that you can drop directly into an Xcode project:

- **ModelPredictor.swift** -- handles model loading, image preprocessing, and inference. Supports both `UIImage` and `CVPixelBuffer` inputs. Configurable compute units (CPU, GPU, Neural Engine).

- **ModelPostProcessor.swift** -- when NMS was stripped during conversion, this file includes a complete on-device NMS implementation with configurable confidence and IoU thresholds. For non-detection models, it generates a structured output parser.

```
outputs/run_20260330/
├── ModelPredictor.swift        <- ready-to-use inference class
├── ModelPostProcessor.swift    <- NMS + decode logic
├── model.mlpackage
└── report.html
```

Usage in Swift:

```swift
let predictor = try ModelPredictor()
let outputs = try predictor.predict(image: uiImage)
let detections = ModelPostProcessor.process(
    outputs: outputs,
    confidenceThreshold: 0.25,
    iouThreshold: 0.45
)
```

To skip code generation, pass `--no-codegen`.

---

## iOS Device Benchmark

By default, benchmarks run on your local Mac. To benchmark on a connected iPhone instead, use the `--device` flag:

```bash
model2mobile run --model ./model.pt --device iphone
model2mobile benchmark --coreml ./model.mlpackage --device iphone
```

This uses `xcrun devicectl` to deploy and measure on the actual target hardware, giving you real-world Neural Engine and GPU performance numbers instead of Mac-based estimates.

---

## CLI Reference

### `run` -- Full pipeline (primary command)

Runs the complete evaluation: convert, benchmark, validate, diagnose, generate Swift code, and produce reports.

```bash
model2mobile run --model ./model.pt
model2mobile run --model ./model.pt --input-size 320 --compute-unit CPU_AND_NE
model2mobile run --model ./model.pt --device iphone
model2mobile run --model ./model.pt --compare-units       # benchmark all compute units
model2mobile run --model ./model.pt --no-codegen --quiet
```

| Flag | Description |
|------|-------------|
| `--model, -m` | Path or URL to PyTorch model (.pt / .pth / .torchscript) |
| `--task, -t` | Model task (default: `detect`) |
| `--input-size, -s` | Input image size (auto-detected if omitted) |
| `--output-dir, -o` | Output directory (default: `outputs`) |
| `--compute-unit` | `ALL`, `CPU_ONLY`, `CPU_AND_GPU`, or `CPU_AND_NE` |
| `--device, -d` | Benchmark device: `local` (default) or `iphone` |
| `--compare-units` | Benchmark across all compute units |
| `--no-benchmark` | Skip benchmark stage |
| `--no-validation` | Skip validation stage |
| `--no-codegen` | Skip Swift code generation |
| `--warmup` | Warmup iterations (default: 5) |
| `--iterations` | Measurement iterations (default: 20) |
| `--config` | Path to YAML config file |
| `--quiet, -q` | Suppress intermediate output, show only final result |
| `--verbose, -v` | Enable debug logging |

### `init` -- Interactive guided setup

```bash
model2mobile init
```

Walks you through model path, task, input size, compute unit, and benchmark/validation options interactively.

### `convert` -- Convert only

```bash
model2mobile convert --model ./model.pt
model2mobile convert --model ./model.pt --input-size 320 --compute-unit CPU_AND_NE
```

### `benchmark` -- Benchmark only

```bash
model2mobile benchmark --coreml ./model.mlpackage
model2mobile benchmark --coreml ./model.mlpackage --device iphone --compare-units
```

### `validate` -- Validate only

```bash
model2mobile validate --model ./model.pt --coreml ./model.mlpackage
```

### `optimize` -- Find optimal quantization

```bash
model2mobile optimize --model ./model.pt
model2mobile optimize --model ./model.mlpackage --target-fps 60
```

### `compare` -- Compare two runs

```bash
model2mobile compare outputs/run_a outputs/run_b
model2mobile compare outputs/run_a outputs/run_b -o comparison.html
```

Side-by-side comparison of readiness, benchmark, validation, and bottleneck across two runs. Useful for measuring the effect of input size changes, model modifications, or compute unit switches.

### `report` -- Regenerate reports

```bash
model2mobile report --run-dir outputs/run_20260330 --format html
```

Regenerates reports from an existing run directory. Formats: `markdown`, `html`, `json`, or `all`.

---

## Configuration

All options can be passed as CLI flags or in a YAML file:

```yaml
model_path: ./model.pt
task: detect
input_size: 640
device: local                  # "local" or "iphone"

# Pipeline toggles
benchmark_enabled: true
validation_enabled: true
codegen_enabled: true
optimize_enabled: false

# Conversion
compute_unit: ALL              # ALL, CPU_ONLY, CPU_AND_GPU, CPU_AND_NE
compare_compute_units: false

# Benchmark
warmup_iterations: 5
measurement_iterations: 20

# Performance thresholds
latency_threshold_ms: 100.0
fps_threshold: 15.0
memory_threshold_mb: 500.0

# Validation
confidence_tolerance: 0.05
bbox_tolerance: 5.0
```

```bash
model2mobile run --model ./model.pt --config config.yaml
```

---

## Diagnosis Categories

When something fails, the error is classified -- not just dumped:

| Category | Meaning |
|----------|---------|
| `unsupported_op` | Core ML cannot convert an operator |
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

1. **URL support** -- pass a URL instead of a file path; the model is downloaded and cached automatically
2. **Auto-install** -- if the `.pt` file needs `ultralytics`, `timm`, or any other package, it is installed automatically
3. **Auto-detect input size** -- inferred from model attributes or trial forward passes
4. **Auto-fix** -- 10 built-in recipes patch known issues before you even see an error
5. **Auto-codegen** -- Swift Predictor and PostProcessor files generated from the converted model spec

---

## Requirements

- Python 3.10+
- macOS (required for Core ML runtime)
- PyTorch >= 2.0
- coremltools >= 7.0
- Xcode Command Line Tools (for `--device iphone`)

---

## Scope

- **Input**: PyTorch (`.pt`, `.pth`, TorchScript)
- **Target**: Core ML (`.mlpackage`)
- **Task**: Object detection
- **Benchmark**: Local Mac or connected iPhone
- **Code Generation**: Swift (iOS 15+ / macOS 12+)

---

## License

MIT
