# Model2Mobile Specification

## 1. Overview

**Model2Mobile** is a repository and CLI tool for evaluating whether a user-provided **PyTorch vision model** is ready for **on-device deployment on iPhone**.

This project is **not** just a conversion script.

Its purpose is to answer the following questions:

- Can this model be converted to Core ML?
- Can it run on a real iPhone?
- Is it fast enough for practical use?
- Does it behave consistently with the original PyTorch pipeline?
- If not, what likely needs to be changed?

The primary output is **not only** a converted Core ML artifact.  
The primary output is a **deployment readiness result** with diagnostics, benchmark results, validation status, and suggested next actions.

---

## 2. Relationship to Existing Repository

There is already an existing repository named `CoreML-Models`.

That repository is a **model zoo of already converted Core ML models**.

Model2Mobile must be treated as a **separate repository with a different purpose**.

- `CoreML-Models` is for users who need an already converted model.
- `Model2Mobile` is for users who already have a model and want to know whether it can be deployed to iPhone.

Do not merge the responsibilities.

Model2Mobile may later link to `CoreML-Models`, but it must remain independent in concept, README, CLI, and folder structure.

---

## 3. Target User

The primary user is a **developer or ML engineer** who has a **PyTorch vision model** and wants to try it on iPhone, but may not know the Core ML deployment process.

The user may understand PyTorch, model inference, and vision tasks, but should **not** be required to understand Core ML internals.

The UX must feel like:

> “I want to know whether my model is usable on iPhone.”

The UX must **not** feel like:

> “I need to manually learn and orchestrate a long Core ML conversion process.”

---

## 4. Product Goals

The repository must achieve these goals in v1:

- Accept a user-provided PyTorch vision model as input
- Attempt conversion to Core ML
- Classify conversion or runtime failures into normalized categories
- Run on-device benchmark on a connected iPhone
- Validate end-to-end behavior against PyTorch reference output
- Generate a human-readable deployment readiness report
- Generate machine-readable JSON outputs for automation

The repository must be **CLI-first**.

GUI is out of scope as the primary product surface for v1, but a lightweight result-viewing layer is desirable after the core pipeline is stable.

---

## 5. Non-Goals for v1

The following are explicitly out of scope for v1:

- Training framework
- Model distillation workflow
- Automated architecture rewriting
- Android / TFLite support
- Segmentation support
- Pose support
- OCR support
- Fully general multi-task support
- Cloud service
- Web UI as the primary workflow
- Production inference SDK
- One-click support for every model architecture

v1 must focus narrowly on:

- **Input framework:** PyTorch
- **Target framework:** Core ML
- **Target device:** iPhone
- **Task:** object detection only

---

## 6. Supported Scope for v1

v1 scope:

- Input framework: PyTorch
- Target framework: Core ML
- Target device: iPhone
- Task: object detection only
- Preferred input: fixed input size image model
- Operation style: CLI
- Output: report + artifacts + diagnostics

Support should initially focus on a **limited number of realistic model families**.

The design should allow future extension, but the implementation must avoid pretending to support everything.

---

## 7. Core UX Principles

### 7.1 Single-command first experience

The user should start from a **single command**.

The default mental model must be:

> “Give the tool a model and get a readiness result.”

### 7.2 Decision-oriented output

The main result is a **readiness decision**, not merely a converted file.

### 7.3 Deployment language over converter language

The tool must communicate in deployment language, not only converter language.

Preferred concepts:

- conversion success
- runtime success
- benchmark result
- validation status
- readiness summary
- likely issue
- suggested next action

Avoid forcing the user to interpret raw Core ML internals as the first layer of output.

### 7.4 Two layers of output

There must be two layers of output:

- a concise human-readable summary
- a deeper machine-readable and expert-readable diagnostic output

---

## 8. CLI Design

There must be three user-facing modes.

### 8.1 Primary mode: `run`

A single high-level command called `run`.

This command should orchestrate the full flow:

- conversion
- diagnosis
- benchmark
- validation
- report generation

This is the main UX.

Example:

```bash
model2mobile run --model ./model.pt --task detect --input-size 640
```

### 8.2 Guided mode: `init`

A more beginner-friendly entry point called `init` (or equivalent interactive setup mode).

This mode should ask for:

- model path
- task
- input size
- device target
- whether benchmark is enabled
- whether validation is enabled

The interactive mode exists to reduce setup friction.

### 8.3 Expert mode

Advanced users should also be able to run each stage independently.

Stages should be separated conceptually into:

- convert
- benchmark
- validate
- report

These may be separate commands or subcommands, but the structure should remain consistent and unsurprising.

---

## 9. High-Level Pipeline

The default `run` pipeline must follow this sequence:

1. Load model metadata and user configuration
2. Attempt Core ML conversion
3. Normalize and classify conversion failure if conversion fails
4. If conversion succeeds, run device-side benchmark
5. Run end-to-end validation against PyTorch reference
6. Aggregate results into a deployment readiness summary
7. Write outputs to a run directory

This order matters.

Do not generate a final report before benchmark and validation are complete.

---

## 10. Readiness States

The tool must expose a small normalized readiness state.

Use exactly three top-level readiness states:

- `READY`
- `PARTIAL`
- `NOT_READY`

### 10.1 READY

Conversion succeeded, runtime succeeded, validation is acceptable, and performance is within usable bounds for the tested scenario.

### 10.2 PARTIAL

Some parts succeeded, but there are meaningful limitations.

Examples:

- conversion succeeded but parity has warnings
- runtime succeeded but postprocessing is too slow
- model runs but memory usage is too high for practical use

### 10.3 NOT_READY

The model cannot currently be treated as deployable for the tested scenario.

Examples:

- conversion failed
- runtime failed
- validation failed severely

Readiness is **scenario-specific**, not universal.

The report must state that the result depends on:

- device
- input size
- preprocessing and postprocessing pipeline
- test conditions

---

## 11. Required Outputs

Each run must produce a dedicated output directory.

The run directory must contain at least:

- a concise summary file in Markdown
- a structured metrics JSON
- a diagnosis JSON
- a validation JSON
- logs
- optional sample visualization outputs
- converted Core ML artifact if conversion succeeded

The run directory must be reviewable later without rerunning the process.

---

## 12. Summary File Requirements

The summary file is one of the most important outputs.

It must begin with a short decision-oriented summary, not raw logs.

The first section must contain:

- readiness state
- conversion result
- runtime result
- validation result
- main bottleneck
- suggested next actions

The summary should allow a user to understand the situation in under one minute.

Only after the summary should the document include deeper sections.

---

## 13. Benchmark Requirements

The benchmark system must measure at least:

- preprocess latency
- inference latency
- postprocess latency
- end-to-end latency
- estimated FPS
- memory-related observations if available

The benchmark must distinguish between these stages, because a model may appear slow even when the actual model inference is acceptable and postprocessing is the real bottleneck.

Benchmark methodology must include:

- warmup
- repeated measurements
- average or median latency
- a tail metric such as p95 if feasible

Benchmark results are device-specific and must be labeled clearly with device identity.

---

## 14. Validation Requirements

Validation must compare the original PyTorch pipeline with the converted Core ML pipeline.

This validation must include preprocessing and postprocessing, not just raw tensor shape checks.

For v1 detection, validation should check at least:

- output presence
- output tensor shape consistency where relevant
- detection count behavior
- class consistency
- confidence consistency
- bounding box consistency within reasonable tolerance

Validation does not need to prove perfect equivalence, but must be sufficient to detect cases where a model “converted successfully” but is not practically behaving the same.

The report must present validation as one of:

- `PASS`
- `WARNING`
- `FAIL`

This is separate from the top-level readiness state.

---

## 15. Failure Diagnosis Requirements

The tool must not simply dump raw errors as the main diagnostic experience.

It must normalize known failures into a controlled set of categories.

v1 categories should include at least:

- `unsupported_op`
- `dynamic_shape`
- `output_shape_mismatch`
- `runtime_failure`
- `numeric_instability`
- `postprocess_bottleneck`
- `memory_issue`
- `unknown`

Each diagnosis must include:

- raw error or raw symptom
- normalized category
- likely cause
- suggested next steps

Diagnosis may initially be rule-based.

A rule-based system is acceptable for v1.

---

## 16. Suggestion System Requirements

The tool must suggest likely next actions after diagnosing issues.

Suggestions should be simple and practical, not speculative research advice.

Example suggestion themes:

- use fixed input shape
- move NMS outside the model
- simplify output head
- reduce input size
- reduce candidate boxes before postprocess
- replace export-unfriendly block
- retest on a different compute unit setting if supported

Suggestions do not need to be automatically applied in v1.

They only need to be surfaced clearly.

---

## 17. Configuration Model

The tool must support configuration from CLI arguments and optionally from a simple config file.

Configuration must cover at least:

- model path
- task
- input size
- device selection
- output directory
- benchmark enable / disable
- validation enable / disable

Defaults should favor simplicity.

The quick-start path must require as few flags as possible.

---

## 18. Repository Structure Expectations

The repository should be structured around responsibilities, not around ad-hoc scripts.

Recommended top-level conceptual areas:

- `convert/`
- `benchmark/`
- `diagnose/`
- `validate/`
- `report/`
- `examples/`
- `outputs/`
- `docs/`

The actual folder names may vary slightly, but the architecture must clearly separate concerns.

Avoid a flat repository full of unrelated scripts.

---

## 19. Implementation Constraints

The implementation should prioritize:

- clarity
- debuggability
- explicit stage separation

Important constraints:

- Do not hide major pipeline stages inside an opaque monolith
- Keep intermediate outputs inspectable
- Keep logs stage-aware
- Make failures attributable to a stage
- Avoid overengineering plugin systems in v1
- Prefer explicitness over abstraction if there is a tradeoff

The architecture should be extendable, but v1 should not be built as a prematurely generalized framework.

---

## 20. Example User Story

A typical user story for v1 is:

A user has a PyTorch object detection model and wants to try it on a connected iPhone.  
The user does not know the Core ML conversion details.  
The user runs one command.  
The tool attempts conversion, benchmark, and validation.  
At the end, the tool tells the user whether the model is `READY`, `PARTIAL`, or `NOT_READY`, why, and what to try next.

This user story should be the primary design anchor for CLI, README, and output formatting.

---

## 21. README Expectations

The README must position the repository as a **deployment readiness tool**, not as a simple converter.

The README should answer:

- what problem this solves
- who it is for
- what “readiness” means
- how to try it in one command
- what outputs are generated
- why this is different from a conversion script

The README should not begin with deep Core ML internals.

The first impression should be:

> “This helps me evaluate whether my model can actually be deployed on iPhone.”

---

## 22. Quality Bar for v1

v1 is successful if:

- a user can run the tool with one main command
- the tool produces a readable readiness summary
- the tool separates benchmark stages
- the tool performs at least minimal parity validation
- the tool classifies common failure types
- the repository feels like a focused deployment evaluation tool, not a loose pile of scripts

v1 is **not** required to be universally compatible.

It is better to support a narrow but solid path than pretend to support many architectures poorly.

---

## 23. Phased Delivery Plan

### Phase 1
Implement the core CLI skeleton and run orchestration flow.

### Phase 2
Implement conversion and normalized diagnosis.

### Phase 3
Implement on-device benchmark.

### Phase 4
Implement end-to-end validation.

### Phase 5
Implement Markdown and JSON report generation.

### Phase 6
Polish README, example workflow, and output quality.

The coding agent should complete phases in this order.

Do not start with a broad architecture-generalization pass.

---

## 24. Acceptance Criteria

The repository will be considered acceptable for the first milestone if all of the following are true:

- there is a single main command for full evaluation
- a model can be evaluated end-to-end for the narrow supported path
- output includes readiness state
- output includes benchmark breakdown
- output includes diagnosis category
- output includes suggested next steps
- output includes validation result
- results are persisted in a run directory
- README explains the value proposition clearly

---

## 25. GUI Strategy

GUI should **not** be the primary surface for v1.

The core value of this project is:

- readiness evaluation
- diagnosis
- benchmark
- validation
- next-action guidance

These should be implemented in the CLI and reporting pipeline first.

However, a lightweight GUI layer is desirable after the core pipeline is stable, because diagnostic tools benefit from visual comparison and result browsing.

The recommended order is:

1. CLI core workflow
2. HTML report output
3. Lightweight local GUI for browsing results

---

## 26. GUI Principles

Any GUI added to this project should follow these principles:

### 26.1 The GUI is a results viewer first

The first useful GUI is **not** a complex conversion UI.  
It is a way to browse, compare, and understand run results.

### 26.2 The GUI must not replace the CLI pipeline

The CLI remains the source of truth for orchestration and automation.

The GUI should sit on top of generated run outputs, not duplicate core logic.

### 26.3 The GUI should expose decision-oriented information

The GUI should foreground:

- readiness state
- main bottleneck
- benchmark breakdown
- validation status
- suggested next actions

It should not foreground raw backend internals as the first screen.

### 26.4 The GUI should support comparison

The GUI becomes valuable when users can compare:

- run A vs run B
- device A vs device B
- input size A vs input size B
- model version A vs model version B

---

## 27. Proposed GUI Surfaces

The recommended initial GUI scope is three surfaces.

### 27.1 Run Summary View

Purpose: show the latest run result in a highly readable way.

Should include:

- readiness badge (`READY`, `PARTIAL`, `NOT_READY`)
- conversion status
- runtime status
- validation status
- main bottleneck
- suggested next steps
- key benchmark numbers

### 27.2 Benchmark Breakdown View

Purpose: visualize where time is spent.

Should include:

- preprocess latency
- inference latency
- postprocess latency
- end-to-end latency
- estimated FPS
- device identity
- run conditions

A bar chart or stacked visualization is appropriate.

### 27.3 Run Comparison View

Purpose: compare two or more runs.

Should include comparisons by:

- device
- input size
- model version
- readiness state
- latency breakdown
- validation result

This is especially useful for iterative adaptation work.

---

## 28. GUI Implementation Recommendation

The recommended path is:

### Stage 1: HTML report
Generate a self-contained HTML report per run.

Benefits:

- no app framework overhead
- easy to share
- easy to inspect in browser
- aligns with existing CLI output model

### Stage 2: Local result browser
A lightweight local GUI can be added later to browse multiple run directories.

This could be implemented as:

- a small local web app
- a lightweight desktop shell
- any simple viewer layer

The implementation choice is less important than keeping it thin and report-driven.

Do not build a heavy GUI-first architecture in v1.

---

## 29. HTML Report Requirement

Even if no full GUI is implemented in v1, the reporting system should be designed so that HTML output can be added easily.

The HTML report should be able to display:

- summary status
- benchmark charts
- validation summary
- diagnosis category
- suggested actions
- metadata about device and configuration

This HTML report is the preferred first “visual layer” of the project.

---

## 30. Acceptance Criteria for Initial GUI Layer

The initial GUI layer can be considered acceptable if:

- it reads existing run outputs rather than duplicating pipeline logic
- it clearly shows readiness state
- it clearly shows benchmark stage breakdown
- it clearly shows validation result
- it clearly shows diagnosis and suggested next actions
- it improves result readability without becoming a blocker for the CLI-first workflow

---

## 31. Tone and Product Framing

The tone of the project must be practical and diagnostic.

This project should feel like:

- a model health check
- a deployment readiness bench
- a mobile adaptation evaluator

It should not feel like:

- a toy converter
- a generic script collection
- a broad model zoo
- a vague future framework

---

## 32. Suggested Tagline

**Evaluate whether your PyTorch vision model is actually ready for on-device deployment.**

---

## 33. Suggested Quick Start Example

```bash
model2mobile run \
  --model ./model.pt \
  --task detect \
  --input-size 640
```

Expected result:

- conversion attempt
- normalized diagnosis
- on-device benchmark
- validation result
- readiness summary
- output directory with reports and artifacts
