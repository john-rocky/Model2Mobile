#!/bin/bash
# Example: Full deployment readiness evaluation

# Basic run
model2mobile run --model ./model.pt --task detect --input-size 640

# With compute unit comparison
model2mobile run --model ./model.pt --task detect --input-size 640 --compare-units

# CPU-only benchmark with more iterations
model2mobile run --model ./model.pt --task detect --input-size 320 \
  --compute-unit CPU_ONLY \
  --warmup 10 \
  --iterations 50

# Convert only (expert mode)
model2mobile convert --model ./model.pt --input-size 640

# Benchmark an existing Core ML model
model2mobile benchmark --coreml ./outputs/convert/model.mlpackage --input-size 640

# Interactive setup
model2mobile init
