"""Benchmark module."""

from model2mobile.benchmark.runner import run_benchmark

__all__ = ["run_benchmark", "list_devices", "run_device_benchmark"]


def list_devices():
    from model2mobile.benchmark.device import list_devices as _list_devices
    return _list_devices()


def run_device_benchmark(coreml_path, config):
    from model2mobile.benchmark.device import run_device_benchmark as _run_device_benchmark
    return _run_device_benchmark(coreml_path, config)
