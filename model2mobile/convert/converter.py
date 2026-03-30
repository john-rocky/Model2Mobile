"""Core ML conversion from PyTorch models."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
import traceback
import warnings
from collections import Counter
from pathlib import Path

import coremltools as ct
import torch
import torch.nn as nn
from rich.console import Console

from model2mobile.config import RunConfig
from model2mobile.models import ConversionResult, ModelInfo

logger = logging.getLogger(__name__)
console = Console()

_COMPUTE_UNIT_MAP: dict[str, ct.ComputeUnit] = {
    "ALL": ct.ComputeUnit.ALL,
    "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
    "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
}

_MAX_INSTALL_RETRIES = 10


def _pip_install(package: str) -> bool:
    console.print(f"  [yellow]Installing missing package: {package}[/yellow]")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package, "-q"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        console.print(f"  [green]Installed {package}[/green]")
        return True
    except subprocess.CalledProcessError:
        console.print(f"  [red]Failed to install {package}[/red]")
        return False


def _try_load_rfdetr(model_path: Path) -> nn.Module | None:
    """Try to load an RF-DETR model via the rfdetr or peaceofcake package.

    RF-DETR models are checkpoint files that require the rfdetr package to
    construct the architecture before loading weights.  A plain torch.load()
    will only produce a state_dict, not a runnable model.
    """
    stem = model_path.stem.lower()
    if "rfdetr" not in stem and "rf-detr" not in stem and "rf_detr" not in stem:
        return None

    # Try peaceofcake first (unified API)
    try:
        from peaceofcake import RFDETR
        model_wrapper = RFDETR(str(model_path))
        model = model_wrapper.model
        model.cpu().eval()
        model.export()
        logger.info("Loaded RF-DETR via peaceofcake")
        return model
    except Exception:
        pass

    # Try rfdetr package directly
    try:
        from rfdetr import RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge
        size_map = {"nano": RFDETRNano, "small": RFDETRSmall, "medium": RFDETRMedium, "large": RFDETRLarge}
        for name, cls in size_map.items():
            if name[0] in stem or name in stem:
                obj = cls(pretrain_weights=str(model_path))
                model = obj.model.model
                model.cpu().eval()
                model.export()
                logger.info("Loaded RF-DETR-%s via rfdetr", name)
                return model
    except Exception:
        pass

    return None


def _try_load_dfine(model_path: Path) -> nn.Module | None:
    """Try to load a D-FINE model via peaceofcake.

    Returns a ready-to-trace wrapper that outputs flat tensors
    (confidence, coordinates) — the same format as the peaceofcake exporter.
    """
    stem = model_path.stem.lower()
    if "dfine" not in stem:
        return None

    try:
        from peaceofcake import DFINE
        model_wrapper = DFINE(str(model_path))
        # Use peaceofcake's own exporter setup for proper model preparation
        model_wrapper.model.cpu().eval()
        exporter_cls = model_wrapper.task_map["detect"]["exporter"]
        exporter = exporter_cls(model_wrapper, {})
        # Get the deploy-ready model + postprocessor via the exporter's helper
        model, postprocessor = exporter._get_model_and_postprocessor()

        import torch.nn.functional as F

        class _Wrapper(nn.Module):
            def __init__(self, m, pp):
                super().__init__()
                self.model = m.deploy()
                pp.deploy()
                self.use_focal_loss = pp.use_focal_loss

            def forward(self, images):
                outputs = self.model(images)
                logits = outputs["pred_logits"]
                boxes = outputs["pred_boxes"]
                if self.use_focal_loss:
                    confidence = F.sigmoid(logits)
                else:
                    confidence = F.softmax(logits, dim=-1)[:, :, :-1]
                return confidence.squeeze(0), boxes.squeeze(0)

        wrapper = _Wrapper(model, postprocessor).eval()

        # Fix project tensor for CoreML linear op
        decoder = wrapper.model.decoder.decoder
        if hasattr(decoder, "project") and decoder.project.dim() == 1:
            decoder.project = nn.Parameter(decoder.project.unsqueeze(0), requires_grad=False)

        logger.info("Loaded D-FINE via peaceofcake (deploy + wrapped)")
        return wrapper
    except Exception:
        return None


def _load_model(model_path: Path) -> nn.Module | torch.jit.ScriptModule:
    suffix = model_path.suffix.lower()

    # Explicit TorchScript extensions
    if suffix in (".torchscript", ".ts"):
        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
        return model

    # Try TorchScript first (many .pt files are actually TorchScript)
    try:
        model = torch.jit.load(str(model_path), map_location="cpu")
        model.eval()
        return model
    except Exception:
        pass

    # Try specialised loaders for models that need their package to construct
    # the architecture (RF-DETR needs rfdetr, D-FINE needs peaceofcake).
    for loader in (_try_load_rfdetr, _try_load_dfine):
        model = loader(model_path)
        if model is not None:
            return model

    # Pickle-based loading with auto-install of missing packages.
    # Models downloaded from the internet (e.g. YOLO, timm) are often saved
    # with torch.save(model) which pickles the class definition.  The class
    # won't unpickle unless the originating package is installed.
    for _ in range(_MAX_INSTALL_RETRIES):
        try:
            loaded = torch.load(str(model_path), map_location="cpu", weights_only=False)
            break
        except ModuleNotFoundError as exc:
            pkg = exc.name.split(".")[0] if exc.name else None
            if not pkg or not _pip_install(pkg):
                raise
    else:
        raise RuntimeError("Too many missing packages, giving up")

    if isinstance(loaded, nn.Module):
        loaded.eval()
        return loaded

    if isinstance(loaded, dict):
        if "model" in loaded and isinstance(loaded["model"], nn.Module):
            loaded["model"].eval()
            return loaded["model"]
        # Some checkpoints nest the model under 'ema' or other keys
        for key in ("ema", "net", "network"):
            if key in loaded and isinstance(loaded[key], nn.Module):
                loaded[key].eval()
                return loaded[key]
        raise ValueError(
            "File contains a state_dict / checkpoint, not a runnable model.\n"
            "Save the full model: torch.save(model, 'model.pt')"
        )

    raise ValueError(f"Unexpected type in file: {type(loaded).__name__}")


# ---------------------------------------------------------------------------
# Model analysis
# ---------------------------------------------------------------------------


def _count_parameters(model: nn.Module | torch.jit.ScriptModule) -> int:
    try:
        return sum(p.numel() for p in model.parameters())
    except Exception:
        return 0


def _estimate_size_mb(model: nn.Module | torch.jit.ScriptModule) -> float:
    try:
        total_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
        return total_bytes / (1024 * 1024)
    except Exception:
        return 0.0


def _detect_dynamic_shapes(model: nn.Module | torch.jit.ScriptModule) -> bool:
    if isinstance(model, torch.jit.ScriptModule):
        try:
            graph_str = str(model.graph)
            if "Dynamic" in graph_str or "aten::size" in graph_str:
                return True
        except Exception:
            pass
    return False


def _summarize_ops(model: nn.Module | torch.jit.ScriptModule) -> dict[str, int]:
    counter: Counter[str] = Counter()
    if isinstance(model, torch.jit.ScriptModule):
        try:
            for node in model.graph.nodes():
                counter[node.kind()] += 1
        except Exception:
            pass
    else:
        for module in model.modules():
            counter[type(module).__name__] += 1
    return dict(counter.most_common())


def _detect_architecture(model: nn.Module | torch.jit.ScriptModule) -> str:
    cls_name = type(model).__name__
    if cls_name in ("RecursiveScriptModule", "ScriptModule"):
        try:
            return model.original_name  # type: ignore[attr-defined]
        except AttributeError:
            return "TorchScript"
    return cls_name


def _infer_input_size(model: nn.Module | torch.jit.ScriptModule) -> int | None:
    """Try to guess the expected input size from the model structure."""
    # Walk modules looking for the first Conv2d to estimate scale,
    # but the real signal is common detection sizes baked into the arch.
    # Many models embed input size in attributes or config dicts.
    for attr in ("img_size", "input_size", "imgsz", "image_size"):
        val = getattr(model, attr, None)
        if isinstance(val, int) and val > 0:
            return val
        if isinstance(val, (list, tuple)) and len(val) > 0:
            return int(val[-1])

    # For TorchScript, try a forward pass with a few common sizes and
    # see which one doesn't error.  Skip if model is large (slow).
    if isinstance(model, torch.jit.ScriptModule):
        return None

    # Heuristic: try a small forward pass to see if model constrains size
    for candidate in (640, 512, 416, 320, 224):
        try:
            dummy = torch.zeros(1, 3, candidate, candidate)
            with torch.no_grad():
                model(dummy)
            return candidate
        except Exception:
            continue
    return None


def _analyze_model(
    model: nn.Module | torch.jit.ScriptModule,
    model_path: str,
    input_shape: tuple[int, ...],
) -> ModelInfo:
    return ModelInfo(
        path=model_path,
        parameter_count=_count_parameters(model),
        input_shape=input_shape,
        estimated_size_mb=_estimate_size_mb(model),
        architecture=_detect_architecture(model),
        has_dynamic_shapes=_detect_dynamic_shapes(model),
        op_summary=_summarize_ops(model),
    )


# ---------------------------------------------------------------------------
# Task inference
# ---------------------------------------------------------------------------


def _infer_task(model: nn.Module | torch.jit.ScriptModule, input_shape: tuple[int, ...]) -> str:
    """Guess the task from model output shape.

    Logic:
    - (N, num_classes) with ndim==2 -> "classify"
    - (N, C, H, W) where H,W are close to input H,W -> "segment"
    - (N, 1, H, W) -> "depth"
    - (N, boxes, 5+) or multiple outputs -> "detect"
    - Default: "classify"
    """
    try:
        dummy = torch.randn(*input_shape)
        with torch.no_grad():
            raw = model(dummy)
    except Exception:
        return "classify"

    # Normalize output to a list of tensors
    tensors: list[torch.Tensor] = []
    if isinstance(raw, torch.Tensor):
        tensors = [raw]
    elif isinstance(raw, (tuple, list)):
        for item in raw:
            if isinstance(item, torch.Tensor):
                tensors.append(item)
            elif isinstance(item, dict):
                for v in item.values():
                    if isinstance(v, torch.Tensor):
                        tensors.append(v)
    elif isinstance(raw, dict):
        for v in raw.values():
            if isinstance(v, torch.Tensor):
                tensors.append(v)

    if not tensors:
        return "classify"

    # Multiple outputs often indicate detection
    if len(tensors) > 2:
        return "detect"

    input_h, input_w = input_shape[2], input_shape[3]

    for t in tensors:
        shape = t.shape

        # 2-D output: (N, num_classes) -> classification
        if t.ndim == 2 and shape[1] > 1:
            return "classify"

        # 4-D output: check spatial dims
        if t.ndim == 4:
            n, c, h, w = shape
            # (N, 1, H, W) -> depth estimation
            if c == 1:
                return "depth"
            # (N, C, H, W) where H,W are close to input -> segmentation
            h_ratio = h / input_h
            w_ratio = w / input_w
            if 0.25 <= h_ratio <= 1.5 and 0.25 <= w_ratio <= 1.5:
                return "segment"

        # 3-D output: (N, boxes, 5+) -> detection
        if t.ndim == 3 and shape[2] >= 5:
            return "detect"

    return "classify"


# ---------------------------------------------------------------------------
# Internal: trace + convert + save
# ---------------------------------------------------------------------------


def _trace_and_convert(
    model: nn.Module | torch.jit.ScriptModule,
    input_shape: tuple[int, ...],
    compute_unit_enum: ct.ComputeUnit,
) -> ct.models.MLModel:
    dummy_input = torch.randn(*input_shape)
    if isinstance(model, torch.jit.ScriptModule):
        traced = model
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            traced = torch.jit.trace(model, dummy_input)
    return ct.convert(
        traced,
        inputs=[ct.TensorType(shape=input_shape)],
        compute_units=compute_unit_enum,
    )


def _save_mlpackage(
    coreml_model: ct.models.MLModel,
    model_path: Path,
    output_dir: Path,
) -> tuple[Path, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    mlpackage_path = output_dir / (model_path.stem + ".mlpackage")
    coreml_model.save(str(mlpackage_path))
    coreml_size_mb = 0.0
    if mlpackage_path.exists():
        total = sum(f.stat().st_size for f in mlpackage_path.rglob("*") if f.is_file())
        coreml_size_mb = total / (1024 * 1024)
    return mlpackage_path, coreml_size_mb


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def convert_model(
    config: RunConfig, output_dir: Path
) -> tuple[ModelInfo, ConversionResult]:
    from model2mobile.convert.recipes.registry import apply_recipes

    model_path = Path(config.model_path)

    # --- Load ---
    try:
        model = _load_model(model_path)
    except Exception as exc:
        input_shape = (1, 3, config.input_size, config.input_size)
        info = ModelInfo(path=str(model_path), input_shape=input_shape)
        return info, ConversionResult(
            success=False,
            compute_unit=config.compute_unit,
            error_message=f"Failed to load model: {exc}",
            raw_error=traceback.format_exc(),
        )

    # --- Auto-detect input size if not explicitly set ---
    input_size = config.input_size
    if config.input_size_auto:
        detected = _infer_input_size(model)
        if detected is not None:
            input_size = detected
            console.print(f"  [cyan]Auto-detected input size: {input_size}[/cyan]")
    input_shape = (1, 3, input_size, input_size)

    # --- Infer task if set to "auto" ---
    if config.task == "auto":
        detected_task = _infer_task(model, input_shape)
        config.task = detected_task
        console.print(f"  [cyan]Auto-detected task: {detected_task}[/cyan]")

    # --- Analyze ---
    architecture = _detect_architecture(model)
    info = _analyze_model(model, str(model_path), input_shape)
    info.task = config.task
    logger.info(
        "Model: %s | params=%d | size=%.1f MB",
        info.architecture, info.parameter_count, info.estimated_size_mb,
    )

    compute_unit_enum = _COMPUTE_UNIT_MAP.get(
        config.compute_unit.upper(), ct.ComputeUnit.ALL
    )

    # --- Pre-emptive recipes: apply known fixes BEFORE first attempt ---
    pre_recipes = apply_recipes(model, architecture, error=None)
    if pre_recipes:
        names = [r.recipe_name for r in pre_recipes]
        console.print(f"  [cyan]Applied recipes: {', '.join(names)}[/cyan]")

    # --- Attempt conversion with retry loop ---
    # Some models (e.g. DINOv2-based RF-DETR) trigger multiple coremltools
    # bugs sequentially.  Each retry applies new error-triggered recipes.
    _MAX_RECIPE_RETRIES = 3
    applied_recipes = list(pre_recipes)
    total_time = 0.0
    last_error = ""
    last_traceback = ""
    coreml_model = None

    for attempt in range(_MAX_RECIPE_RETRIES + 1):
        start = time.perf_counter()
        try:
            coreml_model = _trace_and_convert(model, input_shape, compute_unit_enum)
            total_time += time.perf_counter() - start
            break
        except Exception as exc:
            total_time += time.perf_counter() - start
            last_error = str(exc)
            last_traceback = traceback.format_exc()

            if attempt >= _MAX_RECIPE_RETRIES:
                break

            # Apply error-triggered recipes for the new error
            retry_recipes = apply_recipes(model, architecture, error=last_error)
            if not retry_recipes:
                break

            applied_recipes.extend(retry_recipes)
            names = [r.recipe_name for r in retry_recipes]
            console.print(
                f"  [yellow]Attempt {attempt + 1} failed, retrying with: {', '.join(names)}[/yellow]"
            )

    if coreml_model is None:
        return info, ConversionResult(
            success=False,
            compute_unit=config.compute_unit,
            conversion_time_s=round(total_time, 3),
            error_message=f"Core ML conversion failed: {last_error}",
            raw_error=last_traceback,
            warnings=[f"Tried {len(applied_recipes)} recipes"],
        )

    conversion_time = total_time

    # --- Save ---
    try:
        mlpackage_path, coreml_size_mb = _save_mlpackage(coreml_model, model_path, output_dir)
    except Exception as exc:
        return info, ConversionResult(
            success=False,
            compute_unit=config.compute_unit,
            conversion_time_s=round(conversion_time, 3),
            error_message=f"Failed to save mlpackage: {exc}",
            raw_error=traceback.format_exc(),
        )

    warnings = [f"Recipe applied: {r.recipe_name} — {r.description}" for r in applied_recipes]

    logger.info(
        "Conversion succeeded in %.1fs -> %s (%.1f MB)",
        conversion_time, mlpackage_path, coreml_size_mb,
    )

    return info, ConversionResult(
        success=True,
        coreml_path=str(mlpackage_path),
        coreml_size_mb=round(coreml_size_mb, 2),
        compute_unit=config.compute_unit,
        conversion_time_s=round(conversion_time, 3),
        warnings=warnings,
    )
