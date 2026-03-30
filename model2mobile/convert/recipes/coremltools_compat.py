"""Recipe: Patch coremltools op converters for tricky traced ops.

Some models (especially DINOv2-based architectures) produce traced ops that
trigger bugs in coremltools' PyTorch frontend.  This recipe patches the
affected coremltools functions at runtime:

1. **_cast (int op)** — ``int()`` on a multi-element const array fails with
   "only 0-dimensional arrays can be converted to Python scalars".
   Fix: use ``.item()`` before casting.
2. **torch_tensor_assign** — In-place copy with shape ``(1,)`` where ``[]``
   is expected.  Fix: relax the type-inference check.
3. **meshgrid** — ``torch.meshgrid`` inputs may be non-1D after tracing
   through ``torch.linspace`` with traced step counts.
   Fix: flatten inputs before building the grid.
4. **split / split_with_sizes** — Split sizes arrive as a Python list of
   ``Var`` objects instead of a single array ``Var``.
   Fix: extract concrete values from the list and build an ``np.ndarray``.
"""

from __future__ import annotations

import logging

import numpy as np
import torch.nn as nn

from model2mobile.convert.recipes.base import Recipe, RecipeResult

logger = logging.getLogger(__name__)


def _patch_cast() -> bool:
    try:
        import coremltools.converters.mil.frontend.torch.ops as co
        from coremltools.converters.mil.frontend.torch.ops import _get_inputs
        from coremltools.converters.mil import Builder as mb
    except ImportError:
        return False

    def _fixed_cast(context, node, dtype, dtype_name):
        x = _get_inputs(context, node, expected=1)[0]
        if x.can_be_folded_to_const():
            val = x.val
            if hasattr(val, "item"):
                val = dtype(val.item())
            elif not isinstance(val, dtype):
                val = dtype(val)
            context.add(mb.const(val=val, name=node.name), node.name)
        elif len(x.shape) > 0:
            sq = mb.squeeze(x=x, name=node.name + "_sq")
            context.add(mb.cast(x=sq, dtype=dtype_name, name=node.name), node.name)
        else:
            context.add(mb.cast(x=x, dtype=dtype_name, name=node.name), node.name)

    co._cast = _fixed_cast
    return True


def _patch_tensor_assign() -> bool:
    try:
        import coremltools.converters.mil.frontend.torch.dialect_ops as cd
        cd.torch_tensor_assign.type_inference = lambda self: self.x.sym_type
        return True
    except (ImportError, AttributeError):
        return False


def _patch_meshgrid() -> bool:
    try:
        from coremltools.converters.mil.frontend.torch.ops import _get_inputs, _get_kwinputs
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import Var
        from coremltools.converters.mil.frontend import _utils
        import coremltools.converters.mil.frontend.torch.torch_op_registry as reg
    except ImportError:
        return False

    def _fixed_meshgrid(context, node):
        inputs = _get_inputs(context, node)
        if isinstance(inputs[0], (list, tuple)):
            ti = list(inputs[0])
            ix = inputs[1].val if len(inputs) > 1 else "ij"
        else:
            ti = [i for i in inputs if isinstance(i, Var)]
            ix = "ij"
        ik = _get_kwinputs(context, node, "indexing", default=[ix])
        ix = ik[0] if isinstance(ik, (list, tuple)) else ik

        # Flatten non-1D inputs
        ti = [
            mb.reshape(x=t, shape=[-1], name=node.name + f"_f{j}") if t.rank > 1 else t
            for j, t in enumerate(ti)
        ]
        rs = _utils.maybe_replace_symbols_with_source_tensor_shape_variables(
            [t.shape[0] for t in ti], ti
        )
        grids = []
        for i in range(len(ti)):
            vs = [1] * len(ti)
            vs[i] = -1
            v = mb.reshape(x=ti[i], shape=tuple(vs), name=node.name + f"_v{i}")
            rp = rs.copy()
            rp[i] = 1
            if any(isinstance(r, Var) for r in rp):
                rp = mb.concat(values=rp, axis=0)
            r = mb.tile(x=v, reps=rp, name=node.name + f"_e{i}")
            if ix == "xy":
                r = mb.transpose(
                    x=r, perm=[1, 0] + list(range(2, len(ti))),
                    name=node.name + f"_t{i}",
                )
            grids.append(r)
        context.add(tuple(grids), node.name)

    reg._TORCH_OPS_REGISTRY.set_func_by_name(_fixed_meshgrid, "meshgrid")
    return True


def _patch_split() -> bool:
    try:
        from coremltools.converters.mil.frontend.torch.ops import _get_inputs, _get_kwinputs
        from coremltools.converters.mil import Builder as mb
        from coremltools.converters.mil.mil import Var
        import coremltools.converters.mil.frontend.torch.torch_op_registry as reg
    except ImportError:
        return False

    def _fixed_split(context, node):
        inputs = _get_inputs(context, node, min_expected=2)
        x, ss = inputs[0], inputs[1]
        dim = inputs[2] if len(inputs) > 2 else 0
        dk = _get_kwinputs(context, node, "dim", default=[dim])
        dim = dk[0] if isinstance(dk, (list, tuple)) else dk
        if isinstance(dim, Var):
            dim = dim.val

        if isinstance(ss, (list, tuple)):
            vals = []
            for s in ss:
                if isinstance(s, Var) and s.val is not None:
                    vals.append(int(s.val))
                elif not isinstance(s, Var):
                    vals.append(int(s))
                else:
                    vals.append(None)
            if all(v is not None for v in vals):
                res = mb.split(
                    x=x, split_sizes=np.array(vals, dtype=np.int32),
                    axis=dim, name=node.name,
                )
            else:
                parts = []
                for j, s in enumerate(ss):
                    if isinstance(s, Var):
                        parts.append(
                            mb.expand_dims(x=s, axes=[0], name=node.name + f"_d{j}")
                            if s.rank == 0 else s
                        )
                    else:
                        parts.append(mb.const(val=np.array([int(s)], dtype=np.int32)))
                res = mb.split(
                    x=x,
                    split_sizes=mb.concat(values=parts, axis=0, name=node.name + "_ss"),
                    axis=dim, name=node.name,
                )
        elif isinstance(ss, Var) and ss.val is not None and isinstance(ss.val, np.ndarray):
            res = mb.split(x=x, split_sizes=ss, axis=dim, name=node.name)
        else:
            sv = int(ss.val) if isinstance(ss, Var) else int(ss)
            total = x.shape[dim]
            szs = [sv] * (total // sv) + ([total % sv] if total % sv else [])
            res = mb.split(
                x=x, split_sizes=np.array(szs, dtype=np.int32),
                axis=dim, name=node.name,
            )
        context.add(res, torch_name=node.name)

    for name in ("split", "split_with_sizes", "split_with_sizes_copy"):
        reg._TORCH_OPS_REGISTRY.set_func_by_name(_fixed_split, name)
    return True


class CoreMLToolsCompatRecipe(Recipe):
    name = "coremltools_compat"

    def match(self, model: nn.Module, architecture: str, error: str | None = None) -> bool:
        # Error-triggered: match specific coremltools conversion errors
        if error:
            err_lower = error.lower()
            triggers = [
                "0-dimensional arrays",
                "updates tensor should have shape",
                "meshgrid received non-1d",
                "must be const",
                "'list' object has no attribute 'val'",
            ]
            if any(t in err_lower for t in triggers):
                return True

        # Pre-emptive: apply for DINOv2-based and D-FINE models since these
        # coremltools bugs are triggered by their traced graphs.
        arch_lower = architecture.lower()
        if any(tag in arch_lower for tag in ("rfdetr", "rf-detr", "rf_detr", "dinov2", "dfine", "d-fine", "d_fine")):
            return True

        # Check for DINOv2 / deformable attention / D-FINE modules
        for m in model.modules():
            cls_name = type(m).__name__
            if any(tag in cls_name for tag in ("Dinov2", "DINOv2", "MSDeformAttn", "DFINEDecoder", "HybridEncoder")):
                return True

        return False

    def apply(self, model: nn.Module) -> RecipeResult:
        mods: list[str] = []
        if _patch_cast():
            mods.append("Patched _cast: handle multi-element const → scalar")
        if _patch_tensor_assign():
            mods.append("Patched tensor_assign: relaxed shape check")
        if _patch_meshgrid():
            mods.append("Patched meshgrid: flatten non-1D inputs")
        if _patch_split():
            mods.append("Patched split: handle list-of-Var split sizes")
        return RecipeResult(
            applied=len(mods) > 0,
            recipe_name=self.name,
            description=f"Patched {len(mods)} coremltools op converters",
            modifications=mods,
        )
