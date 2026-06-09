"""Build fully precomputed static assets for the PINNfluence web demo.

The output is deliberately simple: JSON manifests plus typed-array binary files.
The browser can render everything without importing Python, Torch, DeepXDE, or
loading full influence matrices.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import deepxde as dde
import numpy as np
import torch

from pinnfluence import problem_factory
from pinnfluence.utils.models import ModelWrapper
from pinnfluence.utils.utils import loss_term_names

CORE_MATRIX_IDS = {
    "influences_total_loss_output_0",
    "influences_total_loss_total_loss",
    "grad_dot_total_loss_output_0",
    "grad_dot_total_loss_total_loss",
}

DTYPES = {
    "float32": np.float32,
    "uint32": np.uint32,
    "uint16": np.uint16,
    "uint8": np.uint8,
    "int16": np.int16,
}
DTYPE_EXTENSIONS = {
    "float32": "f32",
    "uint32": "u32",
    "uint16": "u16",
    "uint8": "u8",
    "int16": "i16",
}

SCHEMA_VERSION = 5
DEFAULT_RASTER_MAX_RESOLUTION = 512
DEFAULT_MAX_LOCAL_INFLUENCE_POINTS = 64
DEFAULT_ROW_CHUNK_SIZE = 256
BUNDLE_SIZE_BUDGET_BYTES = 750 * 1024 * 1024
DEFAULT_MATRIX_ID = "influences_total_loss_total_loss"


@dataclass
class RunPaths:
    folder: Path
    problem: str
    run_prefix: str
    checkpoint: Path | None
    influence_dir: Path | None
    validation_dir: Path | None


@dataclass
class RasterGrid:
    points: np.ndarray
    mask: np.ndarray
    width: int
    height: int
    bounds: dict[str, list[float]]
    axes: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build static, fully precomputed web demo artifacts."
    )
    parser.add_argument(
        "--problems",
        nargs="*",
        default=None,
        help="Problem folders to process, e.g. allen_cahn_float64 burgers_float64. Default: all.",
    )
    parser.add_argument(
        "--max_local_influence_points",
        default=64,
        type=int,
        help="Number Calculated Train-Influences per Candidate Point",
    )
    parser.add_argument("--row-chunk-size", default=256, type=int)  # NOTE:
    parser.add_argument("--bundle-size-budget-mb", default=750, type=int)  # NOTE:
    parser.add_argument(
        "--raster-max-resolution",
        default=512,
        type=int,
        help="Maximum pixel resolution of the longer axis for 2D prediction/loss rasters.",
    )
    parser.add_argument(
        "--matrix-mode",
        choices=["core", "all"],
        default="core",
        help="core keeps the public bundle compact; all exports every influence matrix.",
    )
    parser.add_argument(
        "--workers",
        default=min(4, os.cpu_count() or 1),
        type=int,
        help="Number of worker processes for influence matrix export. Use 1 for serial.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--skip-fields",
        action="store_true",
        help="Only build points and influence top-k data; skip prediction/loss fields.",
    )
    parser.add_argument(
        "--skip-incomplete",
        action="store_true",
        help="Do not include incomplete runs in webdemo/public/data/index.json.",
    )
    parser.add_argument(
        "--force-run",
        default=None,
        help="Only process one run as '<folder>/<run_prefix>'.",
    )
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_array(
    base_dir: Path,
    rel_path: str,
    array: np.ndarray,
    dtype: str,
) -> dict[str, Any]:
    path = base_dir / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(array, dtype=DTYPES[dtype])
    with path.open("wb") as handle:
        arr.tofile(handle)
    return {
        "path": rel_path,
        "dtype": dtype,
        "shape": list(arr.shape),
        "bytes": path.stat().st_size,
    }


def robust_display_domain(values: np.ndarray, mask: np.ndarray | None = None) -> list[float]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if mask is not None:
        values = values[np.asarray(mask, dtype=bool).reshape(-1)]
    values = values[np.isfinite(values)]
    if values.size == 0:
        return [0.0, 1.0]
    low, high = np.quantile(values.astype(np.float64), [0.02, 0.98])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        center = float(values[0]) if values.size else 0.0
        low = center - 1.0
        high = center + 1.0
    return [float(low), float(high)]


def quantize_uint16_linear(
    values: np.ndarray,
    mask: np.ndarray | None = None,
    missing: int = 65535,
) -> tuple[np.ndarray, dict[str, Any], list[float]]:
    values = np.asarray(values, dtype=np.float32)
    flat_values = values.reshape(-1)
    valid = np.isfinite(flat_values)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool).reshape(-1)
    if np.any(valid):
        vmin = float(np.nanmin(flat_values[valid]))
        vmax = float(np.nanmax(flat_values[valid]))
    else:
        vmin = 0.0
        vmax = 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmax = vmin + 1.0
    span = vmax - vmin
    quantized = np.full(flat_values.shape, missing, dtype=np.uint16)
    scaled = np.rint(np.clip((flat_values[valid] - vmin) / span, 0.0, 1.0) * 65534.0)
    quantized[valid] = scaled.astype(np.uint16)
    encoding = {"kind": "linear", "min": vmin, "max": vmax, "missing": missing}
    return quantized.reshape(values.shape), encoding, robust_display_domain(values, mask)


def quantize_int16_symmetric(values: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.astype(np.int16), 1.0
    max_abs = float(np.nanmax(np.abs(values)))
    scale = max_abs / 32767.0 if np.isfinite(max_abs) and max_abs > 0 else 1.0
    quantized = np.rint(values / scale)
    quantized = np.clip(quantized, -32767, 32767).astype(np.int16)
    return quantized, float(scale)


def bundle_file_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        if path.name == "manifest.json":
            return "manifest"
        if path.name == "index.json":
            return "index"
        return "json"
    if suffix in {".f32", ".u32", ".u16", ".u8", ".i16"}:
        parts = set(path.parts)
        if "chunks" in parts:
            return "influence_chunks"
        if path.name.startswith("summary_"):
            return "global_summaries"
        if "raster" in path.stem:
            return "field_rasters"
        return "arrays"
    return suffix.removeprefix(".") or "other"


def build_bundle_report(root: Path, budget_bytes: int = BUNDLE_SIZE_BUDGET_BYTES) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    by_suffix: dict[str, int] = {}
    files: list[dict[str, Any]] = []
    total = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        size = path.stat().st_size
        rel = path.relative_to(root).as_posix()
        kind = bundle_file_kind(path)
        suffix = path.suffix.lower() or "<none>"
        by_kind[kind] = by_kind.get(kind, 0) + size
        by_suffix[suffix] = by_suffix.get(suffix, 0) + size
        files.append({"path": rel, "bytes": size, "kind": kind})
        total += size
    return {
        "schema_version": SCHEMA_VERSION,
        "root": ".",
        "total_bytes": total,
        "budget_bytes": budget_bytes,
        "within_budget": total <= budget_bytes,
        "by_kind": dict(sorted(by_kind.items())),
        "by_suffix": dict(sorted(by_suffix.items())),
        "files": files,
    }


def slug_path(path: Path) -> str:
    return path.as_posix()


def dtype_extension(dtype: str) -> str:
    return DTYPE_EXTENSIONS[dtype]


def problem_from_folder(folder: Path) -> str:
    name = folder.name
    return name.removesuffix("_float64")


def discover_runs(data_root: Path) -> list[RunPaths]:
    runs: dict[tuple[str, str], RunPaths] = {}
    for folder in sorted(data_root.glob("*_float64")):
        if not folder.is_dir():
            continue
        problem = problem_from_folder(folder)

        for checkpoint in sorted(folder.glob("*_full.pt")):
            run_prefix = checkpoint.name.removesuffix("_full.pt")
            key = (folder.name, run_prefix)
            runs[key] = RunPaths(
                folder=folder,
                problem=problem,
                run_prefix=run_prefix,
                checkpoint=checkpoint,
                influence_dir=folder / f"{run_prefix}_influence_scores",
                validation_dir=folder / f"{run_prefix}_validation",
            )

        for influence_dir in sorted(folder.glob("*_influence_scores")):
            if not influence_dir.is_dir():
                continue
            run_prefix = influence_dir.name.removesuffix("_influence_scores")
            key = (folder.name, run_prefix)
            if key not in runs:
                runs[key] = RunPaths(
                    folder=folder,
                    problem=problem,
                    run_prefix=run_prefix,
                    checkpoint=folder / f"{run_prefix}_full.pt",
                    influence_dir=influence_dir,
                    validation_dir=folder / f"{run_prefix}_validation",
                )

    normalized = []
    for run in runs.values():
        checkpoint = run.checkpoint if run.checkpoint and run.checkpoint.exists() else None
        influence_dir = (
            run.influence_dir if run.influence_dir and run.influence_dir.exists() else None
        )
        validation_dir = (
            run.validation_dir if run.validation_dir and run.validation_dir.exists() else None
        )
        normalized.append(
            RunPaths(
                folder=run.folder,
                problem=run.problem,
                run_prefix=run.run_prefix,
                checkpoint=checkpoint,
                influence_dir=influence_dir,
                validation_dir=validation_dir,
            )
        )
    return sorted(normalized, key=lambda r: (r.folder.name, r.run_prefix))


def filter_runs(runs: list[RunPaths], args: argparse.Namespace) -> list[RunPaths]:
    selected = runs
    if args.problems:
        wanted = set(args.problems)
        selected = [run for run in selected if run.folder.name in wanted or run.problem in wanted]
    if args.force_run:
        if "/" not in args.force_run:
            raise SystemExit("--force-run must look like '<folder>/<run_prefix>'")
        folder, run_prefix = args.force_run.split("/", 1)
        selected = [
            run for run in selected if run.folder.name == folder and run.run_prefix == run_prefix
        ]
    return selected


def load_checkpoint_info(checkpoint: Path) -> dict[str, Any]:
    chkpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = chkpt["model_state_dict"]
    layer_indices = sorted(
        {
            int(match.group(1))
            for key in state
            if (match := re.match(r"linears\.(\d+)\.weight$", key))
        }
    )
    if not layer_indices:
        raise ValueError(f"Could not find linears.* weights in {checkpoint}")

    first_weight = state[f"linears.{layer_indices[0]}.weight"]
    last_bias = state[f"linears.{layer_indices[-1]}.bias"]
    return {
        "checkpoint": chkpt,
        "input_dim": int(first_weight.shape[1]),
        "output_dim": int(last_bias.shape[0]),
    }


def parse_run_parameters(run: RunPaths, checkpoint_info: dict[str, Any]) -> dict[str, Any]:
    prefix = run.run_prefix
    expected = f"{run.problem}_"
    if not prefix.startswith(expected):
        raise ValueError(f"Run prefix does not start with {expected!r}: {prefix}")
    rest = prefix[len(expected) :]
    pattern = re.compile(
        r"^(?P<optimizer>[^_]+)_(?P<n_iterations>\d+)_adam_"
        r"(?P<n_iterations_lbfgs>\d+)_lbfgs_"
        r"(?P<num_domain>\d+)_domain_"
        r"(?P<num_boundary>\d+)_boundary_"
        r"(?P<num_initial>\d+)_initial_"
        r"(?P<n_layers>\d+)_x_"
        r"(?P<hidden_width>\d+)_hidden_float64_"
        r"(?P<float64>True|False)_(?P<seed>\d+)_"
        r"(?P<constraint>soft|hard)(?P<tail>.*)$"
    )
    match = pattern.match(rest)
    if not match:
        raise ValueError(f"Could not parse run prefix: {prefix}")

    tail = match.group("tail")
    broken = tail.endswith("_broken")
    if broken:
        tail = tail.removesuffix("_broken")
    point_removed = "none"
    if tail.startswith("_point_removed_"):
        point_removed = tail.removeprefix("_point_removed_")
    elif tail:
        raise ValueError(f"Unrecognized run suffix in {prefix}: {tail}")

    n_layers = int(match.group("n_layers"))
    hidden_width = int(match.group("hidden_width"))
    layers = [
        checkpoint_info["input_dim"],
        *([hidden_width] * n_layers),
        checkpoint_info["output_dim"],
    ]

    return {
        "problem_name": run.problem,
        "optimizer": match.group("optimizer"),
        "n_iterations": int(match.group("n_iterations")),
        "n_iterations_lbfgs": int(match.group("n_iterations_lbfgs")),
        "num_domain": int(match.group("num_domain")),
        "num_boundary": int(match.group("num_boundary")),
        "num_initial": int(match.group("num_initial")),
        "layers": layers,
        "seed": int(match.group("seed")),
        "float64": match.group("float64") == "True",
        "soft_constrained": match.group("constraint") == "soft",
        "drop_single_point_type": point_removed,
        "broken": broken,
    }


def construct_loaded_model(
    run: RunPaths,
    params: dict[str, Any],
) -> tuple[Any, Any, str, Path]:
    with contextlib.redirect_stdout(io.StringIO()):
        model, data, model_name, checkpoint_path = problem_factory.construct_problem(
            **params,
            load_path=str(run.folder),
            checkpoint_path=str(run.checkpoint),
            model_version="full",
        )
    return model, data, model_name, Path(checkpoint_path)


def candidate_influence_files(run: RunPaths, matrix_mode: str) -> list[Path]:
    if run.influence_dir is None:
        return []
    files = []
    for path in sorted(run.influence_dir.glob("*.npz")):
        if path.name.startswith("."):
            continue
        if matrix_mode == "core" and path.stem not in CORE_MATRIX_IDS:
            continue
        files.append(path)
    return files


def load_matrix_metadata(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "id": path.stem,
            "scores_shape": list(data["scores"].shape),
            "candidate_points": np.asarray(data["candidate_points"]),
            "num_pdes": int(data["num_pdes"]),
            "num_bcs": int(data["num_bcs"]),
            "n_outputs": int(data["n_outputs"]),
            "left_term": str(data["left_term"]),
            "right_term": str(data["right_term"]),
            "self_influence": bool(data["self_influence"]),
        }


def display_problem_name(problem: str) -> str:
    return problem.replace("_", " ").title().replace("Nd", "ND")


def term_label(problem: str, term: str) -> str:
    return loss_term_names.get(problem, {}).get(term, term.replace("_", " "))


def axis_names(dim: int) -> list[str]:
    axes = ["x", "y", "z", "w"]
    return [axes[idx] if idx < len(axes) else f"x{idx}" for idx in range(dim)]


def bounds_dict_from_arrays(
    mins: np.ndarray,
    maxs: np.ndarray,
    axes: list[str] | None = None,
) -> dict[str, list[float]]:
    axes = axes or axis_names(len(mins))
    return {
        axes[dim]: [
            float(mins[dim]),
            float(maxs[dim]),
        ]
        for dim in range(len(mins))
    }


def infer_bounds(points: np.ndarray) -> dict[str, list[float]]:
    axes = axis_names(points.shape[1])
    bounds = {}
    for dim in range(points.shape[1]):
        values = points[:, dim]
        bounds[axes[dim]] = [
            float(np.nanmin(values)),
            float(np.nanmax(values)),
        ]
    return bounds


def infer_min_max(points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] == 0 or len(points) == 0:
        return None
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    if not np.all(np.isfinite(mins)) or not np.all(np.isfinite(maxs)):
        return None
    return mins, maxs


def geom_min_max(geom: Any, fallback_points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        if hasattr(geom, "timedomain") and hasattr(geom, "geometry"):
            x_min, x_max = geom.geometry.bbox
            t_min, t_max = geom.timedomain.bbox
            mins = np.concatenate(
                [
                    np.asarray(x_min, dtype=np.float64).reshape(-1),
                    np.asarray(t_min, dtype=np.float64).reshape(-1),
                ]
            )
            maxs = np.concatenate(
                [
                    np.asarray(x_max, dtype=np.float64).reshape(-1),
                    np.asarray(t_max, dtype=np.float64).reshape(-1),
                ]
            )
        else:
            mins, maxs = geom.bbox
            mins = np.asarray(mins, dtype=np.float64).reshape(-1)
            maxs = np.asarray(maxs, dtype=np.float64).reshape(-1)
        if len(mins) and len(mins) == len(maxs) and np.all(np.isfinite(mins + maxs)):
            return mins, maxs
    except Exception:
        pass
    return infer_min_max(fallback_points)


def raster_dimensions(
    mins: np.ndarray, maxs: np.ndarray, max_axis_resolution=1024
) -> tuple[int, int]:
    if max_axis_resolution < 1:
        raise ValueError("max_axis_resolution must be >= 1")
    span_x = float(maxs[0] - mins[0])
    span_y = float(maxs[1] - mins[1])
    if span_x <= 0 or span_y <= 0:
        raise ValueError(f"Raster bounds must have positive spans, got {span_x=} {span_y=}")
    if span_x >= span_y:
        width = max_axis_resolution
        height = max(1, int(round(max_axis_resolution * span_y / span_x)))
    else:
        height = max_axis_resolution
        width = max(1, int(round(max_axis_resolution * span_x / span_y)))
    return width, height


def raster_points_from_bounds(
    mins: np.ndarray,
    maxs: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    dx = float(maxs[0] - mins[0]) / width
    dy = float(maxs[1] - mins[1]) / height
    x_centers = mins[0] + (np.arange(width, dtype=np.float64) + 0.5) * dx
    y_centers = maxs[1] - (np.arange(height, dtype=np.float64) + 0.5) * dy
    xx, yy = np.meshgrid(x_centers, y_centers)
    return np.stack([xx.ravel(), yy.ravel()], axis=1)


def raster_domain_mask(geom: Any, points: np.ndarray, height: int, width: int) -> np.ndarray:
    if geom is None or not hasattr(geom, "inside"):
        return np.ones((height, width), dtype=np.uint8)
    try:
        mask = np.asarray(geom.inside(points), dtype=bool).reshape(-1)
        if mask.shape[0] != points.shape[0]:
            raise ValueError("inside() returned an unexpected mask shape")
        return mask.astype(np.uint8).reshape(height, width)
    except Exception:
        return np.ones((height, width), dtype=np.uint8)


def make_raster_grid(
    geom: Any,
    fallback_points: np.ndarray,
    max_axis_resolution: int = DEFAULT_RASTER_MAX_RESOLUTION,
) -> RasterGrid | None:
    min_max = geom_min_max(geom, fallback_points)
    if min_max is None:
        return None
    mins, maxs = min_max
    if len(mins) != 2:
        return None
    width, height = raster_dimensions(mins, maxs, max_axis_resolution)
    points = raster_points_from_bounds(mins, maxs, width, height)
    mask = raster_domain_mask(geom, points, height, width)
    axes = axis_names(2)
    return RasterGrid(
        points=points,
        mask=mask,
        width=width,
        height=height,
        bounds=bounds_dict_from_arrays(mins, maxs, axes),
        axes=axes,
    )


def validate_points_match(
    path: Path,
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
) -> None:
    if actual.shape != expected.shape:
        raise ValueError(f"{path.name}: {name} shape {actual.shape} != expected {expected.shape}")
    if len(actual) and not np.allclose(actual, expected):
        raise ValueError(f"{path.name}: {name} values do not match expected points")


def infer_train_labels(data: Any, n_train: int) -> tuple[np.ndarray, np.ndarray]:
    kind = np.zeros(n_train, dtype=np.uint8)
    bc_id = np.full(n_train, -1, dtype=np.int16)
    counts = getattr(data, "num_bcs", None)
    if counts is None:
        return kind, bc_id
    start = 0
    for idx, count in enumerate(counts):
        end = min(start + int(count), n_train)
        kind[start:end] = 1
        bc_id[start:end] = idx
        start = end
    return kind, bc_id


def predict_fields(
    model: Any,
    data: Any,
    points: np.ndarray,
    num_pdes: int,
    num_bcs: int,
    float64: bool,
    batch_size: int = 512,
) -> dict[str, np.ndarray]:
    torch.set_default_device("cpu")
    dtype = torch.float64 if float64 else torch.float32
    model.net.eval()

    predictions = []
    with torch.no_grad():
        for start in range(0, len(points), batch_size):
            batch = torch.tensor(points[start : start + batch_size], dtype=dtype, device="cpu")
            pred = model.net(batch).detach().cpu().numpy()
            predictions.append(pred)
    pred_all = np.concatenate(predictions, axis=0)
    if pred_all.ndim == 1:
        pred_all = pred_all.reshape(-1, 1)

    wrapper = ModelWrapper(model.net, pde=data.pde, bcs=data.bcs).eval()
    residuals = []
    for start in range(0, len(points), batch_size):
        batch = torch.tensor(
            points[start : start + batch_size],
            dtype=dtype,
            device="cpu",
            requires_grad=True,
        )
        res = wrapper(batch).detach().cpu().numpy()
        residuals.append(res)
        dde.grad.clear()
    residual_all = np.concatenate(residuals, axis=0)
    squared = residual_all.astype(np.float64) ** 2

    fields: dict[str, np.ndarray] = {}
    for idx in range(pred_all.shape[1]):
        fields[f"pred_output_{idx}"] = pred_all[:, idx].astype(np.float32)

    fields["loss_total"] = squared.sum(axis=1).astype(np.float32)
    if num_pdes > 0:
        fields["loss_pde"] = squared[:, :num_pdes].sum(axis=1).astype(np.float32)
        for idx in range(num_pdes):
            fields[f"loss_pde_{idx}"] = squared[:, idx].astype(np.float32)
    if num_bcs > 0:
        bc_start = num_pdes
        bc_stop = num_pdes + num_bcs
        fields["loss_bc"] = squared[:, bc_start:bc_stop].sum(axis=1).astype(np.float32)
        for idx in range(num_bcs):
            fields[f"loss_bc_{idx}"] = squared[:, bc_start + idx].astype(np.float32)
    return fields


def topk_sorted(values: np.ndarray, k: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    if mode == "abs":
        order_source = np.abs(values)
        part = np.argpartition(-order_source, kth=k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(order_source, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
    elif mode == "pos":
        part = np.argpartition(-values, kth=k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(values, part, axis=1)
        order = np.argsort(-part_scores, axis=1)
    elif mode == "neg":
        part = np.argpartition(values, kth=k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(values, part, axis=1)
        order = np.argsort(part_scores, axis=1)
    else:
        raise ValueError(f"Unknown top-k mode: {mode}")

    indices = np.take_along_axis(part, order, axis=1)
    top_values = np.take_along_axis(values, indices, axis=1)
    return indices.astype(np.uint32), top_values.astype(np.float32)


def process_influence_matrix(
    path: Path,
    out_dir: Path,
    rel_prefix: str,
    n_train: int,
    row_source: str,
    row_count: int,
    max_local_influence_points: int,
    row_chunk_size: int,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        scores = np.asarray(data["scores"], dtype=np.float32)
        candidate_points = np.asarray(data["candidate_points"])
        metadata: dict[str, Any] = {
            "id": path.stem,
            "method": "GradDot" if path.stem.startswith("grad_dot") else "PINNfluence",
            "left_term": str(data["left_term"]),
            "right_term": str(data["right_term"]),
            "num_pdes": int(data["num_pdes"]),
            "num_bcs": int(data["num_bcs"]),
            "n_outputs": int(data["n_outputs"]),
            "self_influence": bool(data["self_influence"]),
            "scores_shape": list(scores.shape),
        }

    if scores.ndim != 2:
        raise ValueError(f"Expected 2D scores in {path}, got {scores.shape}")
    if scores.shape[0] != row_count:
        raise ValueError(
            f"{path.name}: score rows {scores.shape[0]} != {row_source} count {row_count}"
        )
    if scores.shape[1] != n_train:
        raise ValueError(f"{path.name}: score columns {scores.shape[1]} != n_train {n_train}")

    k = min(max_local_influence_points, scores.shape[1])
    display_scores = (-scores / float(n_train)).astype(np.float32, copy=False)
    matrix_dir = f"{rel_prefix}/{path.stem}"
    index_dtype = "uint16" if n_train <= 65535 else "uint32"

    top_chunks: dict[str, Any] = {}
    for mode in ("abs", "pos", "neg"):
        indices, values = topk_sorted(display_scores, k, mode)
        chunk_entries = []
        for chunk_id, start in enumerate(range(0, row_count, row_chunk_size)):
            stop = min(row_count, start + row_chunk_size)
            chunk_indices = indices[start:stop]
            chunk_values = values[start:stop]
            quantized_values, value_scale = quantize_int16_symmetric(chunk_values)
            chunk_entries.append(
                {
                    "id": chunk_id,
                    "row_start": start,
                    "row_count": stop - start,
                    "k": k,
                    "value_scale": value_scale,
                    "indices": write_array(
                        out_dir,
                        f"{matrix_dir}/{mode}/chunks/{chunk_id}_indices.{dtype_extension(index_dtype)}",
                        chunk_indices,
                        index_dtype,
                    ),
                    "values": write_array(
                        out_dir,
                        f"{matrix_dir}/{mode}/chunks/{chunk_id}_values.i16",
                        quantized_values,
                        "int16",
                    ),
                }
            )
        top_chunks[mode] = {
            "row_chunk_size": row_chunk_size,
            "chunk_count": len(chunk_entries),
            "indices_dtype": index_dtype,
            "values_dtype": "int16",
            "value_encoding": {
                "kind": "symmetric_linear",
                "scale_by": "chunk.value_scale",
            },
            "chunks": chunk_entries,
        }

    abs_scores = np.abs(display_scores)
    summary = {
        "mean_signed": write_array(
            out_dir,
            f"{matrix_dir}/summary_mean_signed.f32",
            display_scores.mean(axis=0),
            "float32",
        ),
        "mean_abs": write_array(
            out_dir,
            f"{matrix_dir}/summary_mean_abs.f32",
            abs_scores.mean(axis=0),
            "float32",
        ),
        "max_abs": write_array(
            out_dir,
            f"{matrix_dir}/summary_max_abs.f32",
            abs_scores.max(axis=0),
            "float32",
        ),
        "positive_mass": write_array(
            out_dir,
            f"{matrix_dir}/summary_positive_mass.f32",
            np.clip(display_scores, 0, None).sum(axis=0),
            "float32",
        ),
        "negative_mass": write_array(
            out_dir,
            f"{matrix_dir}/summary_negative_mass.f32",
            np.clip(display_scores, None, 0).sum(axis=0),
            "float32",
        ),
    }

    metadata.update(
        {
            "candidate_points_shape": list(candidate_points.shape),
            "row_source": row_source,
            "row_count": row_count,
            "k": k,
            "max_local_influence_points": max_local_influence_points,
            "row_chunk_size": row_chunk_size,
            "label": (f"{metadata['method']}: {metadata['right_term']} -> {metadata['left_term']}"),
            "display_label": (
                f"{metadata['method']} / "
                f"{metadata['right_term'].replace('_', ' ')} -> "
                f"{metadata['left_term'].replace('_', ' ')}"
                f"{' (self)' if metadata['self_influence'] else ''}"
            ),
            "top_chunks": top_chunks,
            "summary": summary,
        }
    )
    return metadata


def process_influence_matrix_jobs(
    jobs: list[tuple[int, Path, str, int]],
    out_dir: Path,
    rel_prefix: str,
    n_train: int,
    max_local_influence_points: int,
    row_chunk_size: int,
    workers: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    results: list[dict[str, Any] | None] = [None] * len(jobs)
    errors: list[str] = []
    workers = max(1, min(workers, len(jobs) or 1))

    if workers == 1:
        for idx, matrix_path, row_source, row_count in jobs:
            try:
                results[idx] = process_influence_matrix(
                    matrix_path,
                    out_dir=out_dir,
                    rel_prefix=rel_prefix,
                    n_train=n_train,
                    row_source=row_source,
                    row_count=row_count,
                    max_local_influence_points=max_local_influence_points,
                    row_chunk_size=row_chunk_size,
                )
                print(f"  built {matrix_path.name}")
            except Exception as exc:
                errors.append(f"{matrix_path.name}: {exc}")
    else:
        print(f"  exporting {len(jobs)} matrices with {workers} workers")
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_influence_matrix,
                    matrix_path,
                    out_dir,
                    rel_prefix,
                    n_train,
                    row_source,
                    row_count,
                    max_local_influence_points,
                    row_chunk_size,
                ): (idx, matrix_path)
                for idx, matrix_path, row_source, row_count in jobs
            }
            for future in as_completed(futures):
                idx, matrix_path = futures[future]
                try:
                    results[idx] = future.result()
                    print(f"  built {matrix_path.name}")
                except Exception as exc:
                    errors.append(f"{matrix_path.name}: {exc}")

    return [result for result in results if result is not None], errors


def validation_summary(validation_dir: Path | None) -> dict[str, Any]:
    if validation_dir is None:
        return {"available": False, "counts": {}}
    counts: dict[str, int] = {}
    loo_indices: dict[str, list[int]] = {}
    for path in sorted(validation_dir.glob("*.pt")):
        stem = path.stem
        method = stem.split("_", 1)[0]
        counts[method] = counts.get(method, 0) + 1
        if "_loo_" in stem:
            idx_text = stem.rsplit("_loo_", 1)[-1]
            if idx_text.isdigit():
                loo_indices.setdefault(method, []).append(int(idx_text))
    return {
        "available": True,
        "counts": counts,
        "loo_indices": {k: sorted(v) for k, v in loo_indices.items()},
    }


def build_run(run: RunPaths, args: argparse.Namespace) -> dict[str, Any]:
    print(f"\nProcessing {run.folder.name}/{run.run_prefix}")
    out_dir = args.out_root / run.folder.name / run.run_prefix
    if out_dir.exists():
        if args.overwrite:
            shutil.rmtree(out_dir)
        else:
            raise SystemExit(f"Output exists, pass --overwrite: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix_files = candidate_influence_files(run, args.matrix_mode)
    complete = run.checkpoint is not None and run.influence_dir is not None
    status = "complete" if complete else "incomplete"
    errors: list[str] = []

    if not matrix_files:
        errors.append("No influence matrices found")

    checkpoint_info: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    model = data = None
    train_points = None
    raster_grid: RasterGrid | None = None
    raster_fields: dict[str, np.ndarray] = {}

    if run.checkpoint is not None:
        checkpoint_info = load_checkpoint_info(run.checkpoint)
        train_points = np.asarray(checkpoint_info["checkpoint"]["train_x_all"], dtype=np.float64)
        try:
            params = parse_run_parameters(run, checkpoint_info)
        except Exception as exc:
            errors.append(f"Could not parse model parameters: {exc}")
    else:
        errors.append("Missing _full.pt checkpoint")

    if train_points is None and matrix_files:
        first_meta = load_matrix_metadata(matrix_files[0])
        train_points = np.zeros((first_meta["scores_shape"][1], 2), dtype=np.float64)
        errors.append("Train point coordinates unavailable; wrote placeholder points")

    if train_points is None:
        train_points = np.zeros((0, 2), dtype=np.float64)

    first_meta = load_matrix_metadata(matrix_files[0]) if matrix_files else None
    candidate_meta = None
    for matrix_path in matrix_files:
        meta = load_matrix_metadata(matrix_path)
        if not meta["self_influence"]:
            candidate_meta = meta
            break
    if candidate_meta is None:
        candidate_meta = first_meta
    candidate_points = (
        candidate_meta["candidate_points"]
        if candidate_meta is not None
        else np.zeros((0, train_points.shape[1] if train_points.size else 2))
    )
    num_pdes = int(first_meta["num_pdes"]) if first_meta else 0
    num_bcs = int(first_meta["num_bcs"]) if first_meta else 0
    n_outputs = (
        int(first_meta["n_outputs"])
        if first_meta
        else int(checkpoint_info["output_dim"] if checkpoint_info else 0)
    )

    if params is not None and not args.skip_fields:
        try:
            model, data, _, _ = construct_loaded_model(run, params)
            try:
                raster_grid = make_raster_grid(
                    data.geom, candidate_points, args.raster_max_resolution
                )
                if raster_grid is not None:
                    raster_fields = predict_fields(
                        model=model,
                        data=data,
                        points=raster_grid.points,
                        num_pdes=num_pdes,
                        num_bcs=num_bcs,
                        float64=params["float64"],
                    )
                else:
                    errors.append(
                        "Field raster precomputation skipped: geometry is not 2D or bounds are unavailable"
                    )
            except Exception as exc:
                raster_grid = None
                raster_fields = {}
                errors.append(f"Field raster precomputation failed: {exc}")
        except Exception as exc:
            errors.append(f"Field precomputation failed: {exc}")

    if data is not None:
        train_kind, train_bc_id = infer_train_labels(data, len(train_points))
    else:
        train_kind = np.zeros(len(train_points), dtype=np.uint8)
        train_bc_id = np.full(len(train_points), -1, dtype=np.int16)

    arrays: dict[str, Any] = {
        "candidate_points": write_array(
            out_dir, "arrays/candidate_points.f32", candidate_points, "float32"
        ),
        "train_points": write_array(out_dir, "arrays/train_points.f32", train_points, "float32"),
        "train_kind": write_array(out_dir, "arrays/train_kind.u8", train_kind, "uint8"),
        "train_bc_id": write_array(out_dir, "arrays/train_bc_id.i16", train_bc_id, "int16"),
    }
    field_raster_entry = None
    if raster_grid is not None:
        arrays["field_raster_mask"] = write_array(
            out_dir,
            "arrays/field_raster_mask.u8",
            raster_grid.mask,
            "uint8",
        )
        field_raster_entry = {
            "width": raster_grid.width,
            "height": raster_grid.height,
            "shape": [raster_grid.height, raster_grid.width],
            "bounds": raster_grid.bounds,
            "axes": raster_grid.axes,
            "max_axis_resolution": args.raster_max_resolution,
            "coordinate_order": {
                "columns": "x_ascending",
                "rows": "y_descending",
                "sample": "pixel_center",
            },
            "mask": arrays["field_raster_mask"],
        }

    field_entries: dict[str, Any] = {}
    for name, values in sorted(raster_fields.items()):
        entry = {
            "label": field_label(run.problem, name),
            "kind": "prediction" if name.startswith("pred_") else "loss",
        }
        if raster_grid is not None:
            raster_values = np.asarray(values, dtype=np.float32)
            expected = raster_grid.height * raster_grid.width
            if raster_values.size != expected:
                raise ValueError(
                    f"{name}: raster length {raster_values.size} != {raster_grid.height} * {raster_grid.width}"
                )
            quantized, encoding, display_domain = quantize_uint16_linear(
                raster_values.reshape(raster_grid.height, raster_grid.width),
                raster_grid.mask,
            )
            entry["raster"] = write_array(
                out_dir,
                f"arrays/{name}_raster.u16",
                quantized,
                "uint16",
            )
            entry["encoding"] = encoding
            entry["display_domain"] = display_domain
        field_entries[name] = entry

    matrix_jobs: list[tuple[int, Path, str, int]] = []
    for matrix_path in matrix_files:
        try:
            meta = load_matrix_metadata(matrix_path)
            row_source = "train_points" if meta["self_influence"] else "candidate_points"
            row_points = train_points if meta["self_influence"] else candidate_points
            validate_points_match(
                matrix_path,
                f"{row_source} row points",
                meta["candidate_points"],
                row_points,
            )
            matrix_jobs.append((len(matrix_jobs), matrix_path, row_source, len(row_points)))
        except Exception as exc:
            errors.append(f"{matrix_path.name}: {exc}")

    influence_entries, processing_errors = process_influence_matrix_jobs(
        matrix_jobs,
        out_dir=out_dir,
        rel_prefix="influence",
        n_train=len(train_points),
        max_local_influence_points=args.max_local_influence_points,
        row_chunk_size=args.row_chunk_size,
        workers=int(getattr(args, "workers", 1)),
    )
    errors.extend(processing_errors)

    available_terms = sorted(
        {entry["left_term"] for entry in influence_entries}
        | {entry["right_term"] for entry in influence_entries}
    )
    default_field = (
        "pred_output_0" if "pred_output_0" in field_entries else (next(iter(field_entries), None))
    )
    matrix_ids = {entry["id"] for entry in influence_entries}
    default_matrix = (
        DEFAULT_MATRIX_ID
        if DEFAULT_MATRIX_ID in matrix_ids
        else (influence_entries[0]["id"] if influence_entries else None)
    )

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "problem": run.problem,
        "folder": run.folder.name,
        "run_id": run.run_prefix,
        "display_name": display_problem_name(run.problem),
        "status": status if not errors else ("partial" if complete else "incomplete"),
        "errors": errors,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "matrix_mode": args.matrix_mode,
        "max_local_influence_points": args.max_local_influence_points,
        "row_chunk_size": args.row_chunk_size,
        "axes": ["x", "y"][: candidate_points.shape[1]],
        "bounds": (
            raster_grid.bounds
            if raster_grid is not None
            else (infer_bounds(candidate_points) if len(candidate_points) else {})
        ),
        "n_candidate": len(candidate_points),
        "n_train": len(train_points),
        "n_outputs": n_outputs,
        "num_pdes": num_pdes,
        "num_bcs": num_bcs,
        "available_terms": available_terms,
        "term_labels": {term: term_label(run.problem, term) for term in available_terms},
        "default_field": default_field,
        "default_matrix": default_matrix,
        "arrays": arrays,
        "field_raster": field_raster_entry,
        "fields": field_entries,
        "influence_matrices": influence_entries,
        "validation": validation_summary(run.validation_dir),
    }
    write_json(out_dir / "manifest.json", manifest)
    return {
        "problem": run.problem,
        "folder": run.folder.name,
        "run_id": run.run_prefix,
        "display_name": manifest["display_name"],
        "status": manifest["status"],
        "manifest": slug_path(Path(run.folder.name) / run.run_prefix / "manifest.json"),
        "default_field": default_field,
        "default_matrix": default_matrix,
        "n_candidate": manifest["n_candidate"],
        "n_train": manifest["n_train"],
        "n_matrices": len(influence_entries),
        "n_fields": len(field_entries),
        "errors": errors,
    }


def field_label(problem: str, name: str) -> str:
    if name.startswith("pred_output_"):
        output = "output_" + name.rsplit("_", 1)[-1]
        return f"Prediction {term_label(problem, output)}"
    if name == "loss_total":
        return term_label(problem, "total_loss")
    if name == "loss_pde":
        return term_label(problem, "pde_loss")
    if name == "loss_bc":
        return term_label(problem, "bc_loss")
    if name.startswith("loss_pde_"):
        return term_label(problem, "pde_" + name.rsplit("_", 1)[-1])
    if name.startswith("loss_bc_"):
        return term_label(problem, "bc_" + name.rsplit("_", 1)[-1])
    return name.replace("_", " ")


def main() -> None:
    args = parse_args()
    torch.set_default_device("cpu")
    if args.max_local_influence_points < 1:
        raise SystemExit("--max_local_influence_points must be >= 1")
    if args.row_chunk_size < 1:
        raise SystemExit("--row-chunk-size must be >= 1")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.raster_max_resolution < 1:
        raise SystemExit("--raster-max-resolution must be >= 1")
    bundle_budget_bytes = int(args.bundle_size_budget_mb) * 1024 * 1024

    args.data_root = Path(__file__).resolve().parent.parent / "raw_data"
    args.out_root = Path(__file__).resolve().parent.parent / "webdemo" / "public" / "data"
    runs = filter_runs(discover_runs(args.data_root), args)
    if not runs:
        raise SystemExit("No runs matched the requested filters")

    args.out_root.mkdir(parents=True, exist_ok=True)
    index_entries = []
    for run in runs:
        if args.skip_incomplete and (run.checkpoint is None or run.influence_dir is None):
            continue
        try:
            index_entries.append(build_run(run, args))
        except Exception as exc:
            index_entries.append(
                {
                    "problem": run.problem,
                    "folder": run.folder.name,
                    "run_id": run.run_prefix,
                    "display_name": display_problem_name(run.problem),
                    "status": "failed",
                    "manifest": None,
                    "default_field": None,
                    "default_matrix": None,
                    "errors": [str(exc)],
                }
            )
            print(f"FAILED {run.folder.name}/{run.run_prefix}: {exc}")

    index = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "matrix_mode": args.matrix_mode,
        "max_local_influence_points": args.max_local_influence_points,
        "row_chunk_size": args.row_chunk_size,
        "raster_max_resolution": args.raster_max_resolution,
        "bundle_report": "bundle_report.json",
        "runs": index_entries,
    }
    write_json(args.out_root / "index.json", index)
    report = build_bundle_report(args.out_root, bundle_budget_bytes)
    write_json(args.out_root / "bundle_report.json", report)
    if not report["within_budget"]:
        raise SystemExit(
            f"Deployable bundle is {report['total_bytes'] / (1024 * 1024):.1f} MB, "
            f"above the {args.bundle_size_budget_mb} MB budget"
        )
    print(f"\nWrote {args.out_root / 'index.json'}")


if __name__ == "__main__":
    main()
