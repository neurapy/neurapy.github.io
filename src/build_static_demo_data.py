# pyright: reportMissingTypeStubs=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false
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
import re
import shutil
import time
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
    "uint8": np.uint8,
    "int16": np.int16,
}


@dataclass
class RunPaths:
    folder: Path
    problem: str
    run_prefix: str
    checkpoint: Path | None
    influence_dir: Path | None
    validation_dir: Path | None


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
    parser.add_argument("--k-max", default=200, type=int)
    parser.add_argument(
        "--field-points",
        default=60_000,
        type=int,
        help="Number of precomputed display points for prediction/loss fields. Use 0 to reuse influence candidate points.",
    )
    parser.add_argument(
        "--matrix-mode",
        choices=["core", "all"],
        default="core",
        help="core keeps the public bundle compact; all exports every influence matrix.",
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
        help="Do not include incomplete runs in webdemo/data/index.json.",
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
    arr.tofile(path)
    return {
        "path": rel_path,
        "dtype": dtype,
        "shape": list(arr.shape),
        "bytes": path.stat().st_size,
    }


def slug_path(path: Path) -> str:
    return path.as_posix()


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


def infer_bounds(points: np.ndarray) -> dict[str, list[float]]:
    axes = ["x", "y", "z", "w"]
    bounds = {}
    for dim in range(points.shape[1]):
        values = points[:, dim]
        bounds[axes[dim] if dim < len(axes) else f"x{dim}"] = [
            float(np.nanmin(values)),
            float(np.nanmax(values)),
        ]
    return bounds


def sample_display_points(data: Any, n_points: int, fallback: np.ndarray) -> np.ndarray:
    if n_points <= 0 or data is None:
        return fallback
    geom = data.geom
    for method in ("uniform_points", "random_points"):
        sampler = getattr(geom, method, None)
        if sampler is None:
            continue
        try:
            points = sampler(n_points)
            points = np.asarray(points, dtype=np.float64)
            if points.ndim == 2 and len(points) > 0:
                return points
        except Exception:
            continue
    return fallback


def nearest_candidate_indices(
    display_points: np.ndarray, candidate_points: np.ndarray
) -> np.ndarray:
    if len(display_points) == len(candidate_points) and np.allclose(
        display_points, candidate_points
    ):
        return np.arange(len(candidate_points), dtype=np.uint32)
    try:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=1, algorithm="auto")
        nn.fit(candidate_points)
        indices = np.asarray(nn.kneighbors(display_points, return_distance=False))[:, 0]
        return indices.astype(np.uint32)
    except Exception:
        indices = np.empty(len(display_points), dtype=np.uint32)
        chunk_size = 1024
        candidate = candidate_points.astype(np.float64, copy=False)
        for start in range(0, len(display_points), chunk_size):
            chunk = display_points[start : start + chunk_size].astype(np.float64, copy=False)
            dist2 = ((chunk[:, None, :] - candidate[None, :, :]) ** 2).sum(axis=2)
            indices[start : start + len(chunk)] = np.argmin(dist2, axis=1).astype(np.uint32)
        return indices


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
    k_max: int,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        scores = np.asarray(data["scores"], dtype=np.float32)
        candidate_points = np.asarray(data["candidate_points"])
        metadata: dict[str, Any] = {
            "id": path.stem,
            "source_file": str(path),
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
    if scores.shape[1] != n_train:
        raise ValueError(f"{path.name}: score columns {scores.shape[1]} != n_train {n_train}")

    k = min(k_max, scores.shape[1])
    display_scores = (-scores / float(n_train)).astype(np.float32, copy=False)
    matrix_dir = f"{rel_prefix}/{path.stem}"

    top_entries: dict[str, Any] = {}
    for mode in ("abs", "pos", "neg"):
        indices, values = topk_sorted(display_scores, k, mode)
        top_entries[mode] = {
            "indices": write_array(
                out_dir,
                f"{matrix_dir}/top_{mode}_indices.u32",
                indices,
                "uint32",
            ),
            "values": write_array(
                out_dir,
                f"{matrix_dir}/top_{mode}_values.f32",
                values,
                "float32",
            ),
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
            "k": k,
            "label": (f"{metadata['method']}: {metadata['right_term']} -> {metadata['left_term']}"),
            "display_label": (
                f"{metadata['method']} / "
                f"{metadata['right_term'].replace('_', ' ')} -> "
                f"{metadata['left_term'].replace('_', ' ')}"
            ),
            "top": top_entries,
            "summary": summary,
        }
    )
    return metadata


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
        "path": str(validation_dir),
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
    fields: dict[str, np.ndarray] = {}

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
    candidate_points = (
        first_meta["candidate_points"]
        if first_meta is not None
        else np.zeros((0, train_points.shape[1] if train_points.size else 2))
    )
    display_points = candidate_points
    display_to_candidate = np.arange(len(candidate_points), dtype=np.uint32)

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
            display_points = sample_display_points(data, args.field_points, candidate_points)
            display_to_candidate = nearest_candidate_indices(display_points, candidate_points)
            fields = predict_fields(
                model=model,
                data=data,
                points=display_points,
                num_pdes=num_pdes,
                num_bcs=num_bcs,
                float64=params["float64"],
            )
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
        "display_points": write_array(
            out_dir, "arrays/display_points.f32", display_points, "float32"
        ),
        "display_to_candidate": write_array(
            out_dir,
            "arrays/display_to_candidate.u32",
            display_to_candidate,
            "uint32",
        ),
        "train_points": write_array(out_dir, "arrays/train_points.f32", train_points, "float32"),
        "train_kind": write_array(out_dir, "arrays/train_kind.u8", train_kind, "uint8"),
        "train_bc_id": write_array(out_dir, "arrays/train_bc_id.i16", train_bc_id, "int16"),
    }

    field_entries: dict[str, Any] = {}
    for name, values in sorted(fields.items()):
        field_entries[name] = {
            "label": field_label(run.problem, name),
            "kind": "prediction" if name.startswith("pred_") else "loss",
            "array": write_array(out_dir, f"arrays/{name}.f32", values, "float32"),
        }

    influence_entries = []
    for matrix_path in matrix_files:
        try:
            meta = load_matrix_metadata(matrix_path)
            if not np.allclose(meta["candidate_points"], candidate_points):
                raise ValueError("candidate_points do not match the first matrix")
            influence_entries.append(
                process_influence_matrix(
                    matrix_path,
                    out_dir=out_dir,
                    rel_prefix="influence",
                    n_train=len(train_points),
                    k_max=args.k_max,
                )
            )
            print(f"  built {matrix_path.name}")
        except Exception as exc:
            errors.append(f"{matrix_path.name}: {exc}")

    available_terms = sorted(
        {entry["left_term"] for entry in influence_entries}
        | {entry["right_term"] for entry in influence_entries}
    )

    manifest = {
        "schema_version": 1,
        "problem": run.problem,
        "folder": run.folder.name,
        "run_id": run.run_prefix,
        "display_name": display_problem_name(run.problem),
        "status": status if not errors else ("partial" if complete else "incomplete"),
        "errors": errors,
        "source": {
            "checkpoint": str(run.checkpoint) if run.checkpoint else None,
            "influence_dir": str(run.influence_dir) if run.influence_dir else None,
            "validation_dir": str(run.validation_dir) if run.validation_dir else None,
        },
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "matrix_mode": args.matrix_mode,
        "k_max": args.k_max,
        "axes": ["x", "y"][: candidate_points.shape[1]],
        "bounds": infer_bounds(display_points) if len(display_points) else {},
        "n_candidate": len(candidate_points),
        "n_display": len(display_points),
        "n_train": len(train_points),
        "n_outputs": n_outputs,
        "num_pdes": num_pdes,
        "num_bcs": num_bcs,
        "available_terms": available_terms,
        "term_labels": {term: term_label(run.problem, term) for term in available_terms},
        "arrays": arrays,
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
    if args.k_max < 1:
        raise SystemExit("--k-max must be >= 1")

    args.data_root = Path(__file__).resolve().parent.parent / "raw_data"
    args.out_root = Path(__file__).resolve().parent.parent / "webdemo" / "data"
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
                    "errors": [str(exc)],
                }
            )
            print(f"FAILED {run.folder.name}/{run.run_prefix}: {exc}")

    index = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "data_root": str(args.data_root),
        "matrix_mode": args.matrix_mode,
        "k_max": args.k_max,
        "runs": index_entries,
    }
    write_json(args.out_root / "index.json", index)
    print(f"\nWrote {args.out_root / 'index.json'}")


if __name__ == "__main__":
    main()
