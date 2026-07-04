#!/usr/bin/env python3
"""Build schema-v2 static data for the webdemo Results dashboard."""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pinnfluence.utils.utils import loss_term_names

SCHEMA_VERSION = 2
DEFAULT_RAW_DATA_DIR = Path("raw_data")
DEFAULT_OUTPUT_DIR = Path("webdemo/public/data/results")
DEFAULT_N_BINS = 50

MatrixArray = npt.NDArray[np.floating[Any]]
FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class SourceMatrix:
    term: str
    output_id: str
    path: Path
    matrix_id: str


@dataclass(frozen=True)
class MatrixMeta:
    left_term: str
    right_term: str
    self_influence: bool
    scores_shape: tuple[int, int]
    candidate_points_shape: tuple[int, ...]
    num_pdes: int
    num_bcs: int
    n_outputs: int


@dataclass(frozen=True)
class OutputArrays:
    bin_centers: npt.NDArray[np.float32]
    binned_fractions: npt.NDArray[np.float32]
    binned_fractions_std: npt.NDArray[np.float32]
    binned_coherence: npt.NDArray[np.float32]
    binned_coherence_std: npt.NDArray[np.float32]


@dataclass(frozen=True)
class OutputComputation:
    output_id: str
    output_label: str
    axis: dict[str, str]
    terms: list[dict[str, Any]]
    mean_coherence: float
    std_coherence: float
    n_candidate: int
    n_train: int
    source_matrix_ids: list[str]
    arrays: OutputArrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", type=Path, default=DEFAULT_RAW_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bins", type=int, default=DEFAULT_N_BINS)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_f32(path: Path, values: npt.ArrayLike) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    array.tofile(path)
    return {
        "path": path.name,
        "dtype": "float32",
        "shape": list(array.shape),
        "bytes": int(array.nbytes),
    }


def display_problem_name(problem: str) -> str:
    if problem == "poisson_disk":
        return "Poisson Disk"
    problem = problem.removesuffix("_nd").replace("_disk", "")
    return problem.replace("_", " ").title()


def canonical_problem(problem: str) -> str:
    return "poisson_disk" if problem == "poisson" else problem


def term_label(problem: str, term: str) -> str:
    return loss_term_names.get(canonical_problem(problem), {}).get(term, term.replace("_", " "))


def term_sort_key(term: str) -> tuple[int, int, str]:
    prefix, _, suffix = term.partition("_")
    order = {"pde": 0, "bc": 1}.get(prefix, 2)
    index = int(suffix) if suffix.isdigit() else 999
    return order, index, term


def parse_raw_variant_dir(path: Path) -> tuple[str, str] | None:
    match = re.fullmatch(r"(.+)_float64_(good|bad)", path.name)
    if not match:
        return None
    return canonical_problem(match.group(1)), match.group(2)


def result_axis(problem: str, candidate_points: MatrixArray) -> tuple[dict[str, str], FloatArray]:
    points = np.asarray(candidate_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 1:
        raise ValueError(f"{problem}: candidate_points must be a 2D coordinate array")
    if problem == "poisson_disk":
        if points.shape[1] < 2:
            raise ValueError("poisson_disk: radial axis requires at least 2 coordinate columns")
        return {"id": "r", "label": "r"}, np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    if problem == "navier_stokes_nd":
        return {"id": "x", "label": "x"}, 0.1 * points[:, 0]
    if points.shape[1] < 2:
        raise ValueError(f"{problem}: temporal axis requires coordinate column 1")
    return {"id": "t", "label": "t"}, points[:, 1]


def discover_score_dirs(raw_data_dir: Path) -> list[tuple[str, str, Path]]:
    records: list[tuple[str, str, Path]] = []
    for variant_dir in sorted(raw_data_dir.glob("*_float64_*")):
        parsed = parse_raw_variant_dir(variant_dir)
        if parsed is None:
            continue
        problem, quality = parsed
        score_dirs = sorted(variant_dir.glob("*_influence_scores"))
        if not score_dirs:
            raise FileNotFoundError(f"{variant_dir}: missing *_influence_scores directory")
        for score_dir in score_dirs:
            records.append((problem, quality, score_dir))
    return records


def parse_output_matrix_path(path: Path) -> SourceMatrix | None:
    if path.name.endswith("_self.npz"):
        return None
    stem = path.stem
    prefix = "influences_"
    if not stem.startswith(prefix):
        return None
    payload = stem.removeprefix(prefix)
    term, sep, output_suffix = payload.rpartition("_output_")
    if sep != "_output_" or not output_suffix.isdigit():
        return None
    if term == "total_loss" or term.startswith("output_"):
        return None
    return SourceMatrix(term=term, output_id=f"output_{output_suffix}", path=path, matrix_id=stem)


def discover_output_matrices(score_dir: Path) -> dict[str, dict[str, SourceMatrix]]:
    outputs: dict[str, dict[str, SourceMatrix]] = {}
    for path in sorted(score_dir.glob("*.npz")):
        matrix = parse_output_matrix_path(path)
        if matrix is None:
            continue
        outputs.setdefault(matrix.output_id, {})[matrix.term] = matrix
    return outputs


def selected_terms(available_terms: set[str]) -> list[str]:
    pde_terms = sorted(
        (term for term in available_terms if re.fullmatch(r"pde_\d+", term)), key=term_sort_key
    )
    bc_terms = sorted(
        (term for term in available_terms if re.fullmatch(r"bc_\d+", term)), key=term_sort_key
    )
    if not pde_terms and "pde_loss" in available_terms:
        pde_terms = ["pde_loss"]
    if not bc_terms and "bc_loss" in available_terms:
        bc_terms = ["bc_loss"]
    return pde_terms + bc_terms


def read_matrix_meta(path: Path) -> MatrixMeta:
    with np.load(path, allow_pickle=False) as data:
        return MatrixMeta(
            left_term=str(data["left_term"].item()),
            right_term=str(data["right_term"].item()),
            self_influence=bool(data["self_influence"].item()),
            scores_shape=tuple(int(value) for value in data["scores"].shape),
            candidate_points_shape=tuple(int(value) for value in data["candidate_points"].shape),
            num_pdes=int(data["num_pdes"].item()),
            num_bcs=int(data["num_bcs"].item()),
            n_outputs=int(data["n_outputs"].item()),
        )


def validate_selected_matrices(matrices: list[SourceMatrix], output_id: str) -> MatrixMeta:
    if not matrices:
        raise ValueError(f"{output_id}: no decomposition matrices selected")
    first_meta = read_matrix_meta(matrices[0].path)
    for matrix in matrices:
        meta = read_matrix_meta(matrix.path)
        if meta.left_term != output_id:
            raise ValueError(f"{matrix.path}: expected left_term {output_id}, got {meta.left_term}")
        if meta.right_term != matrix.term:
            raise ValueError(
                f"{matrix.path}: expected right_term {matrix.term}, got {meta.right_term}"
            )
        if meta.self_influence:
            raise ValueError(
                f"{matrix.path}: self-influence matrices are not valid Results sources"
            )
        if meta.scores_shape != first_meta.scores_shape:
            raise ValueError(
                f"{matrix.path}: score shape {meta.scores_shape} does not match {first_meta.scores_shape}"
            )
        if meta.candidate_points_shape != first_meta.candidate_points_shape:
            raise ValueError(
                f"{matrix.path}: candidate_points shape {meta.candidate_points_shape} "
                f"does not match {first_meta.candidate_points_shape}"
            )
        if (
            meta.num_pdes,
            meta.num_bcs,
            meta.n_outputs,
        ) != (first_meta.num_pdes, first_meta.num_bcs, first_meta.n_outputs):
            raise ValueError(
                f"{matrix.path}: loss/output counts do not match the other selected matrices"
            )
    return first_meta


def load_scores_and_points(path: Path) -> tuple[npt.NDArray[np.float32], MatrixArray]:
    with np.load(path, allow_pickle=False) as data:
        scores = np.asarray(data["scores"], dtype=np.float32)
        candidate_points = np.asarray(data["candidate_points"], dtype=np.float64)
    return scores, candidate_points


def compute_output(
    problem: str,
    output_id: str,
    matrices_by_term: dict[str, SourceMatrix],
    n_bins: int,
) -> OutputComputation:
    terms = selected_terms(set(matrices_by_term))
    if not terms:
        raise ValueError(f"{problem}/{output_id}: no PDE or boundary terms available")
    matrices = [matrices_by_term[term] for term in terms]
    meta = validate_selected_matrices(matrices, output_id)
    n_candidate, n_train = meta.scores_shape

    first_scores, candidate_points = load_scores_and_points(matrices[0].path)
    if first_scores.shape != meta.scores_shape:
        raise ValueError(f"{matrices[0].path}: score data shape changed while loading")

    term_abs_sums = np.zeros((len(matrices), n_candidate), dtype=np.float64)
    signed_total = np.zeros(meta.scores_shape, dtype=np.float32)
    abs_total = np.zeros(meta.scores_shape, dtype=np.float32)

    for term_index, matrix in enumerate(matrices):
        if term_index == 0:
            scores = first_scores
        else:
            scores, matrix_points = load_scores_and_points(matrix.path)
            if not np.array_equal(candidate_points, matrix_points):
                raise ValueError(
                    f"{matrix.path}: candidate_points do not match the first selected matrix"
                )
        if scores.shape != meta.scores_shape:
            raise ValueError(
                f"{matrix.path}: expected scores shape {meta.scores_shape}, got {scores.shape}"
            )
        abs_scores = np.abs(scores, dtype=np.float32)
        term_abs_sums[term_index] = abs_scores.sum(axis=1, dtype=np.float64)
        signed_total += scores
        abs_total += abs_scores

    denominator = term_abs_sums.sum(axis=0)
    fractions = np.divide(
        term_abs_sums,
        denominator,
        out=np.zeros_like(term_abs_sums),
        where=denominator[None, :] > 0,
    )
    row_coherence = rowwise_coherence(signed_total, abs_total)
    axis, axis_values = result_axis(problem, candidate_points)
    arrays = bin_rows(axis_values, fractions, row_coherence, n_bins)

    term_entries = [
        {
            "id": term,
            "label": term_label(problem, term),
            "mean_fraction": float(np.mean(fractions[term_index])),
            "std_fraction": float(np.std(fractions[term_index])),
        }
        for term_index, term in enumerate(terms)
    ]
    return OutputComputation(
        output_id=output_id,
        output_label=term_label(problem, output_id),
        axis=axis,
        terms=term_entries,
        mean_coherence=float(np.mean(row_coherence)),
        std_coherence=float(np.std(row_coherence)),
        n_candidate=n_candidate,
        n_train=n_train,
        source_matrix_ids=[matrix.matrix_id for matrix in matrices],
        arrays=arrays,
    )


def rowwise_coherence(
    signed_total: npt.NDArray[np.float32],
    abs_total: npt.NDArray[np.float32],
) -> FloatArray:
    result = np.empty(signed_total.shape[0], dtype=np.float64)
    chunk_rows = max(1, min(512, signed_total.shape[0]))
    for start in range(0, signed_total.shape[0], chunk_rows):
        stop = min(signed_total.shape[0], start + chunk_rows)
        numerator = np.abs(signed_total[start:stop], dtype=np.float32)
        denominator = abs_total[start:stop]
        ratios = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float32),
            where=denominator > 0,
        )
        result[start:stop] = ratios.mean(axis=1, dtype=np.float64)
    return result


def bin_rows(
    axis_values: FloatArray,
    fractions: FloatArray,
    coherence: FloatArray,
    n_bins: int,
) -> OutputArrays:
    if n_bins <= 0:
        raise ValueError("Number of bins must be positive")
    finite = np.isfinite(axis_values)
    if not np.any(finite):
        raise ValueError("Cannot bin Results data: axis values are all non-finite")
    min_value = float(np.min(axis_values[finite]))
    max_value = float(np.max(axis_values[finite]))
    if min_value == max_value:
        min_value -= 0.5
        max_value += 0.5
    edges = np.linspace(min_value, max_value, n_bins + 1, dtype=np.float64)
    centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)
    bin_indices = np.searchsorted(edges, axis_values, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    n_terms = fractions.shape[0]
    binned_fractions = np.zeros((n_terms, n_bins), dtype=np.float32)
    binned_fractions_std = np.zeros((n_terms, n_bins), dtype=np.float32)
    binned_coherence = np.empty(n_bins, dtype=np.float32)
    binned_coherence_std = np.zeros(n_bins, dtype=np.float32)
    global_mean_coherence = float(np.mean(coherence))
    for bin_index in range(n_bins):
        mask = finite & (bin_indices == bin_index)
        if not np.any(mask):
            binned_coherence[bin_index] = global_mean_coherence
            continue
        binned_fractions[:, bin_index] = np.mean(fractions[:, mask], axis=1, dtype=np.float64)
        binned_fractions_std[:, bin_index] = np.std(fractions[:, mask], axis=1, dtype=np.float64)
        binned_coherence[bin_index] = float(np.mean(coherence[mask]))
        binned_coherence_std[bin_index] = float(np.std(coherence[mask]))
    return OutputArrays(
        bin_centers=centers,
        binned_fractions=binned_fractions,
        binned_fractions_std=binned_fractions_std,
        binned_coherence=binned_coherence,
        binned_coherence_std=binned_coherence_std,
    )


def output_record(
    result: OutputComputation,
    relative_dir: str,
    output_dir: Path,
) -> dict[str, Any]:
    out_dir = output_dir / relative_dir
    arrays = {
        "bin_centers": write_f32(out_dir / "bin_centers.f32", result.arrays.bin_centers),
        "binned_fractions": write_f32(
            out_dir / "binned_fractions.f32", result.arrays.binned_fractions
        ),
        "binned_fractions_std": write_f32(
            out_dir / "binned_fractions_std.f32",
            result.arrays.binned_fractions_std,
        ),
        "binned_coherence": write_f32(
            out_dir / "binned_coherence.f32", result.arrays.binned_coherence
        ),
        "binned_coherence_std": write_f32(
            out_dir / "binned_coherence_std.f32",
            result.arrays.binned_coherence_std,
        ),
    }
    for spec in arrays.values():
        spec["path"] = f"{relative_dir}/{spec['path']}"
    return {
        "id": result.output_id,
        "label": result.output_label,
        "mean_coherence": result.mean_coherence,
        "std_coherence": result.std_coherence,
        "n_bins": int(result.arrays.bin_centers.shape[0]),
        "n_terms": len(result.terms),
        "n_candidate": result.n_candidate,
        "n_train": result.n_train,
        "source_matrix_ids": result.source_matrix_ids,
        "terms": result.terms,
        "arrays": arrays,
    }


def build_loss_decompositions(
    raw_data_dir: Path, output_dir: Path, n_bins: int
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for problem, quality, score_dir in discover_score_dirs(raw_data_dir):
        outputs = discover_output_matrices(score_dir)
        if not outputs:
            raise ValueError(f"{score_dir}: no output influence matrices found")
        output_results = [
            compute_output(problem, output_id, outputs[output_id], n_bins)
            for output_id in sorted(outputs, key=term_sort_key)
        ]
        axis = output_results[0].axis
        if any(result.axis != axis for result in output_results):
            raise ValueError(f"{score_dir}: outputs produced inconsistent axis metadata")
        output_records = [
            output_record(
                result,
                f"{problem}/{quality}/{result.output_id}",
                output_dir,
            )
            for result in output_results
        ]
        records.append(
            {
                "problem": problem,
                "display_name": display_problem_name(problem),
                "quality": quality,
                "source_kind": "full_matrix",
                "source_dir": score_dir.as_posix(),
                "axis": axis,
                "outputs": output_records,
            }
        )
    return sorted(records, key=lambda record: (record["problem"], record["quality"]))


def paper_indicators() -> dict[str, list[dict[str, Any]]]:
    temporal = [
        temporal_entry("diffusion", "Diffusion", 0.46, 0.33, 0.02, 0.26, 0.06, 0.41),
        temporal_entry("allen_cahn", "Allen Cahn", 0.43, 0.50, 0.02, 0.32, 0.05),
        temporal_entry("burgers", "Burgers", 0.43, 0.41, 0.02, 0.28, 0.02),
        temporal_entry("drift_diffusion", "Drift Diffusion", 0.46, 0.46, 0.04, 0.21, 0.06),
        temporal_entry("wave", "Wave", 0.43, 0.41, 0.03, 0.11, 0.02),
    ]
    directionality = [
        direction_entry(
            "poisson_disk", "Poisson Disk", "output_0", "u", 0.28, 0.29, 0.03, 0.69, 0.06
        ),
        direction_entry(
            "navier_stokes_nd",
            "Navier Stokes",
            "output_0",
            "x-velocity",
            0.48,
            0.15,
            0.01,
            0.15,
            0.03,
        ),
        direction_entry(
            "navier_stokes_nd",
            "Navier Stokes",
            "output_1",
            "y-velocity",
            0.48,
            0.25,
            0.01,
            0.20,
            0.05,
        ),
        direction_entry(
            "navier_stokes_nd",
            "Navier Stokes",
            "output_2",
            "pressure",
            0.48,
            0.14,
            0.02,
            0.11,
            0.02,
        ),
    ]
    return {"temporal": temporal, "directionality": directionality}


def temporal_entry(
    problem: str,
    display_name: str,
    baseline: float,
    good: float,
    good_std: float,
    bad: float,
    bad_std: float,
    bad_baseline: float | None = None,
) -> dict[str, Any]:
    entry = {
        "problem": problem,
        "display_name": display_name,
        "baseline": baseline,
        "values": {
            "good": {"mean": good, "std": good_std},
            "bad": {"mean": bad, "std": bad_std},
        },
    }
    if bad_baseline is not None:
        entry["bad_baseline"] = bad_baseline
        entry["note"] = "Poorly-Trained model uses a different sampling baseline."
    return entry


def direction_entry(
    problem: str,
    display_name: str,
    output_id: str,
    output_label: str,
    baseline: float,
    good: float,
    good_std: float,
    bad: float,
    bad_std: float,
) -> dict[str, Any]:
    return {
        "problem": problem,
        "display_name": display_name,
        "output_id": output_id,
        "output_label": output_label,
        "baseline": baseline,
        "values": {
            "good": {"mean": good, "std": good_std},
            "bad": {"mean": bad, "std": bad_std},
        },
    }


def main() -> None:
    args = parse_args()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sources": [args.raw_data_dir.as_posix()],
        "loss_decompositions": build_loss_decompositions(
            args.raw_data_dir, args.output_dir, args.bins
        ),
        "indicators": paper_indicators(),
    }
    out_path = args.output_dir / "index.json"
    write_json(out_path, payload)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
