"""Verify schema-v9 static PINNfluence demo artifacts against source influence files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 9
MODEL_QUALITIES = ("good", "bad")
BUNDLE_SIZE_BUDGET_BYTES = 750 * 1024 * 1024

DTYPES = {
    "float32": np.float32,
    "uint32": np.uint32,
    "uint16": np.uint16,
    "uint8": np.uint8,
    "int16": np.int16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=5, type=int)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "webdemo" / "public" / "data",
    )
    parser.add_argument(
        "--raw-data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "raw_data",
    )
    parser.add_argument("--bundle-size-budget-mb", default=750, type=int)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_array(base: Path, spec: dict[str, Any]) -> np.ndarray:
    path = base / spec["path"]
    if not path.exists():
        raise AssertionError(f"Missing array file: {path}")
    dtype = spec["dtype"]
    if dtype not in DTYPES:
        raise AssertionError(f"{path}: unsupported dtype {dtype!r}")
    arr = np.fromfile(path, dtype=DTYPES[dtype])
    expected = int(np.prod(spec["shape"]))
    if arr.size != expected:
        raise AssertionError(f"{path}: expected {expected} values, got {arr.size}")
    return arr.reshape(spec["shape"])


def deterministic_spread_indices(total: int, count: int, label: str) -> np.ndarray:
    if count < 0:
        raise AssertionError(f"{label} count must be >= 0")
    if count > total:
        raise AssertionError(f"{label} count {count} exceeds source count {total}")
    if count == 0:
        return np.arange(0, dtype=np.int64)
    return np.linspace(0, total - 1, count, dtype=np.int64)


def assert_schema(payload: dict[str, Any], path: Path) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError(f"{path}: schema_version must be {SCHEMA_VERSION}")


def verify_bundle_report(data_root: Path, budget_bytes: int) -> None:
    report_path = data_root / "bundle_report.json"
    if not report_path.exists():
        raise AssertionError(f"Missing bundle report: {report_path}")
    report = read_json(report_path)
    assert_schema(report, report_path)
    total = int(report.get("total_bytes", -1))
    if total < 0:
        raise AssertionError("bundle_report.json is missing total_bytes")
    if total > budget_bytes:
        raise AssertionError(
            f"Deployable bundle is {total / (1024 * 1024):.1f} MB, "
            f"above the {budget_bytes / (1024 * 1024):.1f} MB budget"
        )
    if not report.get("within_budget", False):
        raise AssertionError("bundle_report.json marks the bundle as over budget")


def verify_rasters(base: Path, manifest: dict[str, Any]) -> None:
    field_raster = manifest.get("field_raster")
    if manifest["fields"] and not field_raster:
        raise AssertionError("Fields are present but field_raster metadata is missing")
    if not field_raster:
        return

    height = int(field_raster["height"])
    width = int(field_raster["width"])
    shape = [height, width]
    if field_raster.get("shape") != shape:
        raise AssertionError(f"field_raster shape metadata {field_raster.get('shape')} != {shape}")
    mask = read_array(base, field_raster["mask"])
    if mask.dtype != np.uint8:
        raise AssertionError(f"field_raster mask must be uint8, got {mask.dtype}")
    if list(mask.shape) != shape:
        raise AssertionError(f"field_raster mask shape {list(mask.shape)} != {shape}")

    for field_id, field in manifest["fields"].items():
        if "array" in field:
            raise AssertionError(f"{field_id}: deprecated point field array is still present")
        raster_spec = field.get("raster")
        if not raster_spec:
            raise AssertionError(f"{field_id}: raster field data is missing")
        if raster_spec.get("dtype") != "uint16":
            raise AssertionError(f"{field_id}: raster dtype must be uint16")
        encoding = field.get("encoding")
        if encoding.get("kind") != "linear":
            raise AssertionError(f"{field_id}: raster encoding must be linear")
        if encoding.get("missing") != 65535:
            raise AssertionError(f"{field_id}: raster missing sentinel must be 65535")
        display_domain = field.get("display_domain")
        if not isinstance(display_domain, list) or len(display_domain) != 2:
            raise AssertionError(f"{field_id}: display_domain must be [min, max]")
        raster = read_array(base, raster_spec)
        if raster.dtype != np.uint16:
            raise AssertionError(f"{field_id}: raster dtype must be uint16, got {raster.dtype}")
        if list(raster.shape) != shape:
            raise AssertionError(f"{field_id}: raster shape {list(raster.shape)} != {shape}")


def verify_dense_scores(
    base: Path,
    matrix: dict[str, Any],
    expected_rows: int,
    n_train: int,
    raw_scores: np.ndarray | None,
    rows: np.ndarray,
) -> None:
    if "top_chunks" in matrix:
        raise AssertionError(f"{matrix['id']}: deprecated top_chunks metadata is still present")
    if "top" in matrix:
        raise AssertionError(f"{matrix['id']}: deprecated full top arrays are still present")
    if "row_chunk_size" in matrix:
        raise AssertionError(f"{matrix['id']}: deprecated row_chunk_size metadata is still present")
    if int(matrix.get("row_count", -1)) != expected_rows:
        raise AssertionError(
            f"{matrix['id']}: row_count {matrix.get('row_count')} != {expected_rows}"
        )
    if int(matrix.get("k", 0)) < 1 or int(matrix.get("k", 0)) > n_train:
        raise AssertionError(f"{matrix['id']}: k must be in [1, n_train]")

    expected_shape = [expected_rows, n_train]
    if matrix.get("scores_shape") != expected_shape:
        raise AssertionError(
            f"{matrix['id']}: scores_shape {matrix.get('scores_shape')} != {expected_shape}"
        )
    layout = matrix.get("score_layout")
    if not isinstance(layout, dict) or layout.get("kind") != "dense_row_major":
        raise AssertionError(f"{matrix['id']}: score_layout must be dense_row_major")
    expected_row_stride = n_train * np.dtype(np.float32).itemsize
    if int(layout.get("row_stride_bytes", -1)) != expected_row_stride:
        raise AssertionError(f"{matrix['id']}: invalid row_stride_bytes")
    if int(layout.get("data_offset_bytes", -1)) != 0:
        raise AssertionError(f"{matrix['id']}: data_offset_bytes must be 0")

    scores_spec = matrix.get("scores")
    if not isinstance(scores_spec, dict):
        raise AssertionError(f"{matrix['id']}: scores array metadata is missing")
    if scores_spec.get("dtype") != "float32":
        raise AssertionError(f"{matrix['id']}: dense scores must be float32")
    if scores_spec.get("shape") != expected_shape:
        raise AssertionError(f"{matrix['id']}: scores array shape mismatch")
    expected_bytes = expected_rows * expected_row_stride
    if int(scores_spec.get("bytes", -1)) != expected_bytes:
        raise AssertionError(f"{matrix['id']}: scores bytes metadata mismatch")

    scores_path = base / scores_spec["path"]
    if not scores_path.exists():
        raise AssertionError(f"Missing scores file: {scores_path}")
    if scores_path.stat().st_size != expected_bytes:
        raise AssertionError(f"{matrix['id']}: scores file size mismatch")
    scores = read_array(base, scores_spec)
    if scores.dtype != np.float32:
        raise AssertionError(f"{matrix['id']}: scores dtype must be float32, got {scores.dtype}")

    if raw_scores is None:
        return

    for row in rows:
        row_int = int(row)
        if not np.allclose(scores[row_int], raw_scores[row_int], rtol=1e-5, atol=1e-8):
            raise AssertionError(f"{matrix['id']} row {row_int}: dense scores mismatch")


def source_file_for_matrix(
    manifest: dict[str, Any], matrix: dict[str, Any], raw_data_root: Path
) -> Path:
    if matrix.get("source_file"):
        return Path(matrix["source_file"])
    return (
        raw_data_root
        / manifest["folder"]
        / f"{manifest['run_id']}_influence_scores"
        / f"{matrix['id']}.npz"
    )


def verify_matrix_is_supported(matrix: dict[str, Any]) -> None:
    matrix_id = str(matrix.get("id", ""))
    method = str(matrix.get("method", ""))
    if matrix_id.startswith(("grad_dot", "graddot")) or method.lower() in {
        "graddot",
        "grad_dot",
    }:
        raise AssertionError(f"{matrix_id}: GradDot matrices are not supported")


def verify_topk(
    manifest_path: Path,
    samples: int,
    raw_data_root: Path | None = None,
) -> None:
    if raw_data_root is None:
        raw_data_root = Path(__file__).resolve().parent.parent / "raw_data"
    base = manifest_path.parent
    manifest = read_json(manifest_path)
    assert_schema(manifest, manifest_path)
    for key in ("problem", "display_name", "model_quality", "folder", "run_id"):
        if not manifest.get(key):
            raise AssertionError(f"{manifest_path}: manifest is missing {key}")
    if manifest["model_quality"] not in MODEL_QUALITIES:
        raise AssertionError(
            f"{manifest_path}: model_quality must be one of {', '.join(MODEL_QUALITIES)}"
        )
    n_train = manifest["n_train"]
    n_candidate = manifest["n_candidate"]
    source_n_train = int(manifest.get("source_n_train", n_train))
    source_n_candidate = int(manifest.get("source_n_candidate", n_candidate))
    point_selection = manifest.get("point_selection", "deterministic_spread")
    if point_selection != "deterministic_spread":
        raise AssertionError(f"Unsupported point_selection {point_selection!r}")

    candidate_points = read_array(base, manifest["arrays"]["candidate_points"])
    train_points = read_array(base, manifest["arrays"]["train_points"])
    assert candidate_points.shape[0] == n_candidate
    assert train_points.shape[0] == n_train
    train_indices = deterministic_spread_indices(source_n_train, n_train, "train")
    candidate_indices = deterministic_spread_indices(source_n_candidate, n_candidate, "candidate")

    deprecated_arrays = {
        "display_points",
        "display_to_candidate",
        "display_to_train",
        "train_bc_id",
        "train_kind",
    }
    found_deprecated = sorted(deprecated_arrays & set(manifest["arrays"]))
    if found_deprecated:
        raise AssertionError(f"Deprecated arrays are still present: {found_deprecated}")
    if "n_display" in manifest:
        raise AssertionError("Deprecated n_display metadata is still present")
    if "row_chunk_size" in manifest:
        raise AssertionError("Deprecated row_chunk_size metadata is still present")

    verify_rasters(base, manifest)

    for matrix in manifest["influence_matrices"]:
        verify_matrix_is_supported(matrix)
        if "summary" in matrix:
            raise AssertionError(f"{matrix['id']}: deprecated summary metadata is still present")
        row_source = matrix["row_source"]
        if row_source == "candidate_points":
            expected_rows = n_candidate
        elif row_source == "train_points":
            expected_rows = n_train
        else:
            raise AssertionError(f"{matrix['id']}: unknown row_source {row_source!r}")
        source_rows = source_n_train if row_source == "train_points" else source_n_candidate
        source_row_indices = train_indices if row_source == "train_points" else candidate_indices

        source = source_file_for_matrix(manifest, matrix, raw_data_root)
        raw_scores = None
        if not source.exists():
            print(f"  source missing, shape-only check: {matrix['id']}")
        else:
            with np.load(source, allow_pickle=False) as raw:
                source_scores = raw["scores"].astype(np.float32)
            if source_scores.shape[0] != source_rows:
                raise AssertionError(
                    f"{matrix['id']}: source rows {source_scores.shape[0]} != {source_rows}"
                )
            if source_scores.shape[1] != source_n_train:
                raise AssertionError(
                    f"{matrix['id']}: source columns {source_scores.shape[1]} != {source_n_train}"
                )
            raw_scores = source_scores[np.ix_(source_row_indices, train_indices)] / float(
                source_n_train
            )
            if raw_scores.shape[0] != expected_rows:
                raise AssertionError(
                    f"{matrix['id']}: sliced rows {raw_scores.shape[0]} != {expected_rows}"
                )
            if raw_scores.shape[1] != n_train:
                raise AssertionError(
                    f"{matrix['id']}: sliced columns {raw_scores.shape[1]} != {n_train}"
                )

        rows = np.linspace(0, expected_rows - 1, min(samples, expected_rows)).astype(int)
        verify_dense_scores(base, matrix, expected_rows, n_train, raw_scores, rows)


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    budget_bytes = int(args.bundle_size_budget_mb) * 1024 * 1024
    index_path = data_root / "index.json"
    index = read_json(index_path)
    assert_schema(index, index_path)
    if not isinstance(index.get("problems"), list):
        raise AssertionError(f"{index_path}: index.json is missing problems[]")
    if "runs" in index:
        raise AssertionError(f"{index_path}: deprecated runs[] index is still present")
    if "row_chunk_size" in index:
        raise AssertionError(f"{index_path}: deprecated row_chunk_size metadata is still present")
    verify_bundle_report(data_root, budget_bytes)
    checked = 0
    for problem in index["problems"]:
        variants = problem.get("variants")
        if not isinstance(variants, dict):
            raise AssertionError(f"{problem.get('problem', '<unknown>')}: missing variants")
        for quality in MODEL_QUALITIES:
            variant = variants.get(quality)
            if not isinstance(variant, dict):
                raise AssertionError(
                    f"{problem.get('problem', '<unknown>')}: missing {quality} variant"
                )
            manifest_rel = variant.get("manifest")
            if not manifest_rel:
                raise AssertionError(
                    f"{problem.get('problem', '<unknown>')}: {quality} variant has no manifest"
                )
            if variant.get("model_quality") != quality:
                raise AssertionError(
                    f"{problem.get('problem', '<unknown>')}: {quality} variant has "
                    f"model_quality {variant.get('model_quality')!r}"
                )
            manifest_path = data_root / manifest_rel
            print(f"Checking {manifest_path}")
            verify_topk(manifest_path, args.samples, args.raw_data_root)
            checked += 1
    print(f"Verified {checked} static demo run(s)")


if __name__ == "__main__":
    main()
