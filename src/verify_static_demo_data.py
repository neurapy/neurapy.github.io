"""Verify static PINNfluence demo artifacts against source influence files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DTYPES = {
    "float32": np.float32,
    "uint32": np.uint32,
    "uint8": np.uint8,
    "int16": np.int16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", default=5, type=int)
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_array(base: Path, spec: dict) -> np.ndarray:
    path = base / spec["path"]
    if not path.exists():
        raise AssertionError(f"Missing array file: {path}")
    arr = np.fromfile(path, dtype=DTYPES[spec["dtype"]])
    expected = int(np.prod(spec["shape"]))
    if arr.size != expected:
        raise AssertionError(f"{path}: expected {expected} values, got {arr.size}")
    return arr.reshape(spec["shape"])


def verify_topk(manifest_path: Path, samples: int) -> None:
    base = manifest_path.parent
    manifest = read_json(manifest_path)
    n_train = manifest["n_train"]
    n_candidate = manifest["n_candidate"]

    candidate_points = read_array(base, manifest["arrays"]["candidate_points"])
    train_points = read_array(base, manifest["arrays"]["train_points"])
    assert candidate_points.shape[0] == n_candidate
    assert train_points.shape[0] == n_train

    deprecated_arrays = {"display_points", "display_to_candidate", "display_to_train"}
    found_deprecated = sorted(deprecated_arrays & set(manifest["arrays"]))
    if found_deprecated:
        raise AssertionError(f"Deprecated display arrays are still present: {found_deprecated}")
    if "n_display" in manifest:
        raise AssertionError("Deprecated n_display metadata is still present")

    field_raster = manifest.get("field_raster")
    if manifest["fields"] and not field_raster:
        raise AssertionError("Fields are present but field_raster metadata is missing")
    if field_raster:
        height = int(field_raster["height"])
        width = int(field_raster["width"])
        shape = [height, width]
        if field_raster.get("shape") != shape:
            raise AssertionError(
                f"field_raster shape metadata {field_raster.get('shape')} != {shape}"
            )
        mask = read_array(base, field_raster["mask"])
        if list(mask.shape) != shape:
            raise AssertionError(f"field_raster mask shape {list(mask.shape)} != {shape}")
        if mask.size != height * width:
            raise AssertionError(f"field_raster mask length {mask.size} != {height} * {width}")
        for field_id, field in manifest["fields"].items():
            if "array" in field:
                raise AssertionError(f"{field_id}: deprecated point field array is still present")
            if "raster" not in field:
                raise AssertionError(f"{field_id}: raster field data is missing")
            raster = read_array(base, field["raster"])
            if list(raster.shape) != shape:
                raise AssertionError(f"{field_id}: raster shape {list(raster.shape)} != {shape}")
            if raster.size != height * width:
                raise AssertionError(
                    f"{field_id}: raster length {raster.size} != {height} * {width}"
                )

    for matrix in manifest["influence_matrices"]:
        row_source = matrix.get(
            "row_source",
            "train_points" if matrix.get("self_influence") else "candidate_points",
        )
        if row_source == "candidate_points":
            expected_rows = n_candidate
        elif row_source == "train_points":
            expected_rows = n_train
        else:
            raise AssertionError(f"{matrix['id']}: unknown row_source {row_source!r}")

        source = Path(matrix["source_file"])
        raw_scores = None
        if not source.exists():
            print(f"  source missing, shape-only check: {matrix['id']}")
        else:
            with np.load(source, allow_pickle=False) as raw:
                raw_scores = -raw["scores"].astype(np.float32) / float(n_train)
            if raw_scores.shape[0] != expected_rows:
                raise AssertionError(
                    f"{matrix['id']}: source rows {raw_scores.shape[0]} != {expected_rows}"
                )
            if raw_scores.shape[1] != n_train:
                raise AssertionError(
                    f"{matrix['id']}: source columns {raw_scores.shape[1]} != {n_train}"
                )

        rows = np.linspace(0, expected_rows - 1, min(samples, expected_rows))
        rows = rows.astype(int)

        for mode in ("abs", "pos", "neg"):
            indices = read_array(base, matrix["top"][mode]["indices"])
            values = read_array(base, matrix["top"][mode]["values"])
            assert indices.shape == values.shape
            assert indices.shape[0] == expected_rows
            if indices.size:
                assert int(indices.max()) < n_train

            if raw_scores is None:
                continue
            for row in rows:
                k = indices.shape[1]
                row_scores = raw_scores[row]
                if mode == "abs":
                    rank_scores = np.abs(row_scores)
                elif mode == "pos":
                    rank_scores = row_scores
                else:
                    rank_scores = -row_scores
                got = indices[row]

                if len(np.unique(got)) != len(got):
                    raise AssertionError(f"{matrix['id']} {mode} row {row}: duplicate top-k index")
                threshold = np.sort(rank_scores)[::-1][k - 1]
                if np.any(rank_scores[got] < threshold - 1e-8):
                    raise AssertionError(f"{matrix['id']} {mode} row {row}: top-k mismatch")
                if np.any(np.diff(rank_scores[got]) > 1e-8):
                    raise AssertionError(f"{matrix['id']} {mode} row {row}: top-k order mismatch")

                expected_values = row_scores[got]
                if not np.allclose(values[row], expected_values, rtol=1e-5, atol=1e-8):
                    raise AssertionError(f"{matrix['id']} {mode} row {row}: value mismatch")


def main() -> None:
    args = parse_args()
    webdata_root = Path(__file__).resolve().parent.parent / "webdemo" / "data"
    index_path = webdata_root / "index.json"
    index = read_json(index_path)
    checked = 0
    for run in index["runs"]:
        manifest_rel = run.get("manifest")
        if not manifest_rel:
            continue
        manifest_path = webdata_root / manifest_rel
        print(f"Checking {manifest_path}")
        verify_topk(manifest_path, args.samples)
        checked += 1
    print(f"Verified {checked} static demo run(s)")


if __name__ == "__main__":
    main()
