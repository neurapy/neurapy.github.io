from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

build_static = pytest.importorskip("build_static_demo_data")
verify_static = pytest.importorskip("verify_static_demo_data")


def write_array(base: Path, rel_path: str, values: np.ndarray, dtype: str) -> dict[str, Any]:
    path = base / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    typed = values.astype(verify_static.DTYPES[dtype], copy=False)
    typed.tofile(path)
    return {"path": rel_path, "dtype": dtype, "shape": list(typed.shape)}


def write_manifest(base: Path, manifest: dict[str, Any]) -> Path:
    path = base / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def lean_manifest(base: Path) -> dict[str, Any]:
    candidate_points = write_array(
        base,
        "arrays/candidate_points.f32",
        np.array([[0.25, 0.75]], dtype=np.float32),
        "float32",
    )
    train_points = write_array(
        base,
        "arrays/train_points.f32",
        np.array([[0.5, 0.5]], dtype=np.float32),
        "float32",
    )
    mask = write_array(
        base,
        "arrays/field_raster_mask.u8",
        np.ones((2, 2), dtype=np.uint8),
        "uint8",
    )
    raster = write_array(
        base,
        "arrays/loss_total_raster.u16",
        np.arange(4, dtype=np.uint16).reshape(2, 2),
        "uint16",
    )
    return {
        "schema_version": 6,
        "n_candidate": 1,
        "n_train": 1,
        "arrays": {
            "candidate_points": candidate_points,
            "train_points": train_points,
        },
        "field_raster": {
            "width": 2,
            "height": 2,
            "shape": [2, 2],
            "mask": mask,
        },
        "fields": {
            "loss_total": {
                "label": "Loss",
                "kind": "loss",
                "raster": raster,
                "encoding": {
                    "kind": "linear",
                    "min": 0.0,
                    "max": 1.0,
                    "missing": 65535,
                },
                "display_domain": [0.0, 1.0],
            }
        },
        "influence_matrices": [],
    }


def test_verify_accepts_raster_only_fields(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path, lean_manifest(tmp_path))

    verify_static.verify_topk(manifest_path, samples=1)


def test_verify_rejects_schema_v5_manifest(tmp_path: Path) -> None:
    manifest = lean_manifest(tmp_path)
    manifest["schema_version"] = 5
    manifest_path = write_manifest(tmp_path, manifest)

    with pytest.raises(AssertionError, match="schema_version"):
        verify_static.verify_topk(manifest_path, samples=1)


def test_verify_rejects_deprecated_display_arrays(tmp_path: Path) -> None:
    manifest = lean_manifest(tmp_path)
    manifest["n_display"] = 1
    manifest["arrays"]["display_points"] = write_array(
        tmp_path,
        "arrays/display_points.f32",
        np.array([[0.25, 0.75]], dtype=np.float32),
        "float32",
    )
    manifest_path = write_manifest(tmp_path, manifest)

    with pytest.raises(AssertionError, match="Deprecated display arrays"):
        verify_static.verify_topk(manifest_path, samples=1)


def test_verify_rejects_field_without_raster(tmp_path: Path) -> None:
    manifest = lean_manifest(tmp_path)
    del manifest["fields"]["loss_total"]["raster"]
    manifest_path = write_manifest(tmp_path, manifest)

    with pytest.raises(AssertionError, match="raster field data is missing"):
        verify_static.verify_topk(manifest_path, samples=1)


def test_verify_rejects_deprecated_point_field_array(tmp_path: Path) -> None:
    manifest = lean_manifest(tmp_path)
    manifest["fields"]["loss_total"]["array"] = write_array(
        tmp_path,
        "arrays/loss_total.f32",
        np.array([1.0], dtype=np.float32),
        "float32",
    )
    manifest_path = write_manifest(tmp_path, manifest)

    with pytest.raises(AssertionError, match="deprecated point field array"):
        verify_static.verify_topk(manifest_path, samples=1)


def test_verify_rejects_float32_raster(tmp_path: Path) -> None:
    manifest = lean_manifest(tmp_path)
    manifest["fields"]["loss_total"]["raster"] = write_array(
        tmp_path,
        "arrays/loss_total_raster.f32",
        np.arange(4, dtype=np.float32).reshape(2, 2),
        "float32",
    )
    manifest_path = write_manifest(tmp_path, manifest)

    with pytest.raises(AssertionError, match="raster dtype must be uint16"):
        verify_static.verify_topk(manifest_path, samples=1)


def test_verify_rejects_graddot_matrix(tmp_path: Path) -> None:
    manifest = lean_manifest(tmp_path)
    manifest["influence_matrices"] = [
        {
            "id": "grad_dot_total_loss_total_loss",
            "method": "GradDot",
        }
    ]
    manifest_path = write_manifest(tmp_path, manifest)

    with pytest.raises(AssertionError, match="GradDot matrices are not supported"):
        verify_static.verify_topk(manifest_path, samples=1)


def test_verify_rejects_deprecated_summary_metadata(tmp_path: Path) -> None:
    manifest = lean_manifest(tmp_path)
    manifest["influence_matrices"] = [
        {
            "id": "m0",
            "method": "PINNfluence",
            "summary": {},
        }
    ]
    manifest_path = write_manifest(tmp_path, manifest)

    with pytest.raises(AssertionError, match="deprecated summary metadata"):
        verify_static.verify_topk(manifest_path, samples=1)


def test_verify_slices_source_matrix_for_downsampled_manifest(tmp_path: Path) -> None:
    base = tmp_path / "data" / "folder" / "run"
    base.mkdir(parents=True)
    raw_root = tmp_path / "raw"
    raw_matrix = raw_root / "folder" / "run_influence_scores" / "matrix.npz"
    raw_matrix.parent.mkdir(parents=True)

    source_scores = np.arange(30, dtype=np.float32).reshape(5, 6)
    source_candidates = np.column_stack([np.arange(5), np.arange(5) + 0.5])
    np.savez_compressed(
        raw_matrix,
        scores=source_scores,
        candidate_points=source_candidates,
        num_pdes=1,
        num_bcs=0,
        n_outputs=1,
        left_term="output_0",
        right_term="total_loss",
        self_influence=False,
    )

    candidate_indices = build_static.deterministic_spread_indices(5, 3, "candidate")
    train_indices = build_static.deterministic_spread_indices(6, 3, "train")
    matrix = build_static.process_influence_matrix(
        raw_matrix,
        out_dir=base,
        rel_prefix="influence",
        n_train=3,
        source_n_train=6,
        train_indices=train_indices,
        row_source="candidate_points",
        row_count=3,
        row_indices=candidate_indices,
        max_local_influence_points=2,
        row_chunk_size=2,
    )

    candidate_points = source_candidates[candidate_indices].astype(np.float32)
    train_points = np.column_stack([np.arange(6), np.arange(6) + 1.0])[train_indices]
    manifest = {
        "schema_version": 6,
        "folder": "folder",
        "run_id": "run",
        "n_candidate": 3,
        "n_train": 3,
        "source_n_candidate": 5,
        "source_n_train": 6,
        "point_selection": "deterministic_spread",
        "arrays": {
            "candidate_points": write_array(
                base, "arrays/candidate_points.f32", candidate_points, "float32"
            ),
            "train_points": write_array(base, "arrays/train_points.f32", train_points, "float32"),
        },
        "fields": {},
        "influence_matrices": [matrix],
    }
    manifest_path = write_manifest(base, manifest)

    verify_static.verify_topk(manifest_path, samples=2, raw_data_root=raw_root)
