from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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
        "arrays/loss_total_raster.f32",
        np.arange(4, dtype=np.float32).reshape(2, 2),
        "float32",
    )
    return {
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
            }
        },
        "influence_matrices": [],
    }


def test_verify_accepts_raster_only_fields(tmp_path: Path) -> None:
    manifest_path = write_manifest(tmp_path, lean_manifest(tmp_path))

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
