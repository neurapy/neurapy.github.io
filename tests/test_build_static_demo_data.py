from __future__ import annotations

import numpy as np
import pytest

build_static = pytest.importorskip("build_static_demo_data")


def test_raster_dimensions_preserve_physical_aspect_ratio() -> None:
    mins = np.array([0.0, 0.0])
    maxs = np.array([4.0, 2.0])

    assert build_static.raster_dimensions(mins, maxs, max_axis_resolution=512) == (512, 256)

    mins = np.array([0.0, 0.0])
    maxs = np.array([2.0, 4.0])

    assert build_static.raster_dimensions(mins, maxs, max_axis_resolution=512) == (256, 512)


def test_raster_points_are_row_major_pixel_centers_with_descending_y() -> None:
    mins = np.array([0.0, 10.0])
    maxs = np.array([4.0, 12.0])

    points = build_static.raster_points_from_bounds(mins, maxs, width=4, height=2)

    np.testing.assert_allclose(
        points,
        np.array(
            [
                [0.5, 11.5],
                [1.5, 11.5],
                [2.5, 11.5],
                [3.5, 11.5],
                [0.5, 10.5],
                [1.5, 10.5],
                [2.5, 10.5],
                [3.5, 10.5],
            ]
        ),
    )


def test_raster_grid_metadata_shape_and_disk_mask() -> None:
    class UnitDisk:
        bbox = (np.array([-1.0, -1.0]), np.array([1.0, 1.0]))

        def inside(self, points: np.ndarray) -> np.ndarray:
            return np.sum(points**2, axis=1) <= 1.0

    grid = build_static.make_raster_grid(
        UnitDisk(),
        fallback_points=np.empty((0, 2)),
        max_axis_resolution=4,
    )

    assert grid is not None
    assert grid.width == 4
    assert grid.height == 4
    assert grid.axes == ["x", "y"]
    assert grid.bounds == {"x": [-1.0, 1.0], "y": [-1.0, 1.0]}
    assert grid.points.shape == (16, 2)
    assert grid.mask.shape == (4, 4)
    assert grid.mask.dtype == np.uint8
    assert grid.mask[0, 0] == 0
    assert grid.mask[0, 3] == 0
    assert grid.mask[1, 1] == 1
    assert grid.mask[2, 2] == 1


def test_uint16_raster_quantization_uses_missing_sentinel() -> None:
    values = np.array([[0.0, 0.5], [1.0, np.nan]], dtype=np.float32)
    mask = np.array([[1, 1], [1, 0]], dtype=np.uint8)

    quantized, encoding, display_domain = build_static.quantize_uint16_linear(values, mask)

    assert quantized.dtype == np.uint16
    assert quantized.shape == values.shape
    assert quantized[0, 0] == 0
    assert quantized[1, 0] == 65534
    assert quantized[1, 1] == 65535
    assert encoding == {"kind": "linear", "min": 0.0, "max": 1.0, "missing": 65535}
    assert display_domain[0] < display_domain[1]


def test_int16_symmetric_quantization_round_trips_with_scale() -> None:
    values = np.array([[-2.0, 0.0, 1.0]], dtype=np.float32)

    quantized, scale = build_static.quantize_int16_symmetric(values)
    restored = quantized.astype(np.float32) * scale

    assert quantized.dtype == np.int16
    assert scale > 0
    np.testing.assert_allclose(restored, values, atol=scale * 0.55)


def test_deterministic_spread_indices_cover_full_source_range() -> None:
    np.testing.assert_array_equal(
        build_static.deterministic_spread_indices(10, 4, "candidate"),
        np.array([0, 3, 6, 9], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        build_static.deterministic_spread_indices(5, None, "train"),
        np.arange(5, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        build_static.deterministic_spread_indices(8, 1, "train"),
        np.array([0], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        build_static.deterministic_spread_indices(3, 10_000, "train"),
        np.arange(3, dtype=np.int64),
    )
    np.testing.assert_array_equal(
        build_static.deterministic_spread_indices(0, 10_000, "candidate"),
        np.array([], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="count must be >= 1"):
        build_static.deterministic_spread_indices(3, 0, "train")


def write_influence_npz(
    path,
    scores: np.ndarray,
    candidate_points: np.ndarray,
    self_influence: bool = False,
) -> None:
    np.savez_compressed(
        path,
        scores=scores.astype(np.float32),
        candidate_points=candidate_points.astype(np.float64),
        num_pdes=1,
        num_bcs=0,
        n_outputs=1,
        left_term="output_0",
        right_term="total_loss",
        self_influence=self_influence,
    )


def test_process_influence_matrix_subsets_candidate_rows_and_train_columns(tmp_path) -> None:
    matrix_path = tmp_path / "matrix.npz"
    scores = np.arange(30, dtype=np.float32).reshape(5, 6)
    candidate_points = np.column_stack([np.arange(5), np.arange(5) + 0.5])
    write_influence_npz(matrix_path, scores, candidate_points)

    metadata = build_static.process_influence_matrix(
        matrix_path,
        out_dir=tmp_path,
        rel_prefix="influence",
        n_train=3,
        source_n_train=6,
        train_indices=np.array([0, 2, 5], dtype=np.int64),
        row_source="candidate_points",
        row_count=3,
        row_indices=np.array([0, 2, 4], dtype=np.int64),
        max_local_influence_points=2,
        row_chunk_size=2,
    )

    assert metadata["scores_shape"] == [3, 3]
    assert metadata["source_scores_shape"] == [5, 6]
    assert metadata["candidate_points_shape"] == [3, 2]
    assert metadata["source_candidate_points_shape"] == [5, 2]
    summary_spec = metadata["summary"]["mean_signed"]
    summary = np.fromfile(tmp_path / summary_spec["path"], dtype=np.float32)
    assert list(summary.shape) == [3]
    chunk = metadata["top_chunks"]["abs"]["chunks"][0]
    indices = np.fromfile(tmp_path / chunk["indices"]["path"], dtype=np.uint16)
    assert int(indices.max()) < 3


def test_process_influence_matrix_subsets_self_influence_rows_and_columns(tmp_path) -> None:
    matrix_path = tmp_path / "self_matrix.npz"
    scores = np.arange(36, dtype=np.float32).reshape(6, 6)
    train_points = np.column_stack([np.arange(6), np.arange(6) + 0.25])
    train_indices = np.array([0, 3, 5], dtype=np.int64)
    write_influence_npz(matrix_path, scores, train_points, self_influence=True)

    metadata = build_static.process_influence_matrix(
        matrix_path,
        out_dir=tmp_path,
        rel_prefix="influence",
        n_train=3,
        source_n_train=6,
        train_indices=train_indices,
        row_source="train_points",
        row_count=3,
        row_indices=train_indices,
        max_local_influence_points=2,
        row_chunk_size=4,
    )

    assert metadata["scores_shape"] == [3, 3]
    assert metadata["source_scores_shape"] == [6, 6]
    assert metadata["candidate_points_shape"] == [3, 2]
    chunk = metadata["top_chunks"]["pos"]["chunks"][0]
    assert chunk["row_count"] == 3


def test_bundle_report_groups_chunk_files(tmp_path) -> None:
    (tmp_path / "index.json").write_text("{}")
    chunk = tmp_path / "run" / "influence" / "m0" / "abs" / "chunks" / "0_values.i16"
    chunk.parent.mkdir(parents=True)
    np.array([1, 2, 3], dtype=np.int16).tofile(chunk)

    report = build_static.build_bundle_report(tmp_path, budget_bytes=10_000)

    assert report["schema_version"] == 5
    assert report["within_budget"] is True
    assert report["by_kind"]["influence_chunks"] == 6
