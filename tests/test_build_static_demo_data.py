from __future__ import annotations

from pathlib import Path

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


def test_problem_from_folder_accepts_good_bad_raw_data_suffixes() -> None:
    assert build_static.problem_from_folder(Path("allen_cahn_float64")) == "allen_cahn"
    assert build_static.problem_from_folder(Path("allen_cahn_float64_good")) == "allen_cahn"
    assert build_static.problem_from_folder(Path("allen_cahn_float64_bad")) == "allen_cahn"


def test_infer_model_quality_requires_good_bad_suffixes() -> None:
    assert build_static.infer_model_quality(Path("allen_cahn_float64_good")) == "good"
    assert build_static.infer_model_quality(Path("allen_cahn_float64_bad")) == "bad"
    assert build_static.infer_model_quality(Path("allen_cahn_float64")) is None


def test_discover_runs_includes_good_bad_and_ignores_legacy_folders(tmp_path) -> None:
    legacy_folder = tmp_path / "allen_cahn_float64"
    good_folder = tmp_path / "allen_cahn_float64_good"
    bad_folder = tmp_path / "allen_cahn_float64_bad"
    ignored_folder = tmp_path / "allen_cahn_good"
    legacy_folder.mkdir()
    good_folder.mkdir()
    bad_folder.mkdir()
    ignored_folder.mkdir()

    legacy_prefix = "allen_cahn_adam_100000_adam_25000_lbfgs_2500_domain_500_boundary_500_initial_3_x_64_hidden_float64_True_9_soft"
    good_prefix = "allen_cahn_adam_100000_adam_25000_lbfgs_2500_domain_500_boundary_500_initial_3_x_64_hidden_float64_True_0_soft"
    bad_prefix = "allen_cahn_adam_100000_adam_0_lbfgs_2500_domain_500_boundary_500_initial_3_x_64_hidden_float64_True_0_soft"
    (legacy_folder / f"{legacy_prefix}_full.pt").touch()
    (good_folder / f"{good_prefix}_full.pt").touch()
    (bad_folder / f"{bad_prefix}_influence_scores").mkdir()

    runs = build_static.discover_runs(tmp_path)

    assert [(run.folder.name, run.problem, run.model_quality, run.run_prefix) for run in runs] == [
        ("allen_cahn_float64_bad", "allen_cahn", "bad", bad_prefix),
        ("allen_cahn_float64_good", "allen_cahn", "good", good_prefix),
    ]


def test_grouped_schema_v7_index_entries_require_good_and_bad_variants() -> None:
    good = {
        "problem": "navier_stokes_nd",
        "display_name": "Navier Stokes",
        "model_quality": "good",
        "folder": "navier_stokes_nd_float64_good",
        "run_id": "good_run",
        "manifest": "navier_stokes_nd_float64_good/good_run/manifest.json",
    }
    bad = {
        **good,
        "model_quality": "bad",
        "folder": "navier_stokes_nd_float64_bad",
        "run_id": "bad_run",
        "manifest": "navier_stokes_nd_float64_bad/bad_run/manifest.json",
    }

    problems = build_static.build_problem_index_entries([good, bad])

    assert problems == [
        {
            "problem": "navier_stokes_nd",
            "display_name": "Navier Stokes",
            "variants": {"good": good, "bad": bad},
        }
    ]

    with pytest.raises(ValueError, match="missing required model variant\\(s\\): bad"):
        build_static.build_problem_index_entries([good])

    with pytest.raises(ValueError, match="failed to produce manifest\\(s\\) for: bad"):
        build_static.build_problem_index_entries([{**good}, {**bad, "manifest": None}])


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


def test_read_npz_npy_header_reads_scores_shape_and_dtype(tmp_path) -> None:
    matrix_path = tmp_path / "matrix.npz"
    write_influence_npz(
        matrix_path,
        scores=np.zeros((4, 7), dtype=np.float32),
        candidate_points=np.zeros((4, 2), dtype=np.float64),
    )

    header = build_static.read_npz_npy_header(matrix_path, "scores")
    metadata = build_static.load_matrix_metadata(matrix_path)

    assert header == {
        "shape": (4, 7),
        "fortran_order": False,
        "dtype": np.dtype("float32"),
    }
    assert metadata["scores_shape"] == [4, 7]
    assert metadata["candidate_points"].shape == (4, 2)


def test_candidate_influence_files_excludes_graddot_and_limits_core(tmp_path) -> None:
    influence_dir = tmp_path / "run_influence_scores"
    influence_dir.mkdir()
    for name in [
        "influences_total_loss_output_0.npz",
        "influences_total_loss_output_1.npz",
        "influences_total_loss_output_2.npz",
        "influences_total_loss_total_loss.npz",
        "influences_bc_loss_total_loss.npz",
        "grad_dot_total_loss_output_0.npz",
        "graddot_total_loss_total_loss.npz",
    ]:
        (influence_dir / name).touch()

    run = build_static.RunPaths(
        folder=tmp_path,
        problem="fixture",
        model_quality="good",
        run_prefix="run",
        checkpoint=None,
        influence_dir=influence_dir,
        validation_dir=None,
    )

    assert {path.stem for path in build_static.candidate_influence_files(run, "core")} == {
        "influences_total_loss_output_0",
        "influences_total_loss_output_1",
        "influences_total_loss_output_2",
        "influences_total_loss_total_loss",
    }
    assert {path.stem for path in build_static.candidate_influence_files(run, "all")} == {
        "influences_total_loss_output_0",
        "influences_total_loss_output_1",
        "influences_total_loss_output_2",
        "influences_total_loss_total_loss",
        "influences_bc_loss_total_loss",
    }


def test_default_influence_matrix_prefers_first_output_over_total_loss() -> None:
    entries = [
        {"id": "influences_total_loss_total_loss"},
        {"id": "influences_total_loss_output_0"},
    ]

    assert (
        build_static.default_influence_matrix_id({entry["id"] for entry in entries}, entries)
        == "influences_total_loss_output_0"
    )


def test_default_influence_matrix_falls_back_to_total_loss() -> None:
    entries = [
        {"id": "influences_total_loss_total_loss"},
        {"id": "influences_bc_loss_total_loss"},
    ]

    assert (
        build_static.default_influence_matrix_id({entry["id"] for entry in entries}, entries)
        == "influences_total_loss_total_loss"
    )


def test_process_influence_matrix_subsets_candidate_rows_and_train_columns(tmp_path) -> None:
    matrix_path = tmp_path / "matrix.npz"
    scores = np.array(
        [
            [6, -12, 18, -24, 30, -36],
            [-1, -2, -3, -4, -5, -6],
            [12, -18, 24, -30, 36, -42],
            [1, 2, 3, 4, 5, 6],
            [-6, 12, -18, 24, -30, 36],
        ],
        dtype=np.float32,
    )
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
        matrix_metadata=build_static.load_matrix_metadata(matrix_path),
    )

    assert metadata["scores_shape"] == [3, 3]
    assert metadata["source_scores_shape"] == [5, 6]
    assert metadata["candidate_points_shape"] == [3, 2]
    assert metadata["source_candidate_points_shape"] == [5, 2]
    assert "summary" not in metadata
    assert not list(tmp_path.rglob("summary_*.f32"))
    assert "top_chunks" not in metadata
    assert metadata["scores"]["dtype"] == "float32"
    assert metadata["scores"]["shape"] == [3, 3]
    assert metadata["score_layout"] == {
        "kind": "dense_row_major",
        "row_stride_bytes": 12,
        "data_offset_bytes": 0,
    }
    dense_scores = np.fromfile(tmp_path / metadata["scores"]["path"], dtype=np.float32).reshape(
        3, 3
    )
    expected = scores[np.ix_([0, 2, 4], [0, 2, 5])] / 6
    np.testing.assert_allclose(dense_scores, expected.astype(np.float32))


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
    )

    assert metadata["scores_shape"] == [3, 3]
    assert metadata["source_scores_shape"] == [6, 6]
    assert metadata["candidate_points_shape"] == [3, 2]
    dense_scores = np.fromfile(tmp_path / metadata["scores"]["path"], dtype=np.float32).reshape(
        3, 3
    )
    np.testing.assert_allclose(dense_scores, scores[np.ix_(train_indices, train_indices)] / 6)


def test_bundle_report_groups_influence_matrix_files(tmp_path) -> None:
    (tmp_path / "index.json").write_text("{}")
    scores = tmp_path / "run" / "influence" / "m0" / "scores.f32"
    scores.parent.mkdir(parents=True)
    np.array([1, 2, 3], dtype=np.float32).tofile(scores)

    report = build_static.build_bundle_report(tmp_path, budget_bytes=10_000)

    assert report["schema_version"] == 8
    assert report["within_budget"] is True
    assert report["by_kind"]["influence_matrices"] == 12
