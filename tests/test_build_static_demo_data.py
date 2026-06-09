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


def test_bundle_report_groups_chunk_files(tmp_path) -> None:
    (tmp_path / "index.json").write_text("{}")
    chunk = tmp_path / "run" / "influence" / "m0" / "abs" / "chunks" / "0_values.i16"
    chunk.parent.mkdir(parents=True)
    np.array([1, 2, 3], dtype=np.int16).tofile(chunk)

    report = build_static.build_bundle_report(tmp_path, budget_bytes=10_000)

    assert report["schema_version"] == 5
    assert report["within_budget"] is True
    assert report["by_kind"]["influence_chunks"] == 6
