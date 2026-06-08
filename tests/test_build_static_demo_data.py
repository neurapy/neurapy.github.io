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
