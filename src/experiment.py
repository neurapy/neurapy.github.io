"""Small plotting experiment for one run manifest and one target point.

Edit ``MANIFEST_PATH`` and ``TARGET_POINT`` near the top, or override them via CLI:

    uv run python src/experiment.py --manifest <path> --point 0.036133 0.46191

The script loads a static run bundle (manifest + arrays + influence matrix), finds the
closest source point to the requested coordinate, and writes a set of figures into
``plots/``:

* model output
* model loss
* raw influence scatter
* raw Gaussian blur map
* normalized influence scatter
* normalized Gaussian blur map
* an overview sheet with all six panels
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

MANIFEST_PATH = Path(
    "webdemo/public/data/allen_cahn_float64_good/"
    "allen_cahn_adam_100000_adam_25000_lbfgs_2500_domain_500_boundary_500_initial_3_x_64_hidden_float64_True_0_soft/"
    "manifest.json"
)
TARGET_POINT = (0.036133, 0.46191)
OUTPUT_DIR = Path("plots")
PREFERRED_MATRIX_ID: str | None = None
PREFERRED_OUTPUT_FIELD_ID: str | None = None
PREFERRED_LOSS_FIELD_ID: str | None = None
MIN_NORMALIZATION_LOSS = 1e-18

DTYPE_MAP = {
    "float32": np.float32,
    "uint32": np.uint32,
    "uint16": np.uint16,
    "uint8": np.uint8,
    "int16": np.int16,
}

WEBAPP_DIVERGING_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "webapp_diverging",
    ["#2166ac", "#f7f7f7", "#b2182b"],
)
WEBAPP_SEQUENTIAL_CMAP = plt.get_cmap("turbo")


@dataclass(frozen=True)
class SelectedPoint:
    index: int
    point: tuple[float, float]
    distance: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--point", nargs=2, type=float, default=TARGET_POINT, metavar=("X", "Y"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--matrix-id", default=PREFERRED_MATRIX_ID)
    parser.add_argument("--output-field-id", default=PREFERRED_OUTPUT_FIELD_ID)
    parser.add_argument("--loss-field-id", default=PREFERRED_LOSS_FIELD_ID)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_array(base_dir: Path, spec: dict[str, Any]) -> np.ndarray:
    path = base_dir / spec["path"]
    dtype = DTYPE_MAP[spec["dtype"]]
    array = np.fromfile(path, dtype=dtype)
    return array.reshape(spec["shape"])


def load_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
    manifest_path = path / "manifest.json" if path.is_dir() else path
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    return manifest_path.parent, load_json(manifest_path)


def axis_bounds_from_manifest(manifest: dict[str, Any]) -> dict[str, tuple[float, float]]:
    field_raster = manifest.get("field_raster") or {}
    bounds = field_raster.get("bounds") or manifest.get("bounds") or {}
    axes = field_raster.get("axes") or manifest.get("axes") or ["x", "y"]
    if len(axes) < 2:
        axes = list(axes) + ["y"]
    x_axis = axes[0]
    y_axis = axes[1]
    x_bounds = bounds.get(x_axis) or bounds.get("x") or [0.0, 1.0]
    y_bounds = bounds.get(y_axis) or bounds.get("y") or [0.0, 1.0]
    return {
        "x": (float(x_bounds[0]), float(x_bounds[1])),
        "y": (float(y_bounds[0]), float(y_bounds[1])),
    }


def raster_extent(bounds: dict[str, tuple[float, float]]) -> list[float]:
    return [bounds["x"][0], bounds["x"][1], bounds["y"][0], bounds["y"][1]]


def decode_linear_raster(raw: np.ndarray, encoding: dict[str, Any]) -> np.ndarray:
    raw = np.asarray(raw)
    missing = encoding.get("missing")
    vmin = float(encoding["min"])
    vmax = float(encoding["max"])
    span = vmax - vmin if vmax != vmin else 1.0
    values = np.full(raw.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(raw)
    if missing is not None:
        valid &= raw != missing
    values[valid] = vmin + (raw[valid].astype(np.float64) / 65534.0) * span
    return values


def select_field(manifest: dict[str, Any], kind: str, preferred_id: str | None = None) -> str:
    fields = manifest.get("fields") or {}
    if preferred_id and preferred_id in fields:
        return preferred_id
    if preferred_id:
        raise KeyError(f"Field {preferred_id!r} not found")
    default_id = manifest.get("default_field")
    if default_id and fields.get(default_id, {}).get("kind") == kind:
        return default_id
    for field_id, field in fields.items():
        if field.get("kind") == kind:
            return field_id
    raise KeyError(f"No field of kind {kind!r} found")


def select_candidate_matrix(
    manifest: dict[str, Any], preferred_id: str | None = None
) -> dict[str, Any]:
    matrices = manifest.get("influence_matrices") or []
    if preferred_id:
        for matrix in matrices:
            if matrix.get("id") == preferred_id:
                return matrix
        raise KeyError(f"Matrix {preferred_id!r} not found")

    default_id = manifest.get("default_matrix")
    if default_id:
        for matrix in matrices:
            if matrix.get("id") == default_id and matrix.get("row_source") == "candidate_points":
                return matrix
    for matrix in matrices:
        if matrix.get("row_source") == "candidate_points":
            return matrix
    raise KeyError("No candidate-point influence matrix found")


def closest_point(points: np.ndarray, target: Iterable[float]) -> SelectedPoint:
    pts = np.asarray(points, dtype=np.float64)
    target_arr = np.asarray(list(target), dtype=np.float64).reshape(1, -1)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("Expected a 2D point cloud")
    deltas = pts[:, :2] - target_arr[:, :2]
    distances = np.linalg.norm(deltas, axis=1)
    index = int(np.argmin(distances))
    point = (float(pts[index, 0]), float(pts[index, 1]))
    return SelectedPoint(index=index, point=point, distance=float(distances[index]))


def grid_index(
    point: tuple[float, float], bounds: dict[str, tuple[float, float]], width: int, height: int
) -> tuple[float, float]:
    x_min, x_max = bounds["x"]
    y_min, y_max = bounds["y"]
    x_span = max(x_max - x_min, 1e-12)
    y_span = max(y_max - y_min, 1e-12)
    x = np.clip((point[0] - x_min) / x_span * width - 0.5, 0.0, width - 1.0)
    y = np.clip((y_max - point[1]) / y_span * height - 0.5, 0.0, height - 1.0)
    return float(x), float(y)


def accumulate_bilinear_samples(
    points: np.ndarray,
    values: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    numerator = np.zeros((height, width), dtype=np.float64)
    support = np.zeros((height, width), dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    for point, value in zip(pts, vals, strict=True):
        if not np.isfinite(value):
            continue
        gx, gy = grid_index((float(point[0]), float(point[1])), bounds, width, height)
        x0 = int(np.floor(gx))
        y0 = int(np.floor(gy))
        x1 = min(width - 1, x0 + 1)
        y1 = min(height - 1, y0 + 1)
        tx = 0.0 if x1 == x0 else gx - x0
        ty = 0.0 if y1 == y0 else gy - y0
        weights = (
            (x0, y0, (1.0 - tx) * (1.0 - ty)),
            (x1, y0, tx * (1.0 - ty)),
            (x0, y1, (1.0 - tx) * ty),
            (x1, y1, tx * ty),
        )
        for x, y, weight in weights:
            if weight <= 0:
                continue
            numerator[y, x] += value * weight
            support[y, x] += weight
    return numerator, support


def gaussian_kernel(sigma: float) -> np.ndarray:
    sigma = float(max(1.0, sigma))
    radius = max(1, int(math.ceil(sigma * 3.0)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-(offsets * offsets) / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel


def convolve_separable(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    temp = np.empty_like(image, dtype=np.float64)
    for row_index, row in enumerate(image):
        temp[row_index] = np.convolve(row, kernel, mode="same")
    output = np.empty_like(temp, dtype=np.float64)
    for col_index in range(temp.shape[1]):
        output[:, col_index] = np.convolve(temp[:, col_index], kernel, mode="same")
    return output


def median_nearest_neighbor_distance(points: np.ndarray, sample_limit: int = 1024) -> float:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return 1.0
    if pts.shape[0] > sample_limit:
        indices = np.linspace(0, pts.shape[0] - 1, sample_limit, dtype=np.int64)
        pts = pts[indices]
    nearest = np.full(pts.shape[0], np.inf, dtype=np.float64)
    block_size = min(256, pts.shape[0])
    for start in range(0, pts.shape[0], block_size):
        block = pts[start : start + block_size]
        deltas = block[:, None, :] - pts[None, :, :]
        dist2 = np.sum(deltas * deltas, axis=2)
        local_indices = np.arange(block.shape[0])
        dist2[local_indices, start + local_indices] = np.inf
        nearest[start : start + block.shape[0]] = np.sqrt(np.min(dist2, axis=1))
    finite = nearest[np.isfinite(nearest)]
    if finite.size == 0:
        return 1.0
    return float(np.median(finite))


def estimate_sigma(points: np.ndarray, width: int, height: int) -> float:
    area = float(max(1, width * height))
    max_bandwidth = max(3.0, min(width, height) * 0.18)
    if len(points) < 2:
        return float(np.clip(math.sqrt(area / max(1, len(points))) * 0.08, 3.0, max_bandwidth))
    spacing = median_nearest_neighbor_distance(points)
    return float(np.clip(spacing * 1.35, 3.0, max_bandwidth))


def blur_field(
    points: np.ndarray,
    values: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    width: int,
    height: int,
) -> np.ndarray:
    numerator, support = accumulate_bilinear_samples(points, values, bounds, width, height)
    kernel = gaussian_kernel(estimate_sigma(points, width, height))
    blurred_numerator = convolve_separable(numerator, kernel)
    blurred_support = convolve_separable(support, kernel)
    field = np.zeros_like(blurred_numerator, dtype=np.float64)
    mask = blurred_support > 1e-4
    field[mask] = blurred_numerator[mask] / blurred_support[mask]
    return field


def normalize_values_by_loss(
    values: np.ndarray,
    loss_points: np.ndarray,
    minimum_loss: float = 1e-12,
) -> np.ndarray:
    influence = np.asarray(values, dtype=np.float64).reshape(-1)
    loss = np.asarray(loss_points, dtype=np.float64).reshape(-1)
    if influence.shape != loss.shape:
        raise ValueError("Influence and loss arrays must have the same shape")
    normalized = np.full(influence.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(influence) & np.isfinite(loss) & (np.abs(loss) > minimum_loss)
    normalized[valid] = influence[valid] / loss[valid]
    return normalized


def robust_scale_max(values: np.ndarray) -> float:
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 1.0
    abs_vals = np.sort(np.abs(vals))
    index = min(abs_vals.size - 1, max(0, int(math.floor((abs_vals.size - 1) * 0.98))))
    scale = float(abs_vals[index])
    if not np.isfinite(scale) or scale <= 0:
        scale = float(np.max(abs_vals)) if abs_vals.size else 1.0
    return scale if scale > 0 else 1.0


def sample_raster(
    raster: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    point: tuple[float, float],
    mask: np.ndarray | None = None,
) -> float:
    height, width = raster.shape[:2]
    gx, gy = grid_index(point, bounds, width, height)
    x0 = int(np.floor(gx))
    y0 = int(np.floor(gy))
    x1 = min(width - 1, x0 + 1)
    y1 = min(height - 1, y0 + 1)
    tx = 0.0 if x1 == x0 else gx - x0
    ty = 0.0 if y1 == y0 else gy - y0
    samples = (
        (x0, y0, (1.0 - tx) * (1.0 - ty)),
        (x1, y0, tx * (1.0 - ty)),
        (x0, y1, (1.0 - tx) * ty),
        (x1, y1, tx * ty),
    )
    total = 0.0
    weight_sum = 0.0
    for x, y, weight in samples:
        if weight <= 0:
            continue
        if mask is not None and not bool(mask[y, x]):
            continue
        value = raster[y, x]
        if not np.isfinite(value):
            continue
        total += float(value) * weight
        weight_sum += weight
    if weight_sum <= 0:
        y_idx = int(round(gy))
        x_idx = int(round(gx))
        if mask is not None and not bool(mask[y_idx, x_idx]):
            return float("nan")
        return float(raster[y_idx, x_idx])
    return total / weight_sum


def decodable_raster(
    base_dir: Path,
    field: dict[str, Any],
    field_raster: dict[str, Any] | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    raw = load_array(base_dir, field["raster"])
    decoded = decode_linear_raster(raw, field["encoding"])
    mask = None
    if field_raster and field_raster.get("mask"):
        mask = load_array(base_dir, field_raster["mask"]).astype(bool)
        if mask.shape != decoded.shape:
            raise ValueError("field mask shape does not match raster shape")
    return decoded, mask


def field_display_domain(field: dict[str, Any], raster: np.ndarray) -> tuple[float, float]:
    domain = field.get("display_domain")
    if (
        isinstance(domain, list)
        and len(domain) == 2
        and all(isinstance(value, (int, float)) for value in domain)
    ):
        return float(domain[0]), float(domain[1])
    finite = np.asarray(raster, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (0.0, 1.0)
    return float(np.min(finite)), float(np.max(finite))


def set_axis_common(ax: plt.Axes, title: str, bounds: dict[str, tuple[float, float]]) -> None:
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["y"])
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=10)
    ax.grid(False)


def plot_raster_panel(
    ax: plt.Axes,
    raster: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    title: str,
    cmap: mcolors.Colormap,
    vmin: float,
    vmax: float,
    selected_point: tuple[float, float],
    selected_label: str,
    mask: np.ndarray | None = None,
) -> None:
    extent = raster_extent(bounds)
    image = np.array(raster, dtype=np.float64)
    if mask is not None:
        image = np.where(mask, image, np.nan)
    im = ax.imshow(
        image,
        extent=extent,
        origin="upper",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.scatter(
        [selected_point[0]],
        [selected_point[1]],
        marker="*",
        s=160,
        facecolor="#e9a82f",
        edgecolor="#182230",
        linewidth=1.2,
        zorder=5,
        label=selected_label,
    )
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)


def plot_scatter_panel(
    ax: plt.Axes,
    source_points: np.ndarray,
    response_points: np.ndarray,
    values: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    title: str,
    selected_point: tuple[float, float],
    selected_label: str,
) -> None:
    scale_max = robust_scale_max(values)
    norm = mcolors.TwoSlopeNorm(vmin=-scale_max, vcenter=0.0, vmax=scale_max)
    ax.scatter(
        response_points[:, 0],
        response_points[:, 1],
        c=values,
        cmap=WEBAPP_DIVERGING_CMAP,
        norm=norm,
        s=16,
        alpha=0.9,
        linewidths=0,
    )
    ax.scatter(
        [selected_point[0]],
        [selected_point[1]],
        marker="*",
        s=180,
        facecolor="#e9a82f",
        edgecolor="#182230",
        linewidth=1.2,
        zorder=5,
        label=selected_label,
    )
    ax.scatter(
        source_points[:, 0],
        source_points[:, 1],
        c="#526070",
        s=4,
        alpha=0.08,
        linewidths=0,
        zorder=1,
    )
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["y"])
    ax.set_aspect("equal", adjustable="box")
    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=WEBAPP_DIVERGING_CMAP),
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )
    cbar.ax.tick_params(labelsize=9)


def plot_blur_panel(
    ax: plt.Axes,
    response_points: np.ndarray,
    values: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    grid_width: int,
    grid_height: int,
    title: str,
    selected_point: tuple[float, float],
    selected_label: str,
) -> None:
    field = blur_field(response_points, values, bounds, grid_width, grid_height)
    scale_max = robust_scale_max(field[np.isfinite(field)])
    extent = raster_extent(bounds)
    im = ax.imshow(
        field,
        extent=extent,
        origin="upper",
        cmap=WEBAPP_DIVERGING_CMAP,
        vmin=-scale_max,
        vmax=scale_max,
        interpolation="bicubic",
    )
    ax.scatter(
        [selected_point[0]],
        [selected_point[1]],
        marker="*",
        s=180,
        facecolor="#e9a82f",
        edgecolor="#182230",
        linewidth=1.2,
        zorder=5,
        label=selected_label,
    )
    ax.set_title(title, fontsize=14, pad=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["y"])
    ax.set_aspect("equal", adjustable="box")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)


def build_slug(manifest: dict[str, Any], selected: SelectedPoint) -> str:
    point_slug = f"x{selected.point[0]:.6f}_y{selected.point[1]:.6f}".replace(".", "p")
    return f"{manifest['problem']}_{manifest['model_quality']}_{point_slug}_idx{selected.index}"


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    base_dir, manifest = load_manifest(args.manifest)
    bounds = axis_bounds_from_manifest(manifest)

    candidate_points = load_array(base_dir, manifest["arrays"]["candidate_points"])
    train_points = load_array(base_dir, manifest["arrays"]["train_points"])
    selected = closest_point(candidate_points, args.point)

    matrix = select_candidate_matrix(manifest, args.matrix_id)
    matrix_values = load_array(base_dir, matrix["scores"])
    if matrix.get("row_source") != "candidate_points":
        raise ValueError(
            "This experiment currently expects a candidate-point influence matrix. "
            "Use a manifest where the selected matrix rows are candidate points."
        )
    if selected.index >= matrix_values.shape[0]:
        raise IndexError(
            f"Selected candidate index {selected.index} exceeds influence matrix rows "
            f"({matrix_values.shape[0]})"
        )

    output_field_id = select_field(manifest, "prediction", args.output_field_id)
    loss_field_id = select_field(manifest, "loss", args.loss_field_id)
    output_field = manifest["fields"][output_field_id]
    loss_field = manifest["fields"][loss_field_id]
    field_raster = manifest.get("field_raster")
    output_values, output_mask = decodable_raster(base_dir, output_field, field_raster)
    loss_values, loss_mask = decodable_raster(base_dir, loss_field, field_raster)

    output_point_value = sample_raster(output_values, bounds, selected.point, output_mask)
    loss_point_value = sample_raster(loss_values, bounds, selected.point, loss_mask)
    selected_loss_denominator = float(loss_point_value)
    if (
        not np.isfinite(selected_loss_denominator)
        or abs(selected_loss_denominator) < MIN_NORMALIZATION_LOSS
    ):
        selected_loss_denominator = MIN_NORMALIZATION_LOSS

    raw_values = np.asarray(matrix_values[selected.index], dtype=np.float64).reshape(-1)
    if raw_values.shape[0] != train_points.shape[0]:
        raise ValueError(
            f"Influence row length {raw_values.shape[0]} does not match train point count "
            f"{train_points.shape[0]}"
        )

    loss_at_points = np.array(
        [
            sample_raster(loss_values, bounds, (float(point[0]), float(point[1])), loss_mask)
            for point in train_points
        ],
        dtype=np.float64,
    )
    norm_values = normalize_values_by_loss(raw_values, loss_at_points)
    norm_finite = np.isfinite(norm_values)
    if not np.any(norm_finite):
        raise ValueError("Loss-normalized influence values are all invalid")

    output_bounds = bounds
    response_points = train_points
    grid_width = int(field_raster["width"]) if field_raster else 512
    grid_height = int(field_raster["height"]) if field_raster else 512

    output_vmin, output_vmax = field_display_domain(output_field, output_values)
    loss_vmin, loss_vmax = field_display_domain(loss_field, loss_values)

    output_title = output_field.get("label", "Model Output")
    loss_title = loss_field.get("label", "Model Loss")
    raw_influence_title = "Influences at Closest Candidate"
    raw_blur_title = "Gaussian Blur Map"
    norm_influence_title = "Influences / Local Loss"
    norm_blur_title = "Gaussian Blur / Local Loss"

    slug = build_slug(manifest, selected)
    out_dir = args.output_dir / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Manifest: {base_dir}")
    print(f"Selected candidate index: {selected.index}")
    print(f"Selected candidate point: ({selected.point[0]:.6f}, {selected.point[1]:.6f})")
    print(f"Distance to target: {selected.distance:.6g}")
    print(f"Model output at point: {output_point_value:.6g}")
    print(f"Model loss at point: {loss_point_value:.6g}")
    if abs(loss_point_value) < MIN_NORMALIZATION_LOSS or not np.isfinite(loss_point_value):
        print(
            f"Normalization factor: 1 / {selected_loss_denominator:.6g} "
            f"(clamped from {loss_point_value:.6g})"
        )
    else:
        print(f"Normalization factor: 1 / {selected_loss_denominator:.6g}")
    print(f"Output field: {output_field_id}")
    print(f"Loss field: {loss_field_id}")
    print(f"Influence matrix: {matrix['id']}")
    print(f"Writing figures to: {out_dir}")

    summary = {
        "manifest": str(base_dir / "manifest.json"),
        "selected_index": selected.index,
        "selected_point": list(selected.point),
        "selected_target_point": list(args.point),
        "distance": selected.distance,
        "output_field": output_field_id,
        "loss_field": loss_field_id,
        "influence_matrix": matrix["id"],
        "output_point_value": output_point_value,
        "loss_point_value": loss_point_value,
        "normalization_factor": 1.0 / selected_loss_denominator,
        "normalization_denominator": selected_loss_denominator,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    panels: list[tuple[str, callable, dict[str, Any]]] = []
    panels.append(
        (
            "output",
            plot_raster_panel,
            {
                "raster": output_values,
                "bounds": output_bounds,
                "title": f"{output_title}\nvalue at point: {output_point_value:.4g}",
                "cmap": WEBAPP_SEQUENTIAL_CMAP,
                "vmin": output_vmin,
                "vmax": output_vmax,
                "selected_point": selected.point,
                "selected_label": "selected candidate",
                "mask": output_mask,
            },
        )
    )
    panels.append(
        (
            "influences_raw",
            plot_scatter_panel,
            {
                "source_points": candidate_points,
                "response_points": response_points,
                "values": raw_values,
                "bounds": bounds,
                "title": raw_influence_title,
                "selected_point": selected.point,
                "selected_label": "selected candidate",
            },
        )
    )
    panels.append(
        (
            "loss",
            plot_raster_panel,
            {
                "raster": loss_values,
                "bounds": output_bounds,
                "title": f"{loss_title}\nvalue at point: {loss_point_value:.4g}",
                "cmap": WEBAPP_SEQUENTIAL_CMAP,
                "vmin": loss_vmin,
                "vmax": loss_vmax,
                "selected_point": selected.point,
                "selected_label": "selected candidate",
                "mask": loss_mask,
            },
        )
    )
    panels.append(
        (
            "blur_raw",
            plot_blur_panel,
            {
                "response_points": response_points,
                "values": raw_values,
                "bounds": bounds,
                "grid_width": grid_width,
                "grid_height": grid_height,
                "title": raw_blur_title,
                "selected_point": selected.point,
                "selected_label": "selected candidate",
            },
        )
    )
    panels.append(
        (
            "influences_norm",
            plot_scatter_panel,
            {
                "source_points": candidate_points,
                "response_points": response_points,
                "values": norm_values,
                "bounds": bounds,
                "title": norm_influence_title,
                "selected_point": selected.point,
                "selected_label": "selected candidate",
            },
        )
    )
    panels.append(
        (
            "blur_norm",
            plot_blur_panel,
            {
                "response_points": response_points,
                "values": norm_values,
                "bounds": bounds,
                "grid_width": grid_width,
                "grid_height": grid_height,
                "title": norm_blur_title,
                "selected_point": selected.point,
                "selected_label": "selected candidate",
            },
        )
    )

    for panel_name, renderer, kwargs in panels:
        fig, ax = plt.subplots(figsize=(7.5, 6.2), constrained_layout=True)
        renderer(ax=ax, **kwargs)
        ax.scatter(
            [selected.point[0]],
            [selected.point[1]],
            marker="*",
            s=180,
            facecolor="#e9a82f",
            edgecolor="#182230",
            linewidth=1.2,
            zorder=6,
        )
        fig.suptitle(
            f"{manifest['display_name']} | {manifest['model_quality'].title()} | {slug}",
            fontsize=13,
        )
        save_figure(fig, out_dir / f"{panel_name}.png")

    fig, axes = plt.subplots(3, 2, figsize=(15, 18), constrained_layout=True)
    axes = np.asarray(axes)

    plot_raster_panel(
        ax=axes[0, 0],
        raster=output_values,
        bounds=output_bounds,
        title=output_title,
        cmap=WEBAPP_SEQUENTIAL_CMAP,
        vmin=output_vmin,
        vmax=output_vmax,
        selected_point=selected.point,
        selected_label="selected candidate",
        mask=output_mask,
    )
    plot_scatter_panel(
        ax=axes[0, 1],
        source_points=candidate_points,
        response_points=response_points,
        values=raw_values,
        bounds=bounds,
        title=raw_influence_title,
        selected_point=selected.point,
        selected_label="selected candidate",
    )
    plot_raster_panel(
        ax=axes[1, 0],
        raster=loss_values,
        bounds=output_bounds,
        title=loss_title,
        cmap=WEBAPP_SEQUENTIAL_CMAP,
        vmin=loss_vmin,
        vmax=loss_vmax,
        selected_point=selected.point,
        selected_label="selected candidate",
        mask=loss_mask,
    )
    plot_blur_panel(
        ax=axes[1, 1],
        response_points=response_points,
        values=raw_values,
        bounds=bounds,
        grid_width=grid_width,
        grid_height=grid_height,
        title=raw_blur_title,
        selected_point=selected.point,
        selected_label="selected candidate",
    )
    plot_scatter_panel(
        ax=axes[2, 0],
        source_points=candidate_points,
        response_points=response_points,
        values=norm_values,
        bounds=bounds,
        title=norm_influence_title,
        selected_point=selected.point,
        selected_label="selected candidate",
    )
    plot_blur_panel(
        ax=axes[2, 1],
        response_points=response_points,
        values=norm_values,
        bounds=bounds,
        grid_width=grid_width,
        grid_height=grid_height,
        title=norm_blur_title,
        selected_point=selected.point,
        selected_label="selected candidate",
    )
    fig.suptitle(
        f"{manifest['display_name']} | selected candidate {selected.index} | "
        f"loss {loss_point_value:.6g}",
        fontsize=15,
    )
    save_figure(fig, out_dir / "overview.png")


if __name__ == "__main__":
    main()
