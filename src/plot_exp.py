#!/usr/bin/env python3
"""Plot loss decomposition summaries from raw_data/loss_decompositions."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT_DIR = Path("raw_data/loss_decompositions/loss_decompositions")
DEFAULT_OUTPUT_DIR = Path("raw_data/loss_decompositions/plots")
FILENAME_RE = re.compile(
    r"^(?P<problem>.+)_loss_decomp_(?P<quality>good|bad)(?P<output>_output_\d+)?$"
)


@dataclass(frozen=True)
class LossDecomposition:
    path: Path
    problem: str
    quality: str
    output: str | None
    mean_fractions: dict[str, float]
    std_fractions: dict[str, float]
    mean_coherence: float
    std_coherence: float
    binned_fractions: dict[str, np.ndarray]
    binned_fractions_std: dict[str, np.ndarray]
    binned_coherence: np.ndarray
    bin_centers: np.ndarray

    @property
    def group_key(self) -> tuple[str, str | None]:
        return self.problem, self.output

    @property
    def display_name(self) -> str:
        name = self.problem.replace("_", " ").title()
        if self.output is not None:
            name = f"{name} {self.output.replace('_', ' ').strip().title()}"
        return name

    @property
    def short_name(self) -> str:
        suffix = "" if self.output is None else self.output
        return f"{self.problem}{suffix}_{self.quality}"


def resolve_input_dir(input_dir: Path) -> Path:
    if any(input_dir.glob("*.npz")):
        return input_dir

    nested = input_dir / "loss_decompositions"
    if nested.is_dir() and any(nested.glob("*.npz")):
        return nested

    return input_dir


def parse_path(path: Path) -> tuple[str, str, str | None]:
    match = FILENAME_RE.match(path.stem)
    if not match:
        raise ValueError(f"Unexpected loss decomposition filename: {path.name}")

    output = match.group("output")
    return match.group("problem"), match.group("quality"), output


def as_float_dict(values: dict[str, object]) -> dict[str, float]:
    return {key: float(value) for key, value in values.items()}


def as_array_dict(values: dict[str, object]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value, dtype=float) for key, value in values.items()}


def load_decomposition(path: Path) -> LossDecomposition:
    problem, quality, output = parse_path(path)
    with np.load(path, allow_pickle=True) as data:
        res = data["res"].item()

    return LossDecomposition(
        path=path,
        problem=problem,
        quality=quality,
        output=output,
        mean_fractions=as_float_dict(res["mean_fractions"]),
        std_fractions=as_float_dict(res["std_fractions"]),
        mean_coherence=float(res["mean_coherence"]),
        std_coherence=float(res["std_coherence"]),
        binned_fractions=as_array_dict(res["binned_fractions"]),
        binned_fractions_std=as_array_dict(res["binned_fractions_std"]),
        binned_coherence=np.asarray(res["binned_coherence"], dtype=float),
        bin_centers=np.asarray(res["bin_centers"], dtype=float),
    )


def load_all(input_dir: Path) -> list[LossDecomposition]:
    paths = sorted(input_dir.glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    records = [load_decomposition(path) for path in paths]
    return sorted(records, key=record_sort_key)


def record_sort_key(record: LossDecomposition) -> tuple[str, int, int]:
    output_index = -1
    if record.output is not None:
        output_index = int(record.output.rsplit("_", 1)[-1])
    quality_index = {"bad": 0, "good": 1}[record.quality]
    return record.problem, output_index, quality_index


def term_sort_key(term: str) -> tuple[int, int, str]:
    prefix_order = {"pde": 0, "bc": 1}
    prefix, _, suffix = term.partition("_")
    return prefix_order.get(prefix, 2), int(suffix) if suffix.isdigit() else 999, term


def all_terms(records: list[LossDecomposition]) -> list[str]:
    terms = {term for record in records for term in record.mean_fractions}
    return sorted(terms, key=term_sort_key)


def record_label(record: LossDecomposition) -> str:
    label = record.problem.replace("_", " ").title()
    if record.output is not None:
        output_index = record.output.rsplit("_", 1)[-1]
        label = f"{label} Output {output_index}"
    return f"{label} ({record.quality})"


def color_for_terms(terms: list[str]) -> dict[str, object]:
    cmap = plt.get_cmap("tab20")
    return {term: cmap(index % cmap.N) for index, term in enumerate(terms)}


def grouped_records(
    records: list[LossDecomposition],
) -> dict[tuple[str, str | None], list[LossDecomposition]]:
    groups: dict[tuple[str, str | None], list[LossDecomposition]] = {}
    for record in records:
        groups.setdefault(record.group_key, []).append(record)

    for group_records in groups.values():
        group_records.sort(key=lambda record: {"bad": 0, "good": 1}[record.quality])
    return dict(sorted(groups.items(), key=lambda item: group_sort_key(item[0])))


def group_sort_key(group_key: tuple[str, str | None]) -> tuple[str, int]:
    problem, output = group_key
    if output is None:
        return problem, -1
    return problem, int(output.rsplit("_", 1)[-1])


def plot_summary(
    records: list[LossDecomposition],
    output_dir: Path,
    image_format: str,
    dpi: int,
) -> Path:
    terms = all_terms(records)
    colors = color_for_terms(terms)
    y = np.arange(len(records))
    labels = [record_label(record) for record in records]

    fig_height = max(8.0, 0.42 * len(records) + 2.0)
    fig, (ax_frac, ax_coh) = plt.subplots(
        1,
        2,
        figsize=(13.0, fig_height),
        sharey=True,
        gridspec_kw={"width_ratios": [3, 1]},
        constrained_layout=True,
    )

    lefts = np.zeros(len(records), dtype=float)
    for term in terms:
        values = np.array([record.mean_fractions.get(term, 0.0) for record in records])
        ax_frac.barh(y, values, left=lefts, label=term, color=colors[term], height=0.76)
        lefts += values

    coherence = np.array([record.mean_coherence for record in records])
    coherence_std = np.array([record.std_coherence for record in records])
    ax_coh.errorbar(coherence, y, xerr=coherence_std, fmt="o", color="#202020", capsize=3)

    ax_frac.set_title("Loss Decomposition Mean Fractions")
    ax_frac.set_xlabel("Fraction")
    ax_frac.set_xlim(0, 1.02)
    ax_frac.set_yticks(y)
    ax_frac.set_yticklabels(labels, fontsize=8)
    ax_frac.invert_yaxis()
    ax_frac.legend(ncols=min(6, max(1, len(terms))), fontsize=8, loc="upper center")
    ax_frac.grid(axis="x", alpha=0.25)

    ax_coh.set_title("Mean Coherence")
    ax_coh.set_xlabel("Coherence")
    ax_coh.set_xlim(0, 1.05)
    ax_coh.tick_params(axis="y", labelleft=False)
    ax_coh.grid(axis="x", alpha=0.25)

    output_path = output_dir / f"loss_decomposition_summary.{image_format}"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_group(
    records: list[LossDecomposition],
    output_dir: Path,
    image_format: str,
    dpi: int,
) -> Path:
    terms = all_terms(records)
    colors = color_for_terms(terms)
    ncols = len(records)
    fig, axes = plt.subplots(
        2,
        ncols,
        figsize=(6.0 * ncols, 7.0),
        sharey="row",
        constrained_layout=True,
    )
    if ncols == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for col, record in enumerate(records):
        ax_frac = axes[0, col]
        ax_coh = axes[1, col]
        y_values = [
            record.binned_fractions.get(term, np.zeros_like(record.bin_centers)) for term in terms
        ]
        stackplot_kwargs = {"labels": terms} if col == 0 else {}
        ax_frac.stackplot(
            record.bin_centers,
            y_values,
            colors=[colors[term] for term in terms],
            alpha=0.9,
            **stackplot_kwargs,
        )
        ax_frac.set_title(f"{record.display_name} - {record.quality.title()}")
        ax_frac.set_ylim(0, 1.02)
        ax_frac.set_ylabel("Fraction")
        ax_frac.grid(alpha=0.2)

        ax_coh.plot(record.bin_centers, record.binned_coherence, color="#202020", linewidth=1.8)
        lower = np.clip(record.mean_coherence - record.std_coherence, 0.0, 1.0)
        upper = np.clip(record.mean_coherence + record.std_coherence, 0.0, 1.0)
        ax_coh.axhline(record.mean_coherence, color="#606060", linestyle="--", linewidth=1.0)
        ax_coh.axhspan(lower, upper, color="#202020", alpha=0.12)
        ax_coh.set_ylim(0, 1.05)
        ax_coh.set_xlabel("Bin center")
        ax_coh.set_ylabel("Coherence")
        ax_coh.grid(alpha=0.2)

    axes[0, 0].legend(ncols=min(5, max(1, len(terms))), fontsize=8, loc="upper center")
    stem = records[0].problem
    if records[0].output is not None:
        stem += records[0].output
    output_path = output_dir / f"{stem}.{image_format}"
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def write_summary_csv(records: list[LossDecomposition], output_dir: Path) -> Path:
    terms = all_terms(records)
    output_path = output_dir / "loss_decomposition_summary.csv"
    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        header = [
            "problem",
            "output",
            "quality",
            "mean_coherence",
            "std_coherence",
        ]
        header.extend(f"mean_fraction_{term}" for term in terms)
        header.extend(f"std_fraction_{term}" for term in terms)
        writer.writerow(header)

        for record in records:
            row: list[str | float | None] = [
                record.problem,
                record.output,
                record.quality,
                record.mean_coherence,
                record.std_coherence,
            ]
            row.extend(record.mean_fractions.get(term, 0.0) for term in terms)
            row.extend(record.std_fractions.get(term, 0.0) for term in terms)
            writer.writerow(row)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot loss decomposition .npz files produced by PINNfluence experiments."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing loss decomposition .npz files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for generated plots. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--format",
        choices=["png", "pdf", "svg"],
        default="png",
        help="Image format to write.",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Raster output DPI.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = resolve_input_dir(args.input_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = load_all(input_dir)
    written = [
        plot_summary(records, args.output_dir, args.format, args.dpi),
        write_summary_csv(records, args.output_dir),
    ]
    for group in grouped_records(records).values():
        written.append(plot_group(group, args.output_dir, args.format, args.dpi))

    print(f"Loaded {len(records)} loss decomposition files from {input_dir}")
    print(f"Wrote {len(written)} files to {args.output_dir}")
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
