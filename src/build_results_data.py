#!/usr/bin/env python3
"""Build compact static data for the webdemo Results dashboard."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from pinnfluence.utils.utils import loss_term_names

SCHEMA_VERSION = 1
DEFAULT_LOSS_DECOMP_DIR = Path("raw_data/loss_decompositions/loss_decompositions")
DEFAULT_OUTPUT_DIR = Path("webdemo/public/data/results")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loss-decomp-dir", type=Path, default=DEFAULT_LOSS_DECOMP_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def display_problem_name(problem: str) -> str:
    problem = problem.removesuffix("_nd").replace("_disk", "")
    return problem.replace("_", " ").title()


def canonical_problem(problem: str) -> str:
    return "poisson_disk" if problem == "poisson" else problem


def term_label(problem: str, term: str) -> str:
    return loss_term_names.get(canonical_problem(problem), {}).get(term, term.replace("_", " "))


def parse_loss_decomp_name(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    marker = "_loss_decomp_"
    if marker not in stem:
        raise ValueError(f"Unexpected loss decomposition filename: {path.name}")
    problem, rest = stem.split(marker, 1)
    quality = "good" if rest.startswith("good") else "bad"
    output_suffix = rest.removeprefix(quality).removeprefix("_")
    output_id = output_suffix if output_suffix else "output_0"
    return problem, quality, output_id


def loss_axis(problem: str) -> dict[str, str]:
    if problem in {"poisson", "poisson_disk", "navier_stokes_nd"}:
        return {"id": "x", "label": "x"}
    return {"id": "t", "label": "t"}


def load_loss_decompositions(loss_decomp_dir: Path) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(loss_decomp_dir.glob("*.npz")):
        raw_problem, quality, output_id = parse_loss_decomp_name(path)
        problem = canonical_problem(raw_problem)
        with np.load(path, allow_pickle=True) as data:
            result = data["res"].item()
        key = (problem, quality)
        entry = grouped.setdefault(
            key,
            {
                "problem": problem,
                "display_name": display_problem_name(problem),
                "quality": quality,
                "source_kind": "aggregate_summary",
                "axis": loss_axis(problem),
                "outputs": [],
            },
        )
        output_label = term_label(problem, output_id)
        entry["outputs"].append(
            {
                "id": output_id,
                "label": output_label,
                "mean_coherence": float(result["mean_coherence"]),
                "std_coherence": float(result["std_coherence"]),
                "binned_coherence": floats(result["binned_coherence"]),
                "bin_centers": floats(result["bin_centers"]),
                "terms": [
                    {
                        "id": term,
                        "label": term_label(problem, term),
                        "mean_fraction": float(result["mean_fractions"].get(term, 0.0)),
                        "std_fraction": float(result["std_fractions"].get(term, 0.0)),
                        "binned_fraction": floats(result["binned_fractions"].get(term, [])),
                        "binned_fraction_std": floats(result["binned_fractions_std"].get(term, [])),
                    }
                    for term in sorted(result["mean_fractions"], key=term_sort_key)
                ],
            }
        )
    records = list(grouped.values())
    for record in records:
        record["outputs"].sort(key=lambda output: output["id"])
    return sorted(records, key=lambda record: (record["problem"], record["quality"]))


def floats(values: Any) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float).reshape(-1)]


def term_sort_key(term: str) -> tuple[int, int, str]:
    prefix, _, suffix = term.partition("_")
    order = {"pde": 0, "bc": 1}.get(prefix, 2)
    index = int(suffix) if suffix.isdigit() else 999
    return order, index, term


def paper_indicators() -> dict[str, list[dict[str, Any]]]:
    temporal = [
        temporal_entry("diffusion", "Diffusion", 0.46, 0.33, 0.02, 0.26, 0.06, 0.41),
        temporal_entry("allen_cahn", "Allen Cahn", 0.43, 0.50, 0.02, 0.32, 0.05),
        temporal_entry("burgers", "Burgers", 0.43, 0.41, 0.02, 0.28, 0.02),
        temporal_entry("drift_diffusion", "Drift Diffusion", 0.46, 0.46, 0.04, 0.21, 0.06),
        temporal_entry("wave", "Wave", 0.43, 0.41, 0.03, 0.11, 0.02),
    ]
    directionality = [
        direction_entry(
            "poisson_disk", "Poisson Disk", "output_0", "û", 0.28, 0.29, 0.03, 0.69, 0.06
        ),
        direction_entry(
            "navier_stokes_nd",
            "Navier Stokes",
            "output_0",
            "x-velocity",
            0.48,
            0.15,
            0.01,
            0.15,
            0.03,
        ),
        direction_entry(
            "navier_stokes_nd",
            "Navier Stokes",
            "output_1",
            "y-velocity",
            0.48,
            0.25,
            0.01,
            0.20,
            0.05,
        ),
        direction_entry(
            "navier_stokes_nd",
            "Navier Stokes",
            "output_2",
            "pressure",
            0.48,
            0.14,
            0.02,
            0.11,
            0.02,
        ),
    ]
    return {"temporal": temporal, "directionality": directionality}


def temporal_entry(
    problem: str,
    display_name: str,
    baseline: float,
    good: float,
    good_std: float,
    bad: float,
    bad_std: float,
    bad_baseline: float | None = None,
) -> dict[str, Any]:
    entry = {
        "problem": problem,
        "display_name": display_name,
        "baseline": baseline,
        "values": {
            "good": {"mean": good, "std": good_std},
            "bad": {"mean": bad, "std": bad_std},
        },
    }
    if bad_baseline is not None:
        entry["bad_baseline"] = bad_baseline
        entry["note"] = "Poorly-trained model uses a different sampling baseline."
    return entry


def direction_entry(
    problem: str,
    display_name: str,
    output_id: str,
    output_label: str,
    baseline: float,
    good: float,
    good_std: float,
    bad: float,
    bad_std: float,
) -> dict[str, Any]:
    return {
        "problem": problem,
        "display_name": display_name,
        "output_id": output_id,
        "output_label": output_label,
        "baseline": baseline,
        "values": {
            "good": {"mean": good, "std": good_std},
            "bad": {"mean": bad, "std": bad_std},
        },
    }


def main() -> None:
    args = parse_args()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "sources": [
            args.loss_decomp_dir.as_posix(),
            "PINNfluence_extension_for_ICML-10.pdf tables",
        ],
        "loss_decompositions": load_loss_decompositions(args.loss_decomp_dir),
        "indicators": paper_indicators(),
    }
    out_path = args.output_dir / "index.json"
    write_json(out_path, payload)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
