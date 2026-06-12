"""Batch runner for experiment.py.

This script loops over many static demo manifests and many target points, then
invokes ``src/experiment.py`` for each pair.

The output layout is structured by problem group, run name, and point:

    plots/batch/<problem_group>/<run_name>/point_x..._y.../<experiment outputs>

Edit the module-level ``DATA_ROOT``, ``OUTPUT_ROOT``, and ``TARGET_POINTS`` to
change the sweep, or override them via CLI.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

DATA_ROOT = Path("webdemo/public/data")
OUTPUT_ROOT = Path("plots/batch")
TARGET_POINTS: list[tuple[float, float]] = [
    (0.036133, 0.46191),
    (0.0, 0.5),
    (-0.5, 0.25),
]


@dataclass(frozen=True)
class ManifestJob:
    manifest: Path
    problem_group: str
    run_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        default=None,
        help="Limit the sweep to one or more manifest.json files.",
    )
    parser.add_argument(
        "--point",
        action="append",
        nargs=2,
        type=float,
        default=None,
        metavar=("X", "Y"),
        help="Add a target point. Repeat to sweep several points.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run when its output directory already exists.",
    )
    return parser.parse_args()


def slugify_point(point: Iterable[float]) -> str:
    x, y = (float(value) for value in point)
    return f"point_x{x:.6f}_y{y:.6f}".replace(".", "p")


def discover_manifests(data_root: Path) -> list[ManifestJob]:
    jobs: list[ManifestJob] = []
    for manifest in sorted(data_root.rglob("manifest.json")):
        if manifest.name != "manifest.json":
            continue
        run_name = manifest.parent.name
        problem_group = (
            manifest.parent.parent.name if manifest.parent.parent != data_root else data_root.name
        )
        jobs.append(ManifestJob(manifest=manifest, problem_group=problem_group, run_name=run_name))
    return jobs


def selected_manifests(
    all_jobs: list[ManifestJob], manifest_filters: list[Path] | None
) -> list[ManifestJob]:
    if not manifest_filters:
        return all_jobs
    wanted = {path.resolve() for path in manifest_filters}
    return [job for job in all_jobs if job.manifest.resolve() in wanted]


def main() -> None:
    args = parse_args()
    points = [tuple(point) for point in (args.point or TARGET_POINTS)]
    jobs = discover_manifests(args.data_root)
    jobs = selected_manifests(jobs, args.manifest)
    if not jobs:
        raise SystemExit("No manifest.json files found for the requested sweep")

    experiment_script = Path(__file__).with_name("experiment.py")
    total = len(jobs) * len(points)
    completed = 0
    print(f"Found {len(jobs)} manifests and {len(points)} points ({total} runs)")

    for job in jobs:
        for point in points:
            point_dir = args.output_root / job.problem_group / job.run_name / slugify_point(point)
            if args.skip_existing and point_dir.exists():
                print(f"[skip] {job.manifest} @ {point} -> {point_dir}")
                completed += 1
                continue

            point_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(experiment_script),
                "--manifest",
                str(job.manifest),
                "--point",
                str(point[0]),
                str(point[1]),
                "--output-dir",
                str(point_dir),
            ]
            print(f"[{completed + 1}/{total}] {job.manifest} @ {point}")
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as exc:
                error_path = point_dir / "error.txt"
                error_path.write_text(
                    f"Command failed with exit code {exc.returncode}\n"
                    f"Manifest: {job.manifest}\n"
                    f"Point: {point}\n"
                    f"Command: {' '.join(cmd)}\n"
                )
                print(f"[error] {job.manifest} @ {point} -> {error_path}")
            completed += 1

    print(f"Finished {completed} runs. Output root: {args.output_root}")


if __name__ == "__main__":
    main()
