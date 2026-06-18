from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from build_results_data import (
    SCHEMA_VERSION,
    build_loss_decompositions,
    compute_output,
    discover_output_matrices,
    selected_terms,
    write_json,
)


def test_selected_terms_prefers_granular_terms_per_family() -> None:
    assert selected_terms({"pde_0", "pde_loss", "bc_0", "bc_1", "bc_loss", "total_loss"}) == [
        "pde_0",
        "bc_0",
        "bc_1",
    ]
    assert selected_terms({"pde_loss", "bc_0", "bc_loss"}) == ["pde_loss", "bc_0"]
    assert selected_terms({"pde_0", "pde_1", "bc_loss"}) == ["pde_0", "pde_1", "bc_loss"]


def test_full_matrix_fraction_and_coherence_computation(tmp_path: Path) -> None:
    score_dir = make_score_dir(tmp_path)
    points = np.array(
        [
            [0.0, 0.1],
            [0.0, 0.2],
            [0.0, 0.8],
            [0.0, 0.9],
        ],
        dtype=np.float64,
    )
    write_matrix(score_dir, "pde_0", [[1.0, -1.0], [2.0, 0.0], [0.0, 0.0], [1.0, 1.0]], points)
    write_matrix(score_dir, "bc_0", [[1.0, 1.0], [-2.0, 0.0], [0.0, 0.0], [-1.0, 1.0]], points)
    write_matrix(score_dir, "pde_loss", [[10.0, 10.0]] * 4, points)
    write_matrix(score_dir, "bc_loss", [[10.0, 10.0]] * 4, points)

    matrices = discover_output_matrices(score_dir)
    result = compute_output("burgers", "output_0", matrices["output_0"], n_bins=2)

    assert [term["id"] for term in result.terms] == ["pde_0", "bc_0"]
    assert result.mean_coherence == 0.25
    np.testing.assert_allclose(result.arrays.bin_centers, [0.3, 0.7])
    np.testing.assert_allclose(result.arrays.binned_fractions, [[0.5, 0.25], [0.5, 0.25]])
    np.testing.assert_allclose(result.arrays.binned_fractions_std, [[0.0, 0.25], [0.0, 0.25]])
    np.testing.assert_allclose(result.arrays.binned_coherence, [0.25, 0.25])


def test_schema_v2_json_and_f32_array_writing(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw_data"
    output_dir = tmp_path / "results"
    score_dir = raw_dir / "fixture_float64_good" / "fixture_run_influence_scores"
    score_dir.mkdir(parents=True)
    points = np.array([[0.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    write_matrix(score_dir, "pde_loss", [[1.0, 2.0], [3.0, 4.0]], points)
    write_matrix(score_dir, "bc_0", [[1.0, 1.0], [-3.0, 4.0]], points)
    write_matrix(score_dir, "bc_loss", [[9.0, 9.0], [9.0, 9.0]], points)

    loss_decompositions = build_loss_decompositions(raw_dir, output_dir, n_bins=2)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": "2026-06-18T00:00:00+0000",
        "sources": [raw_dir.as_posix()],
        "loss_decompositions": loss_decompositions,
    }
    write_json(output_dir / "index.json", payload)

    index = json.loads((output_dir / "index.json").read_text())
    assert index["schema_version"] == 2
    output = index["loss_decompositions"][0]["outputs"][0]
    assert output["n_terms"] == 2
    assert [term["id"] for term in output["terms"]] == ["pde_loss", "bc_0"]
    assert "binned_fractions" not in output["terms"][0]
    assert output["arrays"]["binned_fractions"]["shape"] == [2, 2]

    fractions_path = output_dir / output["arrays"]["binned_fractions"]["path"]
    assert fractions_path.is_file()
    fractions = np.fromfile(fractions_path, dtype=np.float32).reshape(2, 2)
    assert fractions.shape == (2, 2)


def make_score_dir(tmp_path: Path) -> Path:
    score_dir = tmp_path / "fixture_run_influence_scores"
    score_dir.mkdir()
    return score_dir


def write_matrix(
    score_dir: Path, term: str, scores: list[list[float]], candidate_points: np.ndarray
) -> None:
    np.savez(
        score_dir / f"influences_{term}_output_0.npz",
        scores=np.asarray(scores, dtype=np.float32),
        candidate_points=candidate_points,
        num_pdes=np.array(1, dtype=np.int64),
        num_bcs=np.array(1, dtype=np.int64),
        n_outputs=np.array(1, dtype=np.int64),
        left_term=np.array("output_0"),
        right_term=np.array(term),
        self_influence=np.array(False),
    )
