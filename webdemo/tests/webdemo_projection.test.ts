import { describe, expect, it } from "vitest";

import type { RunManifest } from "../src/types";
import { plotViewport, unprojectPointFromViewport } from "../src/viz/geometry";
import {
  axisLabelsForProblem,
  plotProjectionForManifest,
  projectPointToDisplay,
  regionBoundsFromProjectedViewportDrag,
} from "../src/viz/projection";

const baseManifest = {
  schema_version: 7,
  problem: "fixture",
  display_name: "Fixture",
  model_quality: "good",
  folder: "fixture",
  run_id: "fixture",
  status: "complete",
  errors: [],
  generated_at: "2026-06-09T00:00:00+0000",
  max_local_influence_points: 4,
  row_chunk_size: 2,
  axes: ["x", "y"],
  bounds: { x: [0, 1], y: [0, 1] },
  n_candidate: 4,
  n_train: 4,
  n_outputs: 1,
  num_pdes: 1,
  num_bcs: 0,
  available_terms: [],
  term_labels: {},
  default_field: null,
  default_matrix: null,
  arrays: {
    candidate_points: { path: "", dtype: "float32", shape: [0, 2] },
    train_points: { path: "", dtype: "float32", shape: [0, 2] },
    train_kind: { path: "", dtype: "uint8", shape: [0] },
    train_bc_id: { path: "", dtype: "int16", shape: [0] },
  },
  field_raster: null,
  fields: {},
  influence_matrices: [],
} satisfies RunManifest;

describe("plot projection", () => {
  it("compresses Drift Diffusion x display coordinates to a 2:1 aspect", () => {
    const projection = plotProjectionForManifest(
      { ...baseManifest, problem: "drift_diffusion" },
      { minX: 0, maxX: 2 * Math.PI, minY: 0, maxY: 1 },
    );

    expect(projection.displayBounds).toMatchObject({ minX: 0, maxX: 2, minY: 0, maxY: 1 });
    expect(projectPointToDisplay([Math.PI, 0.5], projection)).toEqual([1, 0.5]);
    expect(projection.formatXTick(0)).toBe("0");
    expect(projection.formatXTick(1)).toBe("π");
    expect(projection.formatXTick(2)).toBe("2π");
  });

  it("inverts projected drag regions back to physical Drift Diffusion coordinates", () => {
    const projection = plotProjectionForManifest(
      { ...baseManifest, problem: "drift_diffusion" },
      { minX: 0, maxX: 2 * Math.PI, minY: 0, maxY: 1 },
    );
    const viewport = plotViewport(projection.displayBounds, 258, 152, 0);
    const start: [number, number] = [viewport.x, viewport.bottom];
    const end: [number, number] = [viewport.right, viewport.y];
    const displayEnd = unprojectPointFromViewport(end[0], end[1], projection.displayBounds, viewport);

    expect(displayEnd[0]).toBeCloseTo(2);
    const region = regionBoundsFromProjectedViewportDrag(start, end, projection, viewport);
    expect(region.minX).toBeCloseTo(0);
    expect(region.maxX).toBeCloseTo(2 * Math.PI);
    expect(region.minY).toBeCloseTo(0);
    expect(region.maxY).toBeCloseTo(1);
  });

  it("maps semantic axis labels for current problem families", () => {
    for (const problem of ["allen_cahn", "burgers", "diffusion", "drift_diffusion", "wave"]) {
      expect(axisLabelsForProblem(problem)).toEqual({ x: "x", y: "t" });
    }
    expect(axisLabelsForProblem("poisson_disk")).toEqual({ x: "x", y: "y" });
    expect(axisLabelsForProblem("navier_stokes_nd")).toEqual({ x: "x", y: "y" });
    expect(axisLabelsForProblem("custom_problem", ["r", "s"])).toEqual({ x: "r", y: "s" });
  });
});
