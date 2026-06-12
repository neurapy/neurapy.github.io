import { describe, expect, it } from "vitest";

import {
  captureVariantState,
  denormalizeSelection,
  resolveRestoredFieldId,
  resolveRestoredMatrixId,
  type VariantStateSnapshot,
} from "../src/state/variantRestore";
import { initialState, reduceState } from "../src/state/store";
import type { InfluenceMatrixManifest, RunManifest } from "../src/types";

function matrix(
  id: string,
  leftTerm: string,
  rightTerm: string,
  method = "PINNfluence",
): InfluenceMatrixManifest {
  return {
    id,
    method,
    left_term: leftTerm,
    right_term: rightTerm,
    num_pdes: 1,
    num_bcs: 0,
    n_outputs: 1,
    self_influence: false,
    scores_shape: [4, 5],
    row_source: "candidate_points",
    row_count: 4,
    k: 4,
    max_local_influence_points: 4,
    label: id,
    display_label: id,
    scores: { path: `${id}/scores.f32`, dtype: "float32", shape: [4, 5], bytes: 80 },
    score_layout: {
      kind: "dense_row_major",
      row_stride_bytes: 20,
      data_offset_bytes: 0,
    },
  };
}

function manifest(overrides: Partial<RunManifest> = {}): RunManifest {
  return {
    schema_version: 8,
    problem: "fixture",
    display_name: "Fixture",
    model_quality: "good",
    folder: "fixture",
    run_id: "fixture",
    status: "complete",
    errors: [],
    generated_at: "2026-06-09T00:00:00+0000",
    max_local_influence_points: 4,
    axes: ["x", "y"],
    bounds: { x: [0, 1], y: [0, 1] },
    n_candidate: 4,
    n_train: 5,
    n_outputs: 1,
    num_pdes: 1,
    num_bcs: 0,
    available_terms: ["output_0", "total_loss"],
    term_labels: { output_0: "Output", total_loss: "Total loss" },
    default_field: "pred_output_0",
    default_matrix: "m0",
    arrays: {
      candidate_points: { path: "candidate.f32", dtype: "float32", shape: [4, 2] },
      train_points: { path: "train.f32", dtype: "float32", shape: [5, 2] },
      train_kind: { path: "kind.u8", dtype: "uint8", shape: [5] },
      train_bc_id: { path: "bc.i16", dtype: "int16", shape: [5] },
    },
    field_raster: null,
    fields: {
      pred_output_0: {
        label: "Prediction",
        kind: "prediction",
        raster: { path: "pred.u16", dtype: "uint16", shape: [2, 2] },
        encoding: { kind: "linear", min: -1, max: 1, missing: 65535 },
        display_domain: [-1, 1],
      },
      loss_total: {
        label: "Total loss",
        kind: "loss",
        raster: { path: "loss.u16", dtype: "uint16", shape: [2, 2] },
        encoding: { kind: "linear", min: 0, max: 2, missing: 65535 },
        display_domain: [0, 2],
      },
    },
    influence_matrices: [matrix("m0", "total_loss", "total_loss")],
    ...overrides,
  };
}

describe("variant restore helpers", () => {
  it("resolves fields by exact id, then field kind, then manifest default", () => {
    const next = manifest({
      default_field: "pred_output_1",
      fields: {
        pred_output_1: {
          label: "Prediction 1",
          kind: "prediction",
          raster: { path: "pred1.u16", dtype: "uint16", shape: [2, 2] },
          encoding: { kind: "linear", min: -1, max: 1, missing: 65535 },
          display_domain: [-1, 1],
        },
        loss_residual: {
          label: "Residual loss",
          kind: "loss",
          raster: { path: "loss-residual.u16", dtype: "uint16", shape: [2, 2] },
          encoding: { kind: "linear", min: 0, max: 2, missing: 65535 },
          display_domain: [0, 2],
        },
      },
    });

    expect(
      resolveRestoredFieldId(next, {
        fieldId: "loss_total",
        fieldKind: "loss",
        matrixId: null,
        matrixSignature: null,
        selection: null,
      }),
    ).toBe("loss_residual");
    expect(resolveRestoredFieldId(next, null)).toBe("pred_output_1");
  });

  it("resolves matrices by exact id, compatible signature, then default", () => {
    const next = manifest({
      default_matrix: "default",
      influence_matrices: [
        matrix("default", "total_loss", "total_loss"),
        matrix("compatible", "total_loss", "output_0", "OtherMethod"),
      ],
    });
    const snapshot: VariantStateSnapshot = {
      fieldId: null,
      fieldKind: null,
      matrixId: "old_output",
      matrixSignature: {
        method: "PINNfluence",
        left_term: "total_loss",
        right_term: "output_0",
        row_source: "candidate_points",
      },
      selection: null,
    };

    expect(resolveRestoredMatrixId(next, snapshot, "preferred")).toBe("compatible");
    expect(resolveRestoredMatrixId(next, null, "missing")).toBe("default");
  });

  it("prefers the configured output matrix over the manifest default", () => {
    const next = manifest({
      default_matrix: "default",
      influence_matrices: [
        matrix("default", "total_loss", "total_loss"),
        matrix("preferred", "total_loss", "output_0"),
      ],
    });

    expect(resolveRestoredMatrixId(next, null, "preferred")).toBe("preferred");
  });

  it("captures and restores point selections by normalized plot position", () => {
    const selected = reduceState(initialState, {
      type: "selection",
      candidateIndex: 2,
      trainIndex: 3,
      coord: [5, 7],
    });
    const snapshot = captureVariantState(selected, manifest(), {
      minX: 0,
      maxX: 10,
      minY: 0,
      maxY: 20,
    });

    const restored = denormalizeSelection(snapshot.selection, {
      minX: -1,
      maxX: 1,
      minY: 100,
      maxY: 200,
    });

    expect(restored?.mode).toBe("point");
    if (restored?.mode !== "point") return;
    expect(restored.coord[0]).toBeCloseTo(0);
    expect(restored.coord[1]).toBeCloseTo(135);
  });

  it("captures and restores regions by normalized plot bounds", () => {
    const selected = reduceState(initialState, {
      type: "regionSelection",
      region: { minX: 2, maxX: 6, minY: 5, maxY: 15 },
      candidateIndices: [1, 2],
    });
    const snapshot = captureVariantState(selected, manifest(), {
      minX: 0,
      maxX: 10,
      minY: 0,
      maxY: 20,
    });

    const restored = denormalizeSelection(snapshot.selection, {
      minX: -1,
      maxX: 1,
      minY: 100,
      maxY: 200,
    });

    expect(restored?.mode).toBe("region");
    if (restored?.mode !== "region") return;
    expect(restored.region).toEqual({
      minX: -0.6,
      maxX: 0.19999999999999996,
      minY: 125,
      maxY: 175,
    });
  });
});
