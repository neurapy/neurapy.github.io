import { describe, expect, it } from "vitest";

import {
  computeCellsInfluenceLayer,
  computeLinearInfluenceField,
  influenceEntriesForBackground,
  type InfluenceField,
  type InfluenceMapLayer,
  robustAbsScaleMax,
} from "../src/viz/plots";
import { divergingColorScale } from "../src/viz/color";
import type { Bounds, PlotViewport } from "../src/types";

const bounds: Bounds = { minX: 0, maxX: 1, minY: 0, maxY: 1 };
const viewport: PlotViewport = {
  x: 0,
  y: 0,
  width: 100,
  height: 100,
  right: 100,
  bottom: 100,
};

function sample(field: { width: number; values: Float32Array }, col: number, row: number): number {
  return field.values[row * field.width + col];
}

function expectRaster(layer: InfluenceMapLayer): InfluenceField {
  expect(layer.kind).toBe("raster");
  return layer as InfluenceField;
}

describe("influence map samples", () => {
  it("averages duplicate sample locations before interpolation", () => {
    const layer = computeCellsInfluenceLayer({
      points: new Float32Array([0.5, 0.5, 0.5, 0.5]),
      dim: 2,
      bounds,
      viewport,
      indices: new Uint16Array([0, 1]),
      values: new Float32Array([2, 4]),
      gridWidth: 101,
      gridHeight: 101,
    });

    expect(layer.cellCount).toBe(1);
    expect(layer.samples[0].value).toBeCloseTo(3, 5);
    expect(layer.renderedCount).toBe(2);
  });

  it("filters local map entries by selected sign when order no longer matters", () => {
    const indices = new Uint16Array([10, 11, 12, 13]);
    const values = new Float32Array([-4, 2, 0, -1]);

    expect(influenceEntriesForBackground(indices, values, "abs")).toEqual({ indices, values });
    expect(influenceEntriesForBackground(indices, values, "pos")).toEqual({
      indices: [11],
      values: [2],
    });
    expect(influenceEntriesForBackground(indices, values, "neg")).toEqual({
      indices: [10, 13],
      values: [-4, -1],
    });
  });
});

describe("influence map interpolation methods", () => {
  it("linearly reproduces triangle vertices and interpolates triangle centers", () => {
    const field = expectRaster(computeLinearInfluenceField({
      points: new Float32Array([0, 0, 1, 0, 0, 1]),
      dim: 2,
      bounds,
      viewport,
      indices: new Uint16Array([0, 1, 2]),
      values: new Float32Array([0, 2, 4]),
      gridWidth: 101,
      gridHeight: 101,
    }));

    expect(sample(field, 0, 100)).toBeCloseTo(0, 5);
    expect(sample(field, 100, 100)).toBeCloseTo(2, 5);
    expect(sample(field, 0, 0)).toBeCloseTo(4, 5);
    expect(sample(field, 33, 67)).toBeCloseTo(2, 1);
  });

  it("falls back to Cells when fewer than three linear samples are available", () => {
    const layer = computeLinearInfluenceField({
      points: new Float32Array([0, 0.5, 1, 0.5]),
      dim: 2,
      bounds,
      viewport,
      indices: new Uint16Array([0, 1]),
      values: new Float32Array([-1, 1]),
      gridWidth: 101,
      gridHeight: 101,
    });

    expect(layer.kind).toBe("cells");
    expect(layer.renderedCount).toBe(2);
    expect((layer.kind === "cells" ? layer.cellCount : 0)).toBe(2);
  });

  it("falls back to Cells when Delaunay triangles are too sparse", () => {
    const layer = computeLinearInfluenceField({
      points: new Float32Array([0, 0, 0.01, 0, 1, 1]),
      dim: 2,
      bounds,
      viewport,
      indices: new Uint16Array([0, 1, 2]),
      values: new Float32Array([0, 1, 10]),
      gridWidth: 101,
      gridHeight: 101,
    });

    expect(layer.kind).toBe("cells");
    expect((layer.kind === "cells" ? layer.cellCount : 0)).toBe(3);
  });

  it("Cells creates one rendered cell per unique valid influence sample", () => {
    const layer = computeCellsInfluenceLayer({
      points: new Float32Array([0.1, 0.1, 0.9, 0.9, 2, 2]),
      dim: 2,
      bounds,
      viewport,
      indices: new Uint16Array([0, 1, 2]),
      values: new Float32Array([1, -2, 4]),
    });

    expect(layer.renderedCount).toBe(2);
    expect(layer.cellCount).toBe(2);
    expect(layer.maxAbs).toBe(2);
  });

  it("robust color scaling handles outliers and all-zero fields", () => {
    const mostlyOne = new Float32Array(100).fill(1);
    mostlyOne[99] = 100;

    expect(robustAbsScaleMax(mostlyOne)).toBe(1);
    expect(robustAbsScaleMax(new Float32Array(32))).toBe(1);
  });

  it("uses a red endpoint for strong negative diverging map values", () => {
    const color = divergingColorScale([-1, 1]);

    expect(color(-1)).toBe("rgb(178, 24, 43)");
    expect(color(0)).toBe("rgb(247, 247, 247)");
  });
});
