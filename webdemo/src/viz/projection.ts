import type { Bounds, PlotViewport, RunManifest } from "../types";
import {
  normalizeRegionBounds,
  normalizedBounds,
  unprojectPointFromViewport,
} from "./geometry";

export interface AxisLabels {
  x: string;
  y: string;
}

export interface PlotProjection {
  physicalBounds: Bounds;
  displayBounds: Bounds;
  labels: AxisLabels;
  projectPoint: (x: number, y: number) => [number, number];
  unprojectPoint: (x: number, y: number) => [number, number];
  formatXTick: (value: number) => string;
  formatYTick: (value: number) => string;
  xTickValues?: number[];
}

const TIME_PROBLEMS = new Set([
  "allen_cahn",
  "burgers",
  "diffusion",
  "drift_diffusion",
  "wave",
]);

const SPACE_PROBLEMS = new Set(["poisson_disk", "navier_stokes_nd"]);

function formatCompactNumber(value: number): string {
  if (!Number.isFinite(value)) return "";
  return value.toLocaleString("en-US", {
    maximumFractionDigits: Math.abs(value) < 10 ? 3 : 2,
    minimumFractionDigits: 0,
  });
}

function piTickLabel(displayX: number): string {
  if (!Number.isFinite(displayX)) return "";
  const rounded = Math.round(displayX);
  if (Math.abs(displayX - rounded) < 1e-6) {
    if (rounded === 0) return "0";
    if (rounded === 1) return "π";
    return `${rounded}π`;
  }
  return `${formatCompactNumber(displayX)}π`;
}

export function axisLabelsForProblem(problem: string, axes: string[] = ["x", "y"]): AxisLabels {
  if (TIME_PROBLEMS.has(problem)) return { x: "x", y: "t" };
  if (SPACE_PROBLEMS.has(problem)) return { x: "x", y: "y" };
  return {
    x: axes[0] ?? "x",
    y: axes[1] ?? "y",
  };
}

function driftDiffusionProjection(
  physicalBounds: Bounds,
  labels: AxisLabels,
): PlotProjection {
  const safeBounds = normalizedBounds(physicalBounds);
  const xScale = Math.PI;
  const displayBounds = {
    minX: safeBounds.minX / xScale,
    maxX: safeBounds.maxX / xScale,
    minY: safeBounds.minY,
    maxY: safeBounds.maxY,
  };
  const minTick = Math.ceil(displayBounds.minX);
  const maxTick = Math.floor(displayBounds.maxX);
  return {
    physicalBounds: safeBounds,
    displayBounds,
    labels,
    projectPoint: (x, y) => [x / xScale, y],
    unprojectPoint: (x, y) => [x * xScale, y],
    formatXTick: piTickLabel,
    formatYTick: formatCompactNumber,
    xTickValues:
      maxTick >= minTick
        ? Array.from({ length: maxTick - minTick + 1 }, (_unused, index) => minTick + index)
        : undefined,
  };
}

export function plotProjectionForManifest(
  manifest: RunManifest,
  physicalBounds: Bounds,
): PlotProjection {
  const safeBounds = normalizedBounds(physicalBounds);
  const labels = axisLabelsForProblem(manifest.problem, manifest.axes);
  if (manifest.problem === "drift_diffusion") {
    return driftDiffusionProjection(safeBounds, labels);
  }
  return {
    physicalBounds: safeBounds,
    displayBounds: safeBounds,
    labels,
    projectPoint: (x, y) => [x, y],
    unprojectPoint: (x, y) => [x, y],
    formatXTick: formatCompactNumber,
    formatYTick: formatCompactNumber,
  };
}

export function projectBounds(bounds: Bounds, projection: PlotProjection): Bounds {
  const safeBounds = normalizedBounds(bounds);
  const min = projection.projectPoint(safeBounds.minX, safeBounds.minY);
  const max = projection.projectPoint(safeBounds.maxX, safeBounds.maxY);
  return normalizeRegionBounds({
    minX: min[0],
    maxX: max[0],
    minY: min[1],
    maxY: max[1],
  });
}

export function projectPointToDisplay(
  point: [number, number],
  projection: PlotProjection,
): [number, number] {
  return projection.projectPoint(point[0], point[1]);
}

export function regionBoundsFromProjectedViewportDrag(
  start: [number, number],
  end: [number, number],
  projection: PlotProjection,
  viewport: PlotViewport,
): Bounds {
  const startDisplay = unprojectPointFromViewport(
    start[0],
    start[1],
    projection.displayBounds,
    viewport,
  );
  const endDisplay = unprojectPointFromViewport(
    end[0],
    end[1],
    projection.displayBounds,
    viewport,
  );
  const startPhysical = projection.unprojectPoint(startDisplay[0], startDisplay[1]);
  const endPhysical = projection.unprojectPoint(endDisplay[0], endDisplay[1]);
  return normalizeRegionBounds({
    minX: startPhysical[0],
    maxX: endPhysical[0],
    minY: startPhysical[1],
    maxY: endPhysical[1],
  });
}
