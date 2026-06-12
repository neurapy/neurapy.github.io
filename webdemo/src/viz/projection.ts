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

function piMultipleTickLabel(multiple: number): string {
  if (!Number.isFinite(multiple)) return "";
  const rounded = Math.round(multiple);
  if (Math.abs(multiple - rounded) < 1e-6) {
    if (rounded === 0) return "0";
    if (rounded === 1) return "π";
    if (rounded === -1) return "-π";
    return `${rounded}π`;
  }
  return `${formatCompactNumber(multiple)}π`;
}

function physicalPiTickLabel(physicalX: number): string {
  return piMultipleTickLabel(physicalX / Math.PI);
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
  const xSpan = safeBounds.maxX - safeBounds.minX || 1;
  const ySpan = safeBounds.maxY - safeBounds.minY || 1;
  const projectX = (x: number) => ((x - safeBounds.minX) / xSpan) * ySpan;
  const unprojectX = (x: number) => safeBounds.minX + (x / ySpan) * xSpan;
  const displayBounds = {
    minX: 0,
    maxX: ySpan,
    minY: safeBounds.minY,
    maxY: safeBounds.maxY,
  };
  const minPiTick = Math.ceil(safeBounds.minX / Math.PI);
  const maxPiTick = Math.floor(safeBounds.maxX / Math.PI);
  return {
    physicalBounds: safeBounds,
    displayBounds,
    labels,
    projectPoint: (x, y) => [projectX(x), y],
    unprojectPoint: (x, y) => [unprojectX(x), y],
    formatXTick: (x) => physicalPiTickLabel(unprojectX(x)),
    formatYTick: formatCompactNumber,
    xTickValues:
      maxPiTick >= minPiTick
        ? Array.from({ length: maxPiTick - minPiTick + 1 }, (_unused, index) =>
            projectX((minPiTick + index) * Math.PI),
          )
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
