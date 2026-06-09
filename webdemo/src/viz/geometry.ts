import { scaleLinear } from "d3";
import type { AxisBounds, Bounds, PlotViewport } from "../types";

export const DEFAULT_PLOT_PADDING = 28;

function positiveSpan(a: number, b: number): number {
  const span = Number(b) - Number(a);
  return Number.isFinite(span) && span > 0 ? span : 1;
}

function finiteNumber(value: unknown, fallback: number): number {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

export function normalizedBounds(bounds: Partial<Bounds> | null | undefined): Bounds {
  return {
    minX: finiteNumber(bounds?.minX, 0),
    maxX: finiteNumber(bounds?.maxX, 1),
    minY: finiteNumber(bounds?.minY, 0),
    maxY: finiteNumber(bounds?.maxY, 1),
  };
}

export function normalizeRegionBounds(bounds: Partial<Bounds> | null | undefined): Bounds {
  const safeBounds = normalizedBounds(bounds);
  return {
    minX: Math.min(safeBounds.minX, safeBounds.maxX),
    maxX: Math.max(safeBounds.minX, safeBounds.maxX),
    minY: Math.min(safeBounds.minY, safeBounds.maxY),
    maxY: Math.max(safeBounds.minY, safeBounds.maxY),
  };
}

export function boundsFromAxisMap(axisBounds: AxisBounds | null | undefined, axes = ["x", "y"]): Bounds {
  const xBounds = axisBounds?.[axes[0]] ?? axisBounds?.x ?? [0, 1];
  const yBounds = axisBounds?.[axes[1]] ?? axisBounds?.y ?? [0, 1];
  return normalizedBounds({
    minX: xBounds[0],
    maxX: xBounds[1],
    minY: yBounds[0],
    maxY: yBounds[1],
  });
}

export function inferPointBounds(points: Float32Array, dim: number): Bounds {
  if (!points.length) return normalizedBounds(null);
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let index = 0; index < points.length / dim; index += 1) {
    const x = points[index * dim];
    const y = points[index * dim + 1] ?? 0;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  return normalizedBounds({ minX, maxX, minY, maxY });
}

export function domainAspectRatio(bounds: Partial<Bounds> | null | undefined): number {
  const safeBounds = normalizedBounds(bounds);
  return (
    positiveSpan(safeBounds.minX, safeBounds.maxX) /
    positiveSpan(safeBounds.minY, safeBounds.maxY)
  );
}

export function plotViewport(
  bounds: Partial<Bounds> | null | undefined,
  canvasWidth: number,
  canvasHeight: number,
  padding = DEFAULT_PLOT_PADDING,
): PlotViewport {
  const width = Math.max(1, Number(canvasWidth) || 1);
  const height = Math.max(1, Number(canvasHeight) || 1);
  const inset = Math.min(
    Math.max(0, Number(padding) || 0),
    Math.max(0, (width - 1) / 2),
    Math.max(0, (height - 1) / 2),
  );
  const availableWidth = Math.max(1, width - inset * 2);
  const availableHeight = Math.max(1, height - inset * 2);
  const targetRatio = domainAspectRatio(bounds);
  const availableRatio = availableWidth / availableHeight;

  let viewportWidth = availableWidth;
  let viewportHeight = availableHeight;
  if (availableRatio > targetRatio) {
    viewportWidth = availableHeight * targetRatio;
  } else {
    viewportHeight = availableWidth / targetRatio;
  }

  const x = inset + (availableWidth - viewportWidth) / 2;
  const y = inset + (availableHeight - viewportHeight) / 2;
  return {
    x,
    y,
    width: viewportWidth,
    height: viewportHeight,
    right: x + viewportWidth,
    bottom: y + viewportHeight,
  };
}

export function fitScales(
  bounds: Partial<Bounds> | null | undefined,
  canvasWidth: number,
  canvasHeight: number,
  padding = DEFAULT_PLOT_PADDING,
) {
  const safeBounds = normalizedBounds(bounds);
  const viewport = plotViewport(safeBounds, canvasWidth, canvasHeight, padding);
  const x = scaleLinear()
    .domain([safeBounds.minX, safeBounds.maxX])
    .range([viewport.x, viewport.right]);
  const y = scaleLinear()
    .domain([safeBounds.minY, safeBounds.maxY])
    .range([viewport.bottom, viewport.y]);
  return { x, y, viewport, bounds: safeBounds };
}

export function projectPointToViewport(
  x: number,
  y: number,
  bounds: Partial<Bounds> | null | undefined,
  viewport: PlotViewport,
): [number, number] {
  const safeBounds = normalizedBounds(bounds);
  const spanX = positiveSpan(safeBounds.minX, safeBounds.maxX);
  const spanY = positiveSpan(safeBounds.minY, safeBounds.maxY);
  return [
    viewport.x + ((x - safeBounds.minX) / spanX) * viewport.width,
    viewport.y + viewport.height - ((y - safeBounds.minY) / spanY) * viewport.height,
  ];
}

export function containsViewportPoint(
  sx: number,
  sy: number,
  viewport: PlotViewport,
  tolerance = 0,
): boolean {
  return (
    sx >= viewport.x - tolerance &&
    sx <= viewport.right + tolerance &&
    sy >= viewport.y - tolerance &&
    sy <= viewport.bottom + tolerance
  );
}

export function unprojectPointFromViewport(
  sx: number,
  sy: number,
  bounds: Partial<Bounds> | null | undefined,
  viewport: PlotViewport,
): [number, number] {
  const safeBounds = normalizedBounds(bounds);
  const spanX = positiveSpan(safeBounds.minX, safeBounds.maxX);
  const spanY = positiveSpan(safeBounds.minY, safeBounds.maxY);
  const nx = Math.max(0, Math.min(1, (sx - viewport.x) / Math.max(1, viewport.width)));
  const ny = Math.max(
    0,
    Math.min(1, (viewport.y + viewport.height - sy) / Math.max(1, viewport.height)),
  );
  return [safeBounds.minX + nx * spanX, safeBounds.minY + ny * spanY];
}

export function regionBoundsFromViewportDrag(
  start: [number, number],
  end: [number, number],
  bounds: Partial<Bounds> | null | undefined,
  viewport: PlotViewport,
): Bounds {
  const startDomain = unprojectPointFromViewport(start[0], start[1], bounds, viewport);
  const endDomain = unprojectPointFromViewport(end[0], end[1], bounds, viewport);
  return normalizeRegionBounds({
    minX: startDomain[0],
    maxX: endDomain[0],
    minY: startDomain[1],
    maxY: endDomain[1],
  });
}

export function selectPointIndicesInBounds(
  points: Float32Array,
  dim: number,
  region: Partial<Bounds> | null | undefined,
  count = Math.floor(points.length / dim),
): number[] {
  const bounds = normalizeRegionBounds(region);
  const maxCount = Math.max(0, Math.min(Math.floor(points.length / dim), Math.trunc(count)));
  const indices: number[] = [];
  for (let index = 0; index < maxCount; index += 1) {
    const x = points[index * dim];
    const y = points[index * dim + 1] ?? 0;
    if (x >= bounds.minX && x <= bounds.maxX && y >= bounds.minY && y <= bounds.maxY) {
      indices.push(index);
    }
  }
  return indices;
}

export function pointAt(points: Float32Array, index: number, dim: number): [number, number] {
  const offset = index * dim;
  return [points[offset] ?? 0, points[offset + 1] ?? 0];
}

export function clampIndex(index: number, count: number): number {
  if (!Number.isFinite(index) || count <= 0) return 0;
  return Math.max(0, Math.min(count - 1, Math.trunc(index)));
}
