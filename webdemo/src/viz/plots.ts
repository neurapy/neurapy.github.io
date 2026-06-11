import {
  Delaunay,
  axisBottom,
  axisLeft,
  color as parseD3Color,
  format,
  pointer,
  select,
} from "d3";
import type {
  BackgroundMode,
  Bounds,
  InfluenceAggregate,
  InfluenceMatrixManifest,
  InfluenceRow,
  InfluenceSign,
  PointArrays,
  PlotViewport,
  RasterData,
  RunManifest,
} from "../types";
import { clearCanvas, prepareCanvas } from "./canvas";
import { divergingColorScale } from "./color";
import {
  boundsFromAxisMap,
  clampIndex,
  fitScales,
  inferPointBounds,
  pointAt,
  plotViewport,
  projectPointToViewport,
} from "./geometry";
import { plotVisualScale, scaledPlotPx } from "./scale";

export interface RasterRenderResult {
  image: HTMLCanvasElement;
  decoded: Float32Array;
  contourValues: number[];
  contourPaths: string[];
}

export interface PlotContext {
  manifest: RunManifest;
  points: PointArrays;
  bounds: Bounds;
  candidateDim: number;
  trainDim: number;
}

export interface InfluenceRenderStats {
  maxAbs: number;
  renderedCount: number;
  backgroundMode: BackgroundMode;
  scaleMax: number;
}

export interface InfluenceField {
  kind: "raster";
  width: number;
  height: number;
  values: Float32Array;
  support: Float32Array;
  maxAbs: number;
  scaleMax: number;
  renderedCount: number;
}

export interface InfluenceSample {
  x: number;
  y: number;
  value: number;
}

export interface CellsInfluenceLayer {
  kind: "cells";
  width: number;
  height: number;
  samples: InfluenceSample[];
  maxAbs: number;
  scaleMax: number;
  renderedCount: number;
  cellCount: number;
}

export type InfluenceMapLayer = InfluenceField | CellsInfluenceLayer;

interface InfluenceFieldArgs {
  points: Float32Array;
  dim: number;
  bounds: Bounds;
  viewport: PlotViewport;
  indices: ArrayLike<number>;
  values: ArrayLike<number>;
  gridWidth?: number;
  gridHeight?: number;
}

interface InfluenceSampleSet {
  width: number;
  height: number;
  samples: InfluenceSample[];
  renderedCount: number;
}

const BASE_INFLUENCE_POINT_SIZE = 2.8;
const MIN_TOP_K_POINT_SIZE = 4.2;
const MAX_TOP_K_POINT_SIZE = 10.5;
const MIN_MAP_GRID_CELL_SIZE_PX = 2;
const MAX_MAP_GRID_CELL_SIZE_PX = 6;
const MAX_MAP_GRID_CELLS = 50_000;
const DUPLICATE_MERGE_TOLERANCE_GRID_PX = 0.25;
export const MAX_VISIBLE_INFLUENCE_LINES = 64;

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function emptyInfluenceStats(backgroundMode: BackgroundMode): InfluenceRenderStats {
  return { maxAbs: 0, renderedCount: 0, backgroundMode, scaleMax: 1 };
}

function maxAbsValue(values: ArrayLike<number>, count = values.length): number {
  let maxAbs = 0;
  for (let index = 0; index < count; index += 1) {
    const value = values[index];
    if (Number.isFinite(value)) maxAbs = Math.max(maxAbs, Math.abs(value));
  }
  return maxAbs;
}

function sliceArrayLike(values: ArrayLike<number>, count: number): ArrayLike<number> {
  const typedValues = values as ArrayLike<number> & {
    subarray?: (start: number, end?: number) => ArrayLike<number>;
  };
  if (typeof typedValues.subarray === "function") return typedValues.subarray(0, count);
  return Array.from({ length: count }, (_unused, index) => values[index]);
}

export function topKInfluenceEntries(
  indices: ArrayLike<number>,
  values: ArrayLike<number>,
  k: number,
): { indices: ArrayLike<number>; values: ArrayLike<number>; count: number } {
  const count = Math.min(Math.max(0, Math.trunc(k)), indices.length, values.length);
  return {
    indices: sliceArrayLike(indices, count),
    values: sliceArrayLike(values, count),
    count,
  };
}

export function topKInfluenceLineEntries(
  indices: ArrayLike<number>,
  values: ArrayLike<number>,
  k: number,
): { indices: ArrayLike<number>; values: ArrayLike<number>; count: number } {
  return topKInfluenceEntries(indices, values, Math.min(k, MAX_VISIBLE_INFLUENCE_LINES));
}

function influenceStrength(value: number, scaleMax: number): number {
  return scaleMax > 0 ? Math.min(1, Math.abs(value) / scaleMax) : 0;
}

function influenceRgba(value: number, scaleMax: number, alpha: number): string {
  const color = divergingColorScale([-scaleMax, scaleMax]);
  const parsed = parseD3Color(color(value));
  if (!parsed) return `rgba(82, 96, 112, ${alpha})`;
  const rgb = parsed.rgb();
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${alpha})`;
}

function influenceGridSize(args: {
  viewport: PlotViewport;
  gridWidth?: number;
  gridHeight?: number;
}): { width: number; height: number } {
  if (args.gridWidth || args.gridHeight) {
    return {
      width: Math.max(
        1,
        Math.round(args.gridWidth ?? args.viewport.width / MIN_MAP_GRID_CELL_SIZE_PX),
      ),
      height: Math.max(
        1,
        Math.round(args.gridHeight ?? args.viewport.height / MIN_MAP_GRID_CELL_SIZE_PX),
      ),
    };
  }
  const area = Math.max(1, args.viewport.width * args.viewport.height);
  const cellSize = clampNumber(
    Math.sqrt(area / MAX_MAP_GRID_CELLS),
    MIN_MAP_GRID_CELL_SIZE_PX,
    MAX_MAP_GRID_CELL_SIZE_PX,
  );
  return {
    width: Math.max(1, Math.ceil(args.viewport.width / cellSize)),
    height: Math.max(1, Math.ceil(args.viewport.height / cellSize)),
  };
}

function gridCoordFromViewport(
  sx: number,
  sy: number,
  viewport: PlotViewport,
  width: number,
  height: number,
): [number, number] {
  return [
    width === 1 ? 0 : ((sx - viewport.x) / Math.max(1, viewport.width)) * (width - 1),
    height === 1 ? 0 : ((sy - viewport.y) / Math.max(1, viewport.height)) * (height - 1),
  ];
}

function collectInfluenceSamples(args: InfluenceFieldArgs): InfluenceSampleSet {
  const { width, height } = influenceGridSize(args);
  const dim = Math.max(1, args.dim);
  const trainCount = Math.floor(args.points.length / dim);
  const count = Math.min(args.indices.length, args.values.length);
  const tolerance = DUPLICATE_MERGE_TOLERANCE_GRID_PX;
  const toleranceSq = tolerance * tolerance;
  const buckets = new Map<string, number[]>();
  const groups: Array<{
    xSum: number;
    ySum: number;
    valueSum: number;
    count: number;
  }> = [];
  let renderedCount = 0;

  const bucketKey = (bucketX: number, bucketY: number): string => `${bucketX},${bucketY}`;
  const addToBucket = (groupIndex: number, x: number, y: number) => {
    const bucketX = Math.floor(x / tolerance);
    const bucketY = Math.floor(y / tolerance);
    const key = bucketKey(bucketX, bucketY);
    const bucket = buckets.get(key);
    if (bucket) {
      bucket.push(groupIndex);
    } else {
      buckets.set(key, [groupIndex]);
    }
  };

  for (let entry = 0; entry < count; entry += 1) {
    const pointIndex = Math.trunc(args.indices[entry]);
    const value = args.values[entry];
    if (!Number.isFinite(value) || pointIndex < 0 || pointIndex >= trainCount) continue;
    const [x, y] = pointAt(args.points, pointIndex, dim);
    const [sx, sy] = projectPointToViewport(x, y, args.bounds, args.viewport);
    if (
      sx < args.viewport.x ||
      sx > args.viewport.right ||
      sy < args.viewport.y ||
      sy > args.viewport.bottom
    ) {
      continue;
    }
    const [gx, gy] = gridCoordFromViewport(sx, sy, args.viewport, width, height);
    renderedCount += 1;

    const bucketX = Math.floor(gx / tolerance);
    const bucketY = Math.floor(gy / tolerance);
    let mergeIndex = -1;
    for (let ny = bucketY - 1; ny <= bucketY + 1 && mergeIndex < 0; ny += 1) {
      for (let nx = bucketX - 1; nx <= bucketX + 1 && mergeIndex < 0; nx += 1) {
        const bucket = buckets.get(bucketKey(nx, ny));
        if (!bucket) continue;
        for (const groupIndex of bucket) {
          const group = groups[groupIndex];
          const groupX = group.xSum / group.count;
          const groupY = group.ySum / group.count;
          const dx = gx - groupX;
          const dy = gy - groupY;
          if (dx * dx + dy * dy <= toleranceSq) {
            mergeIndex = groupIndex;
            break;
          }
        }
      }
    }

    if (mergeIndex >= 0) {
      const group = groups[mergeIndex];
      group.xSum += gx;
      group.ySum += gy;
      group.valueSum += value;
      group.count += 1;
      continue;
    }

    const groupIndex = groups.length;
    groups.push({ xSum: gx, ySum: gy, valueSum: value, count: 1 });
    addToBucket(groupIndex, gx, gy);
  }

  return {
    width,
    height,
    renderedCount,
    samples: groups.map((group) => ({
      x: group.xSum / group.count,
      y: group.ySum / group.count,
      value: group.valueSum / group.count,
    })),
  };
}

function median(values: number[]): number {
  if (!values.length) return 0;
  values.sort((a, b) => a - b);
  const mid = Math.floor(values.length / 2);
  return values.length % 2 ? values[mid] : (values[mid - 1] + values[mid]) / 2;
}

export function robustAbsScaleMax(
  values: ArrayLike<number>,
  support?: ArrayLike<number>,
  minSupport = 0,
): number {
  const absValues: number[] = [];
  let maxAbs = 0;
  for (let index = 0; index < values.length; index += 1) {
    if (support && support[index] <= minSupport) continue;
    const value = values[index];
    if (!Number.isFinite(value)) continue;
    const abs = Math.abs(value);
    maxAbs = Math.max(maxAbs, abs);
    if (abs > 0) absValues.push(abs);
  }
  if (!absValues.length) return maxAbs > 0 ? maxAbs : 1;
  absValues.sort((a, b) => a - b);
  const percentileIndex = Math.min(
    absValues.length - 1,
    Math.max(0, Math.floor((absValues.length - 1) * 0.98)),
  );
  const scaleMax = absValues[percentileIndex];
  return Number.isFinite(scaleMax) && scaleMax > 0 ? scaleMax : maxAbs > 0 ? maxAbs : 1;
}

function finalizeInfluenceField(args: {
  width: number;
  height: number;
  values: Float32Array;
  support: Float32Array;
  renderedCount: number;
  minSupport?: number;
}): InfluenceField {
  const minSupport = args.minSupport ?? 0;
  let maxAbs = 0;
  for (let index = 0; index < args.values.length; index += 1) {
    if (args.support[index] <= minSupport) continue;
    const value = args.values[index];
    if (Number.isFinite(value)) maxAbs = Math.max(maxAbs, Math.abs(value));
  }
  return {
    kind: "raster",
    width: args.width,
    height: args.height,
    values: args.values,
    support: args.support,
    maxAbs,
    scaleMax: robustAbsScaleMax(args.values, args.support, minSupport),
    renderedCount: args.renderedCount,
  };
}

function exactCellSampleArgs(args: InfluenceFieldArgs): InfluenceFieldArgs {
  return {
    ...args,
    gridWidth: Math.max(1, Math.round(args.gridWidth ?? args.viewport.width)),
    gridHeight: Math.max(1, Math.round(args.gridHeight ?? args.viewport.height)),
  };
}

export function computeCellsInfluenceLayer(args: InfluenceFieldArgs): CellsInfluenceLayer {
  const sampleSet = collectInfluenceSamples(exactCellSampleArgs(args));
  const values = Float32Array.from(sampleSet.samples, (sample) => sample.value);
  return {
    kind: "cells",
    width: sampleSet.width,
    height: sampleSet.height,
    samples: sampleSet.samples,
    maxAbs: maxAbsValue(values),
    scaleMax: robustAbsScaleMax(values),
    renderedCount: sampleSet.renderedCount,
    cellCount: sampleSet.samples.length,
  };
}

function delaunayEdgeSpacing(
  samples: InfluenceSample[],
  delaunay: Delaunay<InfluenceSample>,
  width: number,
  height: number,
): number {
  const nearest = new Float32Array(samples.length);
  nearest.fill(Infinity);
  const seen = new Set<string>();
  const addEdge = (left: number, right: number) => {
    if (left === right) return;
    const min = Math.min(left, right);
    const max = Math.max(left, right);
    const key = `${min}:${max}`;
    if (seen.has(key)) return;
    seen.add(key);
    const a = samples[min];
    const b = samples[max];
    if (!a || !b) return;
    const length = Math.hypot(a.x - b.x, a.y - b.y);
    nearest[left] = Math.min(nearest[left], length);
    nearest[right] = Math.min(nearest[right], length);
  };

  const triangles = delaunay.triangles;
  for (let triangle = 0; triangle < triangles.length; triangle += 3) {
    addEdge(triangles[triangle], triangles[triangle + 1]);
    addEdge(triangles[triangle + 1], triangles[triangle + 2]);
    addEdge(triangles[triangle + 2], triangles[triangle]);
  }
  const distances = Array.from(nearest).filter((distance) => Number.isFinite(distance));
  return median(distances) || Math.sqrt(Math.max(1, width * height) / Math.max(1, samples.length));
}

function triangleLongestEdge(a: InfluenceSample, b: InfluenceSample, c: InfluenceSample): number {
  const ab = Math.hypot(a.x - b.x, a.y - b.y);
  const bc = Math.hypot(b.x - c.x, b.y - c.y);
  const ca = Math.hypot(c.x - a.x, c.y - a.y);
  return Math.max(ab, bc, ca);
}

function barycentricWeights(
  x: number,
  y: number,
  a: InfluenceSample,
  b: InfluenceSample,
  c: InfluenceSample,
): [number, number, number] | null {
  const denominator = (b.y - c.y) * (a.x - c.x) + (c.x - b.x) * (a.y - c.y);
  if (Math.abs(denominator) < 1e-12) return null;
  const wa = ((b.y - c.y) * (x - c.x) + (c.x - b.x) * (y - c.y)) / denominator;
  const wb = ((c.y - a.y) * (x - c.x) + (a.x - c.x) * (y - c.y)) / denominator;
  return [wa, wb, 1 - wa - wb];
}

export function computeLinearInfluenceField(args: InfluenceFieldArgs): InfluenceMapLayer {
  const sampleSet = collectInfluenceSamples(args);
  const { width, height, samples, renderedCount } = sampleSet;
  if (samples.length < 3) {
    return computeCellsInfluenceLayer(args);
  }

  const values = new Float32Array(width * height);
  const support = new Float32Array(width * height);
  const delaunay = Delaunay.from(
    samples,
    (sample) => sample.x,
    (sample) => sample.y,
  );
  const triangles = delaunay.triangles;
  if (!triangles.length) return computeCellsInfluenceLayer(args);
  const spacing = delaunayEdgeSpacing(samples, delaunay, width, height);
  const maxTriangleEdge = spacing > 0 ? spacing * 2.5 : Infinity;

  for (let triangle = 0; triangle < triangles.length; triangle += 3) {
    const a = samples[triangles[triangle]];
    const b = samples[triangles[triangle + 1]];
    const c = samples[triangles[triangle + 2]];
    if (!a || !b || !c) continue;
    if (triangleLongestEdge(a, b, c) > maxTriangleEdge) continue;
    const minCol = Math.max(0, Math.floor(Math.min(a.x, b.x, c.x)));
    const maxCol = Math.min(width - 1, Math.ceil(Math.max(a.x, b.x, c.x)));
    const minRow = Math.max(0, Math.floor(Math.min(a.y, b.y, c.y)));
    const maxRow = Math.min(height - 1, Math.ceil(Math.max(a.y, b.y, c.y)));
    for (let row = minRow; row <= maxRow; row += 1) {
      const rowOffset = row * width;
      for (let col = minCol; col <= maxCol; col += 1) {
        const weights = barycentricWeights(col, row, a, b, c);
        if (!weights) continue;
        const [wa, wb, wc] = weights;
        if (wa < -1e-6 || wb < -1e-6 || wc < -1e-6) continue;
        const offset = rowOffset + col;
        values[offset] = wa * a.value + wb * b.value + wc * c.value;
        support[offset] = 1;
      }
    }
  }

  const field = finalizeInfluenceField({ width, height, values, support, renderedCount });
  return field.maxAbs > 0 ? field : computeCellsInfluenceLayer(args);
}

export function resizeSvg(svg: SVGSVGElement, width: number, height: number): void {
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", `${width}`);
  svg.setAttribute("height", `${height}`);
}

export function renderAxes(
  svg: SVGSVGElement,
  bounds: Bounds,
  width: number,
  height: number,
  viewport: PlotViewport,
): void {
  resizeSvg(svg, width, height);
  const visualScale = plotVisualScale(viewport);
  svg.style.setProperty("--plot-visual-scale", String(visualScale));
  svg.style.setProperty("--plot-axis-stroke-width", `${visualScale}px`);
  svg.style.setProperty("--plot-contour-stroke-width", `${0.7 * visualScale}px`);
  const { x, y } = fitScales(bounds, width, height);
  const root = select(svg);
  root.selectAll("*").remove();
  root
    .append("rect")
    .attr("class", "axis-frame")
    .attr("x", viewport.x)
    .attr("y", viewport.y)
    .attr("width", viewport.width)
    .attr("height", viewport.height);
  root
    .append("g")
    .attr("class", "axis axis-x")
    .attr("transform", `translate(0,${viewport.bottom})`)
    .call(axisBottom(x).ticks(Math.max(3, Math.floor(width / 160))).tickFormat(format(".2~g")));
  root
    .append("g")
    .attr("class", "axis axis-y")
    .attr("transform", `translate(${viewport.x},0)`)
    .call(axisLeft(y).ticks(Math.max(3, Math.floor(height / 150))).tickFormat(format(".2~g")));
}

function boundsSpan(min: number, max: number): number {
  const span = max - min;
  return Number.isFinite(span) && span > 0 ? span : 1;
}

function rasterPlotBounds(context: PlotContext): Bounds {
  return context.manifest.field_raster
    ? boundsFromAxisMap(context.manifest.field_raster.bounds, context.manifest.field_raster.axes)
    : context.bounds;
}

function renderContourOverlay(args: {
  svg: SVGSVGElement;
  raster: RasterData | null;
  rasterResult: RasterRenderResult | null;
  rasterBounds: Bounds;
  targetBounds: Bounds;
  viewport: PlotViewport;
}): void {
  if (!args.raster || !args.rasterResult?.contourPaths.length) return;
  const targetSpanX = boundsSpan(args.targetBounds.minX, args.targetBounds.maxX);
  const targetSpanY = boundsSpan(args.targetBounds.minY, args.targetBounds.maxY);
  const rasterSpanX = boundsSpan(args.rasterBounds.minX, args.rasterBounds.maxX);
  const rasterSpanY = boundsSpan(args.rasterBounds.minY, args.rasterBounds.maxY);
  const scaleX =
    (rasterSpanX / targetSpanX) * (args.viewport.width / Math.max(1, args.raster.width));
  const scaleY =
    (rasterSpanY / targetSpanY) * (args.viewport.height / Math.max(1, args.raster.height));
  const offsetX =
    args.viewport.x +
    ((args.rasterBounds.minX - args.targetBounds.minX) / targetSpanX) * args.viewport.width;
  const offsetY =
    args.viewport.y +
    ((args.targetBounds.maxY - args.rasterBounds.maxY) / targetSpanY) * args.viewport.height;
  const root = select(args.svg)
    .append("g")
    .attr("class", "contours")
    .attr("transform", `translate(${offsetX},${offsetY}) scale(${scaleX},${scaleY})`);
  root
    .selectAll("path")
    .data(args.rasterResult.contourPaths)
    .join("path")
    .attr("d", (pathValue) => pathValue);
}

export function drawPointMarker(
  ctx: CanvasRenderingContext2D,
  sx: number,
  sy: number,
  radius: number,
  visualScale = 1,
): void {
  const scaledRadius = Math.max(0, radius * visualScale);
  ctx.beginPath();
  ctx.arc(sx, sy, scaledRadius, 0, Math.PI * 2);
  ctx.fillStyle = "#f2b84b";
  ctx.fill();
  ctx.lineWidth = 2 * visualScale;
  ctx.strokeStyle = "#182230";
  ctx.stroke();
}

export function drawPointCloudLayer(
  ctx: CanvasRenderingContext2D,
  points: Float32Array,
  dim: number,
  bounds: Bounds,
  viewport: PlotViewport,
  options: {
    color?: string;
    alpha?: number;
    size?: number;
    maxPoints?: number;
  } = {},
): void {
  const count = Math.floor(points.length / dim);
  const stride = options.maxPoints && count > options.maxPoints ? Math.ceil(count / options.maxPoints) : 1;
  const visualScale = plotVisualScale(viewport);
  const size =
    options.size === undefined
      ? Math.max(
          1.5 * visualScale,
          Math.min(
            3.5 * visualScale,
            Math.sqrt((viewport.width * viewport.height) / Math.max(1, count)) * 0.2,
          ),
        )
      : options.size * visualScale;
  ctx.save();
  ctx.globalAlpha = options.alpha ?? 0.18;
  ctx.fillStyle = options.color ?? "#364252";
  for (let index = 0; index < count; index += stride) {
    const [x, y] = pointAt(points, index, dim);
    const [sx, sy] = projectPointToViewport(x, y, bounds, viewport);
    ctx.beginPath();
    ctx.arc(sx, sy, size / 2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

export function renderMainPlot(args: {
  canvas: HTMLCanvasElement;
  svg: SVGSVGElement;
  context: PlotContext;
  raster: RasterData | null;
  rasterResult: RasterRenderResult | null;
  selectedCoord: [number, number] | null;
  selectedRegion: Bounds | null;
  draftRegion: Bounds | null;
  showCandidatePoints: boolean;
  showTrainPoints: boolean;
}): PlotViewport {
  const { ctx, width, height } = prepareCanvas(args.canvas);
  clearCanvas(ctx, width, height);
  const rasterBounds = rasterPlotBounds(args.context);
  const viewport = plotViewport(rasterBounds, width, height);

  if (args.raster && args.rasterResult) {
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(args.rasterResult.image, viewport.x, viewport.y, viewport.width, viewport.height);
  }
  if (args.showTrainPoints) {
    drawPointCloudLayer(ctx, args.context.points.train_points, args.context.trainDim, rasterBounds, viewport, {
      color: "#11263a",
      alpha: 0.12,
      maxPoints: 6000,
    });
  }
  if (args.showCandidatePoints) {
    drawPointCloudLayer(
      ctx,
      args.context.points.candidate_points,
      args.context.candidateDim,
      rasterBounds,
      viewport,
      {
        color: "#0c7c78",
        alpha: 0.16,
        maxPoints: 6000,
      },
    );
  }
  if (args.selectedCoord) {
    const [sx, sy] = projectPointToViewport(args.selectedCoord[0], args.selectedCoord[1], rasterBounds, viewport);
    drawPointMarker(ctx, sx, sy, 7, plotVisualScale(viewport));
  }
  if (args.selectedRegion) {
    drawRegionOverlay(ctx, args.selectedRegion, rasterBounds, viewport, false);
  }
  if (args.draftRegion) {
    drawRegionOverlay(ctx, args.draftRegion, rasterBounds, viewport, true);
  }

  renderAxes(args.svg, rasterBounds, width, height, viewport);
  renderContourOverlay({
    svg: args.svg,
    raster: args.raster,
    rasterResult: args.rasterResult,
    rasterBounds,
    targetBounds: rasterBounds,
    viewport,
  });
  return viewport;
}

function drawRegionOverlay(
  ctx: CanvasRenderingContext2D,
  region: Bounds,
  bounds: Bounds,
  viewport: PlotViewport,
  draft: boolean,
): void {
  const [x0, y0] = projectPointToViewport(region.minX, region.maxY, bounds, viewport);
  const [x1, y1] = projectPointToViewport(region.maxX, region.minY, bounds, viewport);
  const x = Math.min(x0, x1);
  const y = Math.min(y0, y1);
  const width = Math.abs(x1 - x0);
  const height = Math.abs(y1 - y0);
  const visualScale = plotVisualScale(viewport);
  ctx.save();
  ctx.fillStyle = draft ? "rgba(242, 184, 75, 0.16)" : "rgba(12, 124, 120, 0.14)";
  ctx.strokeStyle = draft ? "#f2b84b" : "#0c7c78";
  ctx.lineWidth = (draft ? 1.4 : 2) * visualScale;
  ctx.setLineDash(draft ? [6 * visualScale, 4 * visualScale] : []);
  ctx.fillRect(x, y, width, height);
  ctx.strokeRect(x, y, width, height);
  ctx.restore();
}

function drawInfluencePointLayer(args: {
  ctx: CanvasRenderingContext2D;
  context: PlotContext;
  viewport: PlotViewport;
  indices: ArrayLike<number>;
  values: ArrayLike<number>;
  scaleMax: number;
  size?: number;
}): number {
  const color = divergingColorScale([-args.scaleMax, args.scaleMax]);
  const count = Math.min(args.indices.length, args.values.length);
  const trainCount = Math.floor(args.context.points.train_points.length / args.context.trainDim);
  const size = scaledPlotPx(args.size ?? BASE_INFLUENCE_POINT_SIZE, args.viewport);
  let renderedCount = 0;
  args.ctx.save();
  for (let index = 0; index < count; index += 1) {
    const trainIndex = Math.trunc(args.indices[index]);
    const value = args.values[index];
    if (!Number.isFinite(value) || trainIndex < 0 || trainIndex >= trainCount) continue;
    const trainPoint = pointAt(args.context.points.train_points, trainIndex, args.context.trainDim);
    const [sx, sy] = projectPointToViewport(trainPoint[0], trainPoint[1], args.context.bounds, args.viewport);
    if (
      sx < args.viewport.x ||
      sx > args.viewport.right ||
      sy < args.viewport.y ||
      sy > args.viewport.bottom
    ) {
      continue;
    }
    args.ctx.beginPath();
    args.ctx.arc(sx, sy, size / 2, 0, Math.PI * 2);
    args.ctx.fillStyle = color(value);
    args.ctx.fill();
    renderedCount += 1;
  }
  args.ctx.restore();
  return renderedCount;
}

function drawTopKInfluenceLinks(args: {
  ctx: CanvasRenderingContext2D;
  context: PlotContext;
  viewport: PlotViewport;
  indices: ArrayLike<number>;
  values: ArrayLike<number>;
  scaleMax: number;
  rowSx: number;
  rowSy: number;
}): void {
  const count = Math.min(args.indices.length, args.values.length);
  const trainCount = Math.floor(args.context.points.train_points.length / args.context.trainDim);
  args.ctx.save();
  args.ctx.globalAlpha = 0.42;
  args.ctx.strokeStyle = "#526070";
  for (let index = count - 1; index >= 0; index -= 1) {
    const trainIndex = Math.trunc(args.indices[index]);
    const value = args.values[index];
    if (!Number.isFinite(value) || trainIndex < 0 || trainIndex >= trainCount) continue;
    const trainPoint = pointAt(args.context.points.train_points, trainIndex, args.context.trainDim);
    const [sx, sy] = projectPointToViewport(trainPoint[0], trainPoint[1], args.context.bounds, args.viewport);
    const strength = influenceStrength(value, args.scaleMax);
    args.ctx.beginPath();
    args.ctx.moveTo(args.rowSx, args.rowSy);
    args.ctx.lineTo(sx, sy);
    args.ctx.lineWidth = scaledPlotPx(1 + strength * 1.6, args.viewport);
    args.ctx.stroke();
  }
  args.ctx.restore();
}

function drawTopKInfluencePoints(args: {
  ctx: CanvasRenderingContext2D;
  context: PlotContext;
  viewport: PlotViewport;
  indices: ArrayLike<number>;
  values: ArrayLike<number>;
  scaleMax: number;
}): void {
  const count = Math.min(args.indices.length, args.values.length);
  const trainCount = Math.floor(args.context.points.train_points.length / args.context.trainDim);
  args.ctx.save();
  for (let index = count - 1; index >= 0; index -= 1) {
    const trainIndex = Math.trunc(args.indices[index]);
    const value = args.values[index];
    if (!Number.isFinite(value) || trainIndex < 0 || trainIndex >= trainCount) continue;
    const trainPoint = pointAt(args.context.points.train_points, trainIndex, args.context.trainDim);
    const [sx, sy] = projectPointToViewport(trainPoint[0], trainPoint[1], args.context.bounds, args.viewport);
    const strength = influenceStrength(value, args.scaleMax);
    const size = scaledPlotPx(
      MIN_TOP_K_POINT_SIZE + strength * (MAX_TOP_K_POINT_SIZE - MIN_TOP_K_POINT_SIZE),
      args.viewport,
    );
    args.ctx.beginPath();
    args.ctx.arc(sx, sy, size / 2, 0, Math.PI * 2);
    args.ctx.fillStyle = influenceRgba(value, args.scaleMax, 0.94);
    args.ctx.fill();
    args.ctx.lineWidth = scaledPlotPx(1.1, args.viewport);
    args.ctx.strokeStyle = "rgba(24, 34, 48, 0.78)";
    args.ctx.stroke();
  }
  args.ctx.restore();
}

export function influenceEntriesForBackground(
  indices: ArrayLike<number>,
  values: ArrayLike<number>,
  sign: InfluenceSign,
): { indices: ArrayLike<number>; values: ArrayLike<number> } {
  if (sign === "abs") return { indices, values };
  const filteredIndices: number[] = [];
  const filteredValues: number[] = [];
  const count = Math.min(indices.length, values.length);
  for (let index = 0; index < count; index += 1) {
    const value = values[index];
    if ((sign === "pos" && value <= 0) || (sign === "neg" && value >= 0)) continue;
    filteredIndices.push(indices[index]);
    filteredValues.push(value);
  }
  return {
    indices: filteredIndices,
    values: filteredValues,
  };
}

function drawInfluenceFieldLayer(args: {
  ctx: CanvasRenderingContext2D;
  viewport: PlotViewport;
  field: InfluenceField;
}): void {
  const { field } = args;
  if (!field.renderedCount) return;

  const imageCanvas = document.createElement("canvas");
  imageCanvas.width = field.width;
  imageCanvas.height = field.height;
  const imageCtx = imageCanvas.getContext("2d");
  if (!imageCtx) return;
  const imageData = imageCtx.createImageData(field.width, field.height);
  const color = divergingColorScale([-field.scaleMax, field.scaleMax]);

  for (let index = 0; index < field.values.length; index += 1) {
    if (field.support[index] <= 0) continue;
    const value = field.values[index];
    const parsed = parseD3Color(color(value));
    if (!parsed) continue;
    const rgb = parsed.rgb();
    const rgbaOffset = index * 4;
    const strength = field.scaleMax ? Math.min(1, Math.abs(value) / field.scaleMax) : 0;
    imageData.data[rgbaOffset] = rgb.r;
    imageData.data[rgbaOffset + 1] = rgb.g;
    imageData.data[rgbaOffset + 2] = rgb.b;
    imageData.data[rgbaOffset + 3] = Math.round(132 + strength * 82);
  }

  imageCtx.putImageData(imageData, 0, 0);
  args.ctx.save();
  args.ctx.imageSmoothingEnabled = true;
  args.ctx.drawImage(
    imageCanvas,
    args.viewport.x,
    args.viewport.y,
    args.viewport.width,
    args.viewport.height,
  );
  args.ctx.restore();
}

function drawCellsInfluenceLayer(args: {
  ctx: CanvasRenderingContext2D;
  viewport: PlotViewport;
  field: CellsInfluenceLayer;
}): void {
  const { field } = args;
  if (!field.cellCount) return;
  const delaunay = Delaunay.from(
    field.samples,
    (sample) => sample.x,
    (sample) => sample.y,
  );
  const voronoi = delaunay.voronoi([0, 0, field.width, field.height]);
  const color = divergingColorScale([-field.scaleMax, field.scaleMax]);

  args.ctx.save();
  args.ctx.translate(args.viewport.x, args.viewport.y);
  args.ctx.scale(
    args.viewport.width / Math.max(1, field.width),
    args.viewport.height / Math.max(1, field.height),
  );
  for (let index = 0; index < field.samples.length; index += 1) {
    const polygon = voronoi.cellPolygon(index);
    if (!polygon?.length) continue;
    const sample = field.samples[index];
    const parsed = parseD3Color(color(sample.value));
    if (!parsed) continue;
    const rgb = parsed.rgb();
    const strength = field.scaleMax ? Math.min(1, Math.abs(sample.value) / field.scaleMax) : 0;
    args.ctx.beginPath();
    polygon.forEach(([x, y], pointIndex) => {
      if (pointIndex === 0) {
        args.ctx.moveTo(x, y);
      } else {
        args.ctx.lineTo(x, y);
      }
    });
    args.ctx.closePath();
    args.ctx.fillStyle = `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${0.5 + strength * 0.32})`;
    args.ctx.fill();
  }

  const fieldToCssScale = Math.sqrt(
    (args.viewport.width / Math.max(1, field.width)) *
      (args.viewport.height / Math.max(1, field.height)),
  );
  const cssToFieldScale = fieldToCssScale > 0 ? 1 / fieldToCssScale : 1;
  const markerSize = Math.max(
    scaledPlotPx(1, args.viewport) * cssToFieldScale,
    Math.min(
      scaledPlotPx(2.2, args.viewport) * cssToFieldScale,
      Math.sqrt((field.width * field.height) / Math.max(1, field.samples.length)) * 0.04,
    ),
  );
  args.ctx.globalAlpha = 0.42;
  args.ctx.fillStyle = "#17202a";
  for (const sample of field.samples) {
    args.ctx.fillRect(sample.x - markerSize / 2, sample.y - markerSize / 2, markerSize, markerSize);
  }
  args.ctx.restore();
}

function renderInfluenceBackground(args: {
  ctx: CanvasRenderingContext2D;
  context: PlotContext;
  viewport: PlotViewport;
  indices: ArrayLike<number>;
  values: ArrayLike<number>;
  backgroundMode: BackgroundMode;
}): InfluenceRenderStats {
  const maxAbs = maxAbsValue(args.values);
  if (args.backgroundMode === "points") {
    const scaleMax = robustAbsScaleMax(args.values);
    const renderedCount = drawInfluencePointLayer({
      ctx: args.ctx,
      context: args.context,
      viewport: args.viewport,
      indices: args.indices,
      values: args.values,
      scaleMax,
    });
    return { maxAbs, renderedCount, backgroundMode: args.backgroundMode, scaleMax };
  }

  const fieldArgs = {
    points: args.context.points.train_points,
    dim: args.context.trainDim,
    bounds: args.context.bounds,
    viewport: args.viewport,
    indices: args.indices,
    values: args.values,
  };
  const field =
    args.backgroundMode === "cell"
      ? computeCellsInfluenceLayer(fieldArgs)
      : computeLinearInfluenceField(fieldArgs);
  if (field.kind === "cells") {
    drawCellsInfluenceLayer({ ctx: args.ctx, viewport: args.viewport, field });
  } else {
    drawInfluenceFieldLayer({ ctx: args.ctx, viewport: args.viewport, field });
  }
  return {
    maxAbs: field.maxAbs,
    renderedCount: field.renderedCount,
    backgroundMode: args.backgroundMode,
    scaleMax: field.scaleMax,
  };
}

export function renderLocalInfluencePlot(args: {
  canvas: HTMLCanvasElement;
  svg: SVGSVGElement;
  context: PlotContext;
  raster: RasterData | null;
  rasterResult: RasterRenderResult | null;
  matrix: InfluenceMatrixManifest;
  row: InfluenceRow | null;
  selectedCandidateIndex: number;
  selectedTrainIndex: number;
  k: number;
  sign: InfluenceSign;
  backgroundMode: BackgroundMode;
}): InfluenceRenderStats {
  const { ctx, width, height } = prepareCanvas(args.canvas);
  clearCanvas(ctx, width, height);
  const viewport = plotViewport(args.context.bounds, width, height);
  renderAxes(args.svg, args.context.bounds, width, height, viewport);
  renderContourOverlay({
    svg: args.svg,
    raster: args.raster,
    rasterResult: args.rasterResult,
    rasterBounds: rasterPlotBounds(args.context),
    targetBounds: args.context.bounds,
    viewport,
  });
  drawPointCloudLayer(ctx, args.context.points.train_points, args.context.trainDim, args.context.bounds, viewport, {
    color: "#526070",
    alpha: 0.18,
    maxPoints: 10000,
    size: 2,
  });

  const rowSourcePoints =
    args.matrix.row_source === "train_points"
      ? args.context.points.train_points
      : args.context.points.candidate_points;
  const rowDim = args.matrix.row_source === "train_points" ? args.context.trainDim : args.context.candidateDim;
  const rowIndex =
    args.matrix.row_source === "train_points" ? args.selectedTrainIndex : args.selectedCandidateIndex;
  const rowPoint = pointAt(rowSourcePoints, clampIndex(rowIndex, args.matrix.row_count), rowDim);
  const [rowSx, rowSy] = projectPointToViewport(rowPoint[0], rowPoint[1], args.context.bounds, viewport);
  if (!args.row) {
    drawPointMarker(ctx, rowSx, rowSy, 7, plotVisualScale(viewport));
    return emptyInfluenceStats(args.backgroundMode);
  }
  const backgroundEntries = influenceEntriesForBackground(args.row.indices, args.row.values, args.sign);
  const backgroundStats = renderInfluenceBackground({
    ctx,
    context: args.context,
    viewport,
    indices: backgroundEntries.indices,
    values: backgroundEntries.values,
    backgroundMode: args.backgroundMode,
  });
  const { indices: topKIndices, values: topKValues } = topKInfluenceEntries(
    args.row.indices,
    args.row.values,
    args.k,
  );
  const { indices: topKLineIndices, values: topKLineValues } = topKInfluenceLineEntries(
    args.row.indices,
    args.row.values,
    args.k,
  );
  const topKMaxAbs = maxAbsValue(topKValues);
  const scaleMax = topKMaxAbs > 0 ? topKMaxAbs : backgroundStats.scaleMax;

  drawTopKInfluenceLinks({
    ctx,
    context: args.context,
    viewport,
    indices: topKLineIndices,
    values: topKLineValues,
    scaleMax,
    rowSx,
    rowSy,
  });
  drawTopKInfluencePoints({
    ctx,
    context: args.context,
    viewport,
    indices: topKIndices,
    values: topKValues,
    scaleMax,
  });

  drawPointMarker(ctx, rowSx, rowSy, 7, plotVisualScale(viewport));
  return { ...backgroundStats, scaleMax };
}

export function renderRegionalInfluencePlot(args: {
  canvas: HTMLCanvasElement;
  svg: SVGSVGElement;
  context: PlotContext;
  raster: RasterData | null;
  rasterResult: RasterRenderResult | null;
  aggregate: InfluenceAggregate | null;
  k: number;
  backgroundMode: BackgroundMode;
}): InfluenceRenderStats {
  const { ctx, width, height } = prepareCanvas(args.canvas);
  clearCanvas(ctx, width, height);
  const viewport = plotViewport(args.context.bounds, width, height);
  renderAxes(args.svg, args.context.bounds, width, height, viewport);
  renderContourOverlay({
    svg: args.svg,
    raster: args.raster,
    rasterResult: args.rasterResult,
    rasterBounds: rasterPlotBounds(args.context),
    targetBounds: args.context.bounds,
    viewport,
  });
  drawPointCloudLayer(ctx, args.context.points.train_points, args.context.trainDim, args.context.bounds, viewport, {
    color: "#526070",
    alpha: 0.18,
    maxPoints: 10000,
    size: 2,
  });

  if (!args.aggregate) return emptyInfluenceStats(args.backgroundMode);
  const backgroundStats = renderInfluenceBackground({
    ctx,
    context: args.context,
    viewport,
    indices: args.aggregate.indices,
    values: args.aggregate.values,
    backgroundMode: args.backgroundMode,
  });
  const { indices: topKIndices, values: topKValues } = topKInfluenceEntries(
    args.aggregate.indices,
    args.aggregate.values,
    args.k,
  );
  const topKMaxAbs = maxAbsValue(topKValues);
  const scaleMax = topKMaxAbs > 0 ? topKMaxAbs : backgroundStats.scaleMax;
  drawTopKInfluencePoints({
    ctx,
    context: args.context,
    viewport,
    indices: topKIndices,
    values: topKValues,
    scaleMax,
  });
  return { ...backgroundStats, scaleMax };
}

export function buildDelaunay(points: Float32Array, dim: number): Delaunay<number> {
  const count = Math.floor(points.length / dim);
  const indices = Array.from({ length: count }, (_, index) => index);
  return Delaunay.from(
    indices,
    (index) => points[index * dim],
    (index) => points[index * dim + 1] ?? 0,
  );
}

export function pointerInDomain(
  event: PointerEvent,
  canvas: HTMLCanvasElement,
  bounds: Bounds,
  viewport: PlotViewport,
): [number, number] | null {
  const [sx, sy] = pointer(event, canvas);
  if (sx < viewport.x || sx > viewport.right || sy < viewport.y || sy > viewport.bottom) {
    return null;
  }
  const nx = (sx - viewport.x) / Math.max(1, viewport.width);
  const ny = (viewport.bottom - sy) / Math.max(1, viewport.height);
  return [
    bounds.minX + nx * (bounds.maxX - bounds.minX),
    bounds.minY + ny * (bounds.maxY - bounds.minY),
  ];
}

export function rasterSampleAtCoord(
  raster: RasterData | null,
  decoded: Float32Array | null,
  bounds: Bounds,
  coord: [number, number] | null,
): { value: number; x: number; y: number } | null {
  if (!raster || !decoded || !coord) return null;
  const nx = Math.max(0, Math.min(1, (coord[0] - bounds.minX) / (bounds.maxX - bounds.minX || 1)));
  const ny = Math.max(0, Math.min(1, (bounds.maxY - coord[1]) / (bounds.maxY - bounds.minY || 1)));
  const col = clampIndex(Math.round(nx * raster.width - 0.5), raster.width);
  const row = clampIndex(Math.round(ny * raster.height - 0.5), raster.height);
  const index = row * raster.width + col;
  const dx = (bounds.maxX - bounds.minX) / raster.width;
  const dy = (bounds.maxY - bounds.minY) / raster.height;
  return {
    value: decoded[index],
    x: bounds.minX + (col + 0.5) * dx,
    y: bounds.maxY - (row + 0.5) * dy,
  };
}
