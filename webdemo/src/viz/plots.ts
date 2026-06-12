import {
  Delaunay,
  axisBottom,
  axisLeft,
  color as parseD3Color,
  interpolateTurbo,
  pointer,
  scaleLinear,
  select,
  type Selection,
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
  inferPointBounds,
  pointAt,
  plotViewport,
  projectPointToViewport,
  unprojectPointFromViewport,
  type PlotInsets,
} from "./geometry";
import {
  plotProjectionForManifest,
  projectBounds,
  projectPointToDisplay,
  type PlotProjection,
} from "./projection";
import { plotVisualScale, referenceAspectPlotArea, scaledPlotPx } from "./scale";

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

export interface InfluenceRenderResult extends InfluenceRenderStats {
  viewport: PlotViewport;
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

interface InfluenceFieldArgs {
  points: Float32Array;
  dim: number;
  bounds: Bounds;
  viewport: PlotViewport;
  indices: ArrayLike<number>;
  values: ArrayLike<number>;
  projection?: PlotProjection;
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
export const PLOT_DECORATION_INSETS: PlotInsets = {
  top: 18,
  right: 76,
  bottom: 52,
  left: 58,
};
const SELECTION_PULSE_DURATION_MS = 720;

type ColorbarKind = "sequential" | "diverging";

interface ColorbarSpec {
  id: string;
  kind: ColorbarKind;
  domain: [number, number];
}

function clampNumber(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function formatAxisNumber(value: number): string {
  if (!Number.isFinite(value)) return "";
  return value.toLocaleString("en-US", {
    maximumFractionDigits: Math.abs(value) < 10 ? 3 : 2,
    minimumFractionDigits: 0,
  });
}

function emptyInfluenceStats(backgroundMode: BackgroundMode): InfluenceRenderStats {
  return { maxAbs: 0, renderedCount: 0, backgroundMode, scaleMax: 1 };
}

function identityProjection(bounds: Bounds): PlotProjection {
  return {
    physicalBounds: bounds,
    displayBounds: bounds,
    labels: { x: "x", y: "y" },
    projectPoint: (x, y) => [x, y],
    unprojectPoint: (x, y) => [x, y],
    formatXTick: formatAxisNumber,
    formatYTick: formatAxisNumber,
  };
}

function projectionForInfluenceArgs(args: InfluenceFieldArgs): PlotProjection {
  return args.projection ?? identityProjection(args.bounds);
}

function projectPhysicalPointToViewport(
  x: number,
  y: number,
  projection: PlotProjection,
  viewport: PlotViewport,
): [number, number] {
  const [displayX, displayY] = projection.projectPoint(x, y);
  return projectPointToViewport(displayX, displayY, projection.displayBounds, viewport);
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
  const projection = projectionForInfluenceArgs(args);
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
    const [sx, sy] = projectPhysicalPointToViewport(x, y, projection, args.viewport);
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

function smoothBandwidth(samples: InfluenceSample[], width: number, height: number): number {
  const maxBandwidth = Math.max(3, Math.min(width, height) * 0.18);
  if (samples.length < 2) return clampNumber(Math.min(width, height) * 0.08, 3, maxBandwidth);
  const delaunay = Delaunay.from(
    samples,
    (sample) => sample.x,
    (sample) => sample.y,
  );
  const spacing = delaunayEdgeSpacing(samples, delaunay, width, height);
  return clampNumber(spacing * 1.35, 3, maxBandwidth);
}

function gaussianKernel(sigma: number): Float32Array {
  const radius = Math.max(1, Math.ceil(sigma * 3));
  const kernel = new Float32Array(radius * 2 + 1);
  const invTwoSigmaSq = 1 / (2 * sigma * sigma);
  let sum = 0;
  for (let index = 0; index < kernel.length; index += 1) {
    const distance = index - radius;
    const weight = Math.exp(-(distance * distance) * invTwoSigmaSq);
    kernel[index] = weight;
    sum += weight;
  }
  if (sum > 0) {
    for (let index = 0; index < kernel.length; index += 1) kernel[index] /= sum;
  }
  return kernel;
}

function addBilinearSample(args: {
  values: Float32Array;
  support: Float32Array;
  width: number;
  height: number;
  x: number;
  y: number;
  value: number;
}): void {
  const x0 = clampNumber(Math.floor(args.x), 0, args.width - 1);
  const y0 = clampNumber(Math.floor(args.y), 0, args.height - 1);
  const x1 = Math.min(args.width - 1, x0 + 1);
  const y1 = Math.min(args.height - 1, y0 + 1);
  const tx = x1 === x0 ? 0 : args.x - x0;
  const ty = y1 === y0 ? 0 : args.y - y0;
  const weights = [
    [x0, y0, (1 - tx) * (1 - ty)],
    [x1, y0, tx * (1 - ty)],
    [x0, y1, (1 - tx) * ty],
    [x1, y1, tx * ty],
  ] as const;

  for (const [x, y, weight] of weights) {
    if (weight <= 0) continue;
    const offset = y * args.width + x;
    args.values[offset] += args.value * weight;
    args.support[offset] += weight;
  }
}

function convolveSeparable(
  input: Float32Array,
  width: number,
  height: number,
  kernel: Float32Array,
): Float32Array {
  const radius = Math.floor(kernel.length / 2);
  const temp = new Float32Array(input.length);
  const output = new Float32Array(input.length);

  for (let row = 0; row < height; row += 1) {
    const rowOffset = row * width;
    for (let col = 0; col < width; col += 1) {
      let sum = 0;
      for (let k = -radius; k <= radius; k += 1) {
        const sourceCol = col + k;
        if (sourceCol < 0 || sourceCol >= width) continue;
        sum += input[rowOffset + sourceCol] * kernel[k + radius];
      }
      temp[rowOffset + col] = sum;
    }
  }

  for (let row = 0; row < height; row += 1) {
    for (let col = 0; col < width; col += 1) {
      let sum = 0;
      for (let k = -radius; k <= radius; k += 1) {
        const sourceRow = row + k;
        if (sourceRow < 0 || sourceRow >= height) continue;
        sum += temp[sourceRow * width + col] * kernel[k + radius];
      }
      output[row * width + col] = sum;
    }
  }

  return output;
}

export function computeSmoothInfluenceField(args: InfluenceFieldArgs): InfluenceField {
  const sampleSet = collectInfluenceSamples(args);
  const { width, height, samples, renderedCount } = sampleSet;
  const numerator = new Float32Array(width * height);
  const support = new Float32Array(width * height);
  if (!samples.length) {
    return finalizeInfluenceField({ width, height, values: numerator, support, renderedCount });
  }

  for (const sample of samples) {
    addBilinearSample({
      values: numerator,
      support,
      width,
      height,
      x: sample.x,
      y: sample.y,
      value: sample.value,
    });
  }

  const kernel = gaussianKernel(smoothBandwidth(samples, width, height));
  const smoothedNumerator = convolveSeparable(numerator, width, height, kernel);
  const smoothedSupport = convolveSeparable(support, width, height, kernel);
  const minSupport = 1e-4;
  for (let index = 0; index < smoothedNumerator.length; index += 1) {
    if (smoothedSupport[index] <= minSupport) {
      smoothedNumerator[index] = 0;
      smoothedSupport[index] = 0;
      continue;
    }
    smoothedNumerator[index] /= smoothedSupport[index];
  }

  return finalizeInfluenceField({
    width,
    height,
    values: smoothedNumerator,
    support: smoothedSupport,
    renderedCount,
  });
}

export function resizeSvg(svg: SVGSVGElement, width: number, height: number): void {
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", `${width}`);
  svg.setAttribute("height", `${height}`);
}

export function selectionPulseProgress(startedAt: number, now = performance.now()): number {
  if (!startedAt) return 0;
  const elapsed = now - startedAt;
  if (elapsed <= 0) return 0;
  if (elapsed >= SELECTION_PULSE_DURATION_MS) return 0;
  return elapsed / SELECTION_PULSE_DURATION_MS;
}

function finiteDomain(domain: [number, number]): [number, number] {
  const min = Number.isFinite(domain[0]) ? domain[0] : 0;
  const max = Number.isFinite(domain[1]) ? domain[1] : min + 1;
  return min === max ? [min, min + 1] : [min, max];
}

function colorbarTicks(spec: ColorbarSpec): number[] {
  const [min, max] = finiteDomain(spec.domain);
  if (spec.kind === "diverging") {
    const maxAbs = Math.max(Math.abs(min), Math.abs(max)) || 1;
    return [-maxAbs, 0, maxAbs];
  }
  return [min, min + (max - min) / 2, max];
}

function colorbarColor(spec: ColorbarSpec, value: number): string {
  const [min, max] = finiteDomain(spec.domain);
  if (spec.kind === "diverging") {
    return divergingColorScale([min, max])(value);
  }
  const span = max - min || 1;
  return interpolateTurbo(clampNumber((value - min) / span, 0, 1));
}

function renderColorbar(
  root: Selection<SVGSVGElement, unknown, null, undefined>,
  viewport: PlotViewport,
  width: number,
  spec: ColorbarSpec,
): void {
  const [min, max] = finiteDomain(spec.domain);
  const visualScale = plotVisualScale(viewport);
  const barWidth = Math.max(8, Math.min(12, 9 * visualScale));
  const gutterLeft = viewport.right;
  const gutterWidth = Math.max(0, width - gutterLeft);
  if (gutterWidth < 30) return;

  const barHeight = Math.max(46, Math.min(150, viewport.height * 0.56));
  const barX = Math.min(width - 34, gutterLeft + Math.max(10, (gutterWidth - 42) / 2));
  const barY = viewport.y + (viewport.height - barHeight) / 2;
  const gradientId = `${spec.id}-gradient`;
  const defs = root.append("defs");
  const gradient = defs
    .append("linearGradient")
    .attr("id", gradientId)
    .attr("x1", "0%")
    .attr("x2", "0%")
    .attr("y1", "100%")
    .attr("y2", "0%");
  const stops = Array.from({ length: 9 }, (_unused, index) => index / 8);
  gradient
    .selectAll("stop")
    .data(stops)
    .join("stop")
    .attr("offset", (value) => `${value * 100}%`)
    .attr("stop-color", (value) => colorbarColor(spec, min + value * (max - min)));

  const group = root.append("g").attr("class", "colorbar");
  group
    .append("rect")
    .attr("class", "colorbar-track")
    .attr("x", barX)
    .attr("y", barY)
    .attr("width", barWidth)
    .attr("height", barHeight)
    .attr("rx", 2)
    .attr("fill", `url(#${gradientId})`);
  group
    .append("rect")
    .attr("class", "colorbar-frame")
    .attr("x", barX)
    .attr("y", barY)
    .attr("width", barWidth)
    .attr("height", barHeight)
    .attr("rx", 2);

  const tickScale = scaleLinear().domain([min, max]).range([barY + barHeight, barY]);
  const ticks = colorbarTicks(spec);
  const tickGroup = group.append("g").attr("class", "colorbar-ticks");
  tickGroup
    .selectAll("line")
    .data(ticks)
    .join("line")
    .attr("x1", barX + barWidth)
    .attr("x2", barX + barWidth + 4)
    .attr("y1", (value) => tickScale(value))
    .attr("y2", (value) => tickScale(value));
  tickGroup
    .selectAll("text")
    .data(ticks)
    .join("text")
    .attr("x", barX + barWidth + 7)
    .attr("y", (value) => tickScale(value))
    .attr("dy", "0.32em")
    .text((value) => formatAxisNumber(value));
}

export function renderAxes(
  svg: SVGSVGElement,
  bounds: Bounds,
  width: number,
  height: number,
  viewport: PlotViewport,
  projection: PlotProjection,
  colorbar?: ColorbarSpec,
): void {
  resizeSvg(svg, width, height);
  const visualScale = plotVisualScale(viewport);
  svg.style.setProperty("--plot-visual-scale", String(visualScale));
  svg.style.setProperty("--plot-axis-stroke-width", `${visualScale}px`);
  svg.style.setProperty("--plot-contour-stroke-width", `${0.7 * visualScale}px`);
  const x = scaleLinear()
    .domain([projection.displayBounds.minX, projection.displayBounds.maxX])
    .range([viewport.x, viewport.right]);
  const y = scaleLinear()
    .domain([projection.displayBounds.minY, projection.displayBounds.maxY])
    .range([viewport.bottom, viewport.y]);
  const root = select(svg);
  root.selectAll("*").remove();
  root
    .append("rect")
    .attr("class", "axis-frame")
    .attr("x", viewport.x)
    .attr("y", viewport.y)
    .attr("width", viewport.width)
    .attr("height", viewport.height);
  const xAxis = axisBottom(x)
    .ticks(Math.max(3, Math.floor(viewport.width / 150)))
    .tickFormat((value) => projection.formatXTick(Number(value)));
  if (projection.xTickValues?.length) xAxis.tickValues(projection.xTickValues);
  root
    .append("g")
    .attr("class", "axis axis-x")
    .attr("transform", `translate(0,${viewport.bottom})`)
    .call(xAxis);
  root
    .append("g")
    .attr("class", "axis axis-y")
    .attr("transform", `translate(${viewport.x},0)`)
    .call(
      axisLeft(y)
        .ticks(Math.max(3, Math.floor(viewport.height / 130)))
        .tickFormat((value) => projection.formatYTick(Number(value))),
    );
  root
    .append("text")
    .attr("class", "axis-label axis-label-x")
    .attr("x", viewport.x + viewport.width / 2)
    .attr("y", Math.min(height - 9, viewport.bottom + 36))
    .attr("text-anchor", "middle")
    .text(projection.labels.x);
  root
    .append("text")
    .attr("class", "axis-label axis-label-y")
    .attr("x", Math.max(12, viewport.x - 42))
    .attr("y", viewport.y + viewport.height / 2)
    .attr("text-anchor", "middle")
    .attr("transform", `rotate(-90 ${Math.max(12, viewport.x - 42)} ${viewport.y + viewport.height / 2})`)
    .text(projection.labels.y);
  if (colorbar) renderColorbar(root, viewport, width, colorbar);
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
  projection: PlotProjection;
}): void {
  if (!args.raster || !args.rasterResult?.contourPaths.length) return;
  const targetBounds = projectBounds(args.targetBounds, args.projection);
  const rasterBounds = projectBounds(args.rasterBounds, args.projection);
  const targetSpanX = boundsSpan(targetBounds.minX, targetBounds.maxX);
  const targetSpanY = boundsSpan(targetBounds.minY, targetBounds.maxY);
  const rasterSpanX = boundsSpan(rasterBounds.minX, rasterBounds.maxX);
  const rasterSpanY = boundsSpan(rasterBounds.minY, rasterBounds.maxY);
  const scaleX =
    (rasterSpanX / targetSpanX) * (args.viewport.width / Math.max(1, args.raster.width));
  const scaleY =
    (rasterSpanY / targetSpanY) * (args.viewport.height / Math.max(1, args.raster.height));
  const offsetX =
    args.viewport.x +
    ((rasterBounds.minX - targetBounds.minX) / targetSpanX) * args.viewport.width;
  const offsetY =
    args.viewport.y +
    ((targetBounds.maxY - rasterBounds.maxY) / targetSpanY) * args.viewport.height;
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

function drawPlotStage(
  ctx: CanvasRenderingContext2D,
  viewport: PlotViewport,
): void {
  const visualScale = plotVisualScale(viewport);
  ctx.save();
  ctx.shadowColor = "rgba(24, 34, 48, 0.16)";
  ctx.shadowBlur = 16 * visualScale;
  ctx.shadowOffsetY = 5 * visualScale;
  ctx.fillStyle = "#fbfdff";
  ctx.fillRect(viewport.x, viewport.y, viewport.width, viewport.height);
  ctx.restore();

  ctx.save();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.92)";
  ctx.lineWidth = Math.max(1, 1.2 * visualScale);
  ctx.strokeRect(viewport.x + 0.5, viewport.y + 0.5, viewport.width - 1, viewport.height - 1);
  ctx.restore();
}

export function drawPointMarker(
  ctx: CanvasRenderingContext2D,
  sx: number,
  sy: number,
  radius: number,
  visualScale = 1,
  pulse = 0,
): void {
  const scaledRadius = Math.max(0, radius * visualScale);
  const safePulse = clampNumber(pulse, 0, 1);
  if (safePulse > 0) {
    const haloRadius = scaledRadius + (5 + safePulse * 11) * visualScale;
    const haloAlpha = Math.max(0, 0.24 * (1 - safePulse));
    ctx.save();
    ctx.beginPath();
    ctx.arc(sx, sy, haloRadius, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(233, 168, 47, ${haloAlpha})`;
    ctx.fill();
    ctx.restore();
  }
  ctx.save();
  ctx.shadowColor = "rgba(24, 34, 48, 0.22)";
  ctx.shadowBlur = 5 * visualScale;
  ctx.shadowOffsetY = 1.5 * visualScale;
  ctx.beginPath();
  ctx.arc(sx, sy, scaledRadius, 0, Math.PI * 2);
  ctx.fillStyle = "#e9a82f";
  ctx.fill();
  ctx.shadowColor = "transparent";
  ctx.lineWidth = 2 * visualScale;
  ctx.strokeStyle = "#182230";
  ctx.stroke();
  ctx.restore();
}

export function drawPointCloudLayer(
  ctx: CanvasRenderingContext2D,
  points: Float32Array,
  dim: number,
  bounds: Bounds,
  viewport: PlotViewport,
  projection: PlotProjection,
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
  const pointSizingArea = referenceAspectPlotArea(viewport);
  const size =
    options.size === undefined
      ? Math.max(
          1.5 * visualScale,
          Math.min(
            3.5 * visualScale,
            Math.sqrt(pointSizingArea / Math.max(1, count)) * 0.2,
          ),
        )
      : options.size * visualScale;
  ctx.save();
  ctx.globalAlpha = options.alpha ?? 0.18;
  ctx.fillStyle = options.color ?? "#364252";
  for (let index = 0; index < count; index += stride) {
    const [x, y] = pointAt(points, index, dim);
    const [sx, sy] = projectPhysicalPointToViewport(x, y, projection, viewport);
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
  selectionPulse?: number;
}): PlotViewport {
  const { ctx, width, height } = prepareCanvas(args.canvas);
  clearCanvas(ctx, width, height);
  const rasterBounds = rasterPlotBounds(args.context);
  const projection = plotProjectionForManifest(args.context.manifest, rasterBounds);
  const viewport = plotViewport(projection.displayBounds, width, height, PLOT_DECORATION_INSETS);
  drawPlotStage(ctx, viewport);

  if (args.raster && args.rasterResult) {
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(args.rasterResult.image, viewport.x, viewport.y, viewport.width, viewport.height);
  }
  if (args.showTrainPoints) {
    drawPointCloudLayer(
      ctx,
      args.context.points.train_points,
      args.context.trainDim,
      rasterBounds,
      viewport,
      projection,
      {
        color: "#11263a",
        alpha: 0.12,
        maxPoints: 6000,
      },
    );
  }
  if (args.showCandidatePoints) {
    drawPointCloudLayer(
      ctx,
      args.context.points.candidate_points,
      args.context.candidateDim,
      rasterBounds,
      viewport,
      projection,
      {
        color: "#087f7c",
        alpha: 0.16,
        maxPoints: 6000,
      },
    );
  }
  if (args.selectedCoord) {
    const [sx, sy] = projectPhysicalPointToViewport(
      args.selectedCoord[0],
      args.selectedCoord[1],
      projection,
      viewport,
    );
    drawPointMarker(ctx, sx, sy, 7, plotVisualScale(viewport), args.selectionPulse ?? 0);
  }
  if (args.selectedRegion) {
    drawRegionOverlay(ctx, args.selectedRegion, rasterBounds, viewport, projection, false);
  }
  if (args.draftRegion) {
    drawRegionOverlay(ctx, args.draftRegion, rasterBounds, viewport, projection, true);
  }

  renderAxes(
    args.svg,
    rasterBounds,
    width,
    height,
    viewport,
    projection,
    args.raster
      ? { id: "model-colorbar", kind: "sequential", domain: args.raster.displayDomain }
      : undefined,
  );
  renderContourOverlay({
    svg: args.svg,
    raster: args.raster,
    rasterResult: args.rasterResult,
    rasterBounds,
    targetBounds: rasterBounds,
    viewport,
    projection,
  });
  return viewport;
}

function drawRegionOverlay(
  ctx: CanvasRenderingContext2D,
  region: Bounds,
  bounds: Bounds,
  viewport: PlotViewport,
  projection: PlotProjection,
  draft: boolean,
): void {
  const [x0, y0] = projectPhysicalPointToViewport(region.minX, region.maxY, projection, viewport);
  const [x1, y1] = projectPhysicalPointToViewport(region.maxX, region.minY, projection, viewport);
  const x = Math.min(x0, x1);
  const y = Math.min(y0, y1);
  const width = Math.abs(x1 - x0);
  const height = Math.abs(y1 - y0);
  const visualScale = plotVisualScale(viewport);
  ctx.save();
  ctx.fillStyle = draft ? "rgba(233, 168, 47, 0.18)" : "rgba(8, 127, 124, 0.16)";
  ctx.strokeStyle = draft ? "#e9a82f" : "#087f7c";
  ctx.lineWidth = (draft ? 1.4 : 2) * visualScale;
  ctx.setLineDash(draft ? [6 * visualScale, 4 * visualScale] : []);
  ctx.shadowColor = draft ? "rgba(233, 168, 47, 0.24)" : "rgba(8, 127, 124, 0.22)";
  ctx.shadowBlur = 10 * visualScale;
  ctx.fillRect(x, y, width, height);
  ctx.shadowColor = "transparent";
  ctx.strokeRect(x, y, width, height);
  ctx.setLineDash([]);
  ctx.strokeStyle = "rgba(255, 255, 255, 0.72)";
  ctx.lineWidth = 1 * visualScale;
  ctx.strokeRect(x + visualScale, y + visualScale, Math.max(0, width - 2 * visualScale), Math.max(0, height - 2 * visualScale));
  ctx.restore();
}

function drawInfluencePointLayer(args: {
  ctx: CanvasRenderingContext2D;
  context: PlotContext;
  viewport: PlotViewport;
  projection: PlotProjection;
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
    const [sx, sy] = projectPhysicalPointToViewport(
      trainPoint[0],
      trainPoint[1],
      args.projection,
      args.viewport,
    );
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
  projection: PlotProjection;
  indices: ArrayLike<number>;
  values: ArrayLike<number>;
  scaleMax: number;
  rowSx: number;
  rowSy: number;
}): void {
  const count = Math.min(args.indices.length, args.values.length);
  const trainCount = Math.floor(args.context.points.train_points.length / args.context.trainDim);
  args.ctx.save();
  for (let index = count - 1; index >= 0; index -= 1) {
    const trainIndex = Math.trunc(args.indices[index]);
    const value = args.values[index];
    if (!Number.isFinite(value) || trainIndex < 0 || trainIndex >= trainCount) continue;
    const trainPoint = pointAt(args.context.points.train_points, trainIndex, args.context.trainDim);
    const [sx, sy] = projectPhysicalPointToViewport(
      trainPoint[0],
      trainPoint[1],
      args.projection,
      args.viewport,
    );
    const strength = influenceStrength(value, args.scaleMax);
    args.ctx.beginPath();
    args.ctx.moveTo(args.rowSx, args.rowSy);
    args.ctx.lineTo(sx, sy);
    args.ctx.strokeStyle = `rgba(82, 96, 112, ${0.14 + strength * 0.2})`;
    args.ctx.lineWidth = scaledPlotPx(1 + strength * 0.8, args.viewport);
    args.ctx.stroke();
  }
  args.ctx.restore();
}

function drawTopKInfluencePoints(args: {
  ctx: CanvasRenderingContext2D;
  context: PlotContext;
  viewport: PlotViewport;
  projection: PlotProjection;
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
    const [sx, sy] = projectPhysicalPointToViewport(
      trainPoint[0],
      trainPoint[1],
      args.projection,
      args.viewport,
    );
    const strength = influenceStrength(value, args.scaleMax);
    const size = scaledPlotPx(
      MIN_TOP_K_POINT_SIZE + strength * (MAX_TOP_K_POINT_SIZE - MIN_TOP_K_POINT_SIZE),
      args.viewport,
    );
    args.ctx.beginPath();
    args.ctx.arc(sx, sy, size * 0.64, 0, Math.PI * 2);
    args.ctx.fillStyle = influenceRgba(value, args.scaleMax, 0.18 + strength * 0.18);
    args.ctx.fill();
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
  projection: PlotProjection;
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
      projection: args.projection,
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
    projection: args.projection,
    indices: args.indices,
    values: args.values,
  };
  const field =
    args.backgroundMode === "cell"
      ? computeCellsInfluenceLayer(fieldArgs)
      : computeSmoothInfluenceField(fieldArgs);
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
  selectionPulse?: number;
}): InfluenceRenderResult {
  const { ctx, width, height } = prepareCanvas(args.canvas);
  clearCanvas(ctx, width, height);
  const projection = plotProjectionForManifest(args.context.manifest, args.context.bounds);
  const viewport = plotViewport(projection.displayBounds, width, height, PLOT_DECORATION_INSETS);
  drawPlotStage(ctx, viewport);
  renderAxes(args.svg, args.context.bounds, width, height, viewport, projection);
  renderContourOverlay({
    svg: args.svg,
    raster: args.raster,
    rasterResult: args.rasterResult,
    rasterBounds: rasterPlotBounds(args.context),
    targetBounds: args.context.bounds,
    viewport,
    projection,
  });
  drawPointCloudLayer(
    ctx,
    args.context.points.train_points,
    args.context.trainDim,
    args.context.bounds,
    viewport,
    projection,
    {
      color: "#526070",
      alpha: 0.18,
      maxPoints: 10000,
      size: 2,
    },
  );

  const rowSourcePoints =
    args.matrix.row_source === "train_points"
      ? args.context.points.train_points
      : args.context.points.candidate_points;
  const rowDim = args.matrix.row_source === "train_points" ? args.context.trainDim : args.context.candidateDim;
  const rowIndex =
    args.matrix.row_source === "train_points" ? args.selectedTrainIndex : args.selectedCandidateIndex;
  const rowPoint = pointAt(rowSourcePoints, clampIndex(rowIndex, args.matrix.row_count), rowDim);
  const [rowSx, rowSy] = projectPhysicalPointToViewport(
    rowPoint[0],
    rowPoint[1],
    projection,
    viewport,
  );
  if (!args.row) {
    drawPointMarker(ctx, rowSx, rowSy, 7, plotVisualScale(viewport), args.selectionPulse ?? 0);
    return { ...emptyInfluenceStats(args.backgroundMode), viewport };
  }
  const backgroundEntries = influenceEntriesForBackground(args.row.indices, args.row.values, args.sign);
  const backgroundStats = renderInfluenceBackground({
    ctx,
    context: args.context,
    viewport,
    projection,
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
    projection,
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
    projection,
    indices: topKIndices,
    values: topKValues,
    scaleMax,
  });

  drawPointMarker(ctx, rowSx, rowSy, 7, plotVisualScale(viewport), args.selectionPulse ?? 0);
  renderColorbar(select(args.svg), viewport, width, {
    id: "train-colorbar",
    kind: "diverging",
    domain: [-scaleMax, scaleMax],
  });
  return { ...backgroundStats, scaleMax, viewport };
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
}): InfluenceRenderResult {
  const { ctx, width, height } = prepareCanvas(args.canvas);
  clearCanvas(ctx, width, height);
  const projection = plotProjectionForManifest(args.context.manifest, args.context.bounds);
  const viewport = plotViewport(projection.displayBounds, width, height, PLOT_DECORATION_INSETS);
  drawPlotStage(ctx, viewport);
  renderAxes(args.svg, args.context.bounds, width, height, viewport, projection);
  renderContourOverlay({
    svg: args.svg,
    raster: args.raster,
    rasterResult: args.rasterResult,
    rasterBounds: rasterPlotBounds(args.context),
    targetBounds: args.context.bounds,
    viewport,
    projection,
  });
  drawPointCloudLayer(
    ctx,
    args.context.points.train_points,
    args.context.trainDim,
    args.context.bounds,
    viewport,
    projection,
    {
      color: "#526070",
      alpha: 0.18,
      maxPoints: 10000,
      size: 2,
    },
  );

  if (!args.aggregate) return { ...emptyInfluenceStats(args.backgroundMode), viewport };
  const backgroundStats = renderInfluenceBackground({
    ctx,
    context: args.context,
    viewport,
    projection,
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
    projection,
    indices: topKIndices,
    values: topKValues,
    scaleMax,
  });
  renderColorbar(select(args.svg), viewport, width, {
    id: "train-colorbar",
    kind: "diverging",
    domain: [-scaleMax, scaleMax],
  });
  return { ...backgroundStats, scaleMax, viewport };
}

export function buildDelaunay(
  points: Float32Array,
  dim: number,
  projection?: PlotProjection,
): Delaunay<number> {
  const count = Math.floor(points.length / dim);
  const indices = Array.from({ length: count }, (_, index) => index);
  return Delaunay.from(
    indices,
    (index) => {
      const x = points[index * dim];
      const y = points[index * dim + 1] ?? 0;
      return projection ? projection.projectPoint(x, y)[0] : x;
    },
    (index) => {
      const x = points[index * dim];
      const y = points[index * dim + 1] ?? 0;
      return projection ? projection.projectPoint(x, y)[1] : y;
    },
  );
}

export function pointerInDomain(
  event: PointerEvent,
  canvas: HTMLCanvasElement,
  projection: PlotProjection,
  viewport: PlotViewport,
): [number, number] | null {
  const [sx, sy] = pointer(event, canvas);
  if (sx < viewport.x || sx > viewport.right || sy < viewport.y || sy > viewport.bottom) {
    return null;
  }
  const display = unprojectPointFromViewport(sx, sy, projection.displayBounds, viewport);
  return projection.unprojectPoint(display[0], display[1]);
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
