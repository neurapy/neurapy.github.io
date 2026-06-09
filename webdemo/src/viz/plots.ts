import {
  Delaunay,
  axisBottom,
  axisLeft,
  format,
  pointer,
  scaleSqrt,
  select,
} from "d3";
import type {
  Bounds,
  InfluenceAggregate,
  InfluenceMatrixManifest,
  InfluenceRow,
  PointArrays,
  PlotViewport,
  RasterData,
  RunManifest,
} from "../types";
import { clearCanvas, prepareCanvas } from "./canvas";
import { divergingColorScale, fieldColorScale, finiteExtent, sequentialColorScale } from "./color";
import {
  boundsFromAxisMap,
  clampIndex,
  fitScales,
  inferPointBounds,
  pointAt,
  plotViewport,
  projectPointToViewport,
} from "./geometry";

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

export function drawPointMarker(
  ctx: CanvasRenderingContext2D,
  sx: number,
  sy: number,
  radius: number,
): void {
  ctx.beginPath();
  ctx.arc(sx, sy, radius, 0, Math.PI * 2);
  ctx.fillStyle = "#f2b84b";
  ctx.fill();
  ctx.lineWidth = 2;
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
  const size = options.size ?? Math.max(1.5, Math.min(3.5, Math.sqrt((viewport.width * viewport.height) / Math.max(1, count)) * 0.2));
  ctx.save();
  ctx.globalAlpha = options.alpha ?? 0.18;
  ctx.fillStyle = options.color ?? "#364252";
  for (let index = 0; index < count; index += stride) {
    const [x, y] = pointAt(points, index, dim);
    const [sx, sy] = projectPointToViewport(x, y, bounds, viewport);
    ctx.fillRect(sx - size / 2, sy - size / 2, size, size);
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
  const rasterBounds = args.context.manifest.field_raster
    ? boundsFromAxisMap(
        args.context.manifest.field_raster.bounds,
        args.context.manifest.field_raster.axes,
      )
    : args.context.bounds;
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
    drawPointMarker(ctx, sx, sy, 7);
  }
  if (args.selectedRegion) {
    drawRegionOverlay(ctx, args.selectedRegion, rasterBounds, viewport, false);
  }
  if (args.draftRegion) {
    drawRegionOverlay(ctx, args.draftRegion, rasterBounds, viewport, true);
  }

  renderAxes(args.svg, rasterBounds, width, height, viewport);
  if (args.rasterResult?.contourPaths.length) {
    const root = select(args.svg).append("g").attr("class", "contours");
    const scaleX = viewport.width / Math.max(1, args.raster?.width ?? 1);
    const scaleY = viewport.height / Math.max(1, args.raster?.height ?? 1);
    root
      .selectAll("path")
      .data(args.rasterResult.contourPaths)
      .join("path")
      .attr("d", (pathValue) => pathValue)
      .attr("transform", `translate(${viewport.x},${viewport.y}) scale(${scaleX},${scaleY})`);
  }
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
  ctx.save();
  ctx.fillStyle = draft ? "rgba(242, 184, 75, 0.16)" : "rgba(12, 124, 120, 0.14)";
  ctx.strokeStyle = draft ? "#f2b84b" : "#0c7c78";
  ctx.lineWidth = draft ? 1.4 : 2;
  ctx.setLineDash(draft ? [6, 4] : []);
  ctx.fillRect(x, y, width, height);
  ctx.strokeRect(x, y, width, height);
  ctx.restore();
}

export function renderLocalInfluencePlot(args: {
  canvas: HTMLCanvasElement;
  svg: SVGSVGElement;
  context: PlotContext;
  matrix: InfluenceMatrixManifest;
  row: InfluenceRow | null;
  selectedCandidateIndex: number;
  selectedTrainIndex: number;
  k: number;
}): number {
  const { ctx, width, height } = prepareCanvas(args.canvas);
  clearCanvas(ctx, width, height);
  const viewport = plotViewport(args.context.bounds, width, height);
  renderAxes(args.svg, args.context.bounds, width, height, viewport);
  drawPointCloudLayer(ctx, args.context.points.train_points, args.context.trainDim, args.context.bounds, viewport, {
    color: "#526070",
    alpha: 0.18,
    maxPoints: 10000,
    size: 2,
  });

  if (!args.row) return 0;
  const rowSourcePoints =
    args.matrix.row_source === "train_points"
      ? args.context.points.train_points
      : args.context.points.candidate_points;
  const rowDim = args.matrix.row_source === "train_points" ? args.context.trainDim : args.context.candidateDim;
  const rowIndex =
    args.matrix.row_source === "train_points" ? args.selectedTrainIndex : args.selectedCandidateIndex;
  const rowPoint = pointAt(rowSourcePoints, clampIndex(rowIndex, args.matrix.row_count), rowDim);
  const [rowSx, rowSy] = projectPointToViewport(rowPoint[0], rowPoint[1], args.context.bounds, viewport);
  const values = args.row.values.subarray(0, Math.min(args.k, args.row.values.length));
  const indices = args.row.indices.subarray(0, values.length);
  const maxAbs = Math.max(0, ...Array.from(values, (value) => Math.abs(value)));
  const radius = scaleSqrt().domain([0, maxAbs || 1]).range([3, 13]);
  const color = divergingColorScale([-maxAbs || -1, maxAbs || 1]);

  ctx.save();
  ctx.globalAlpha = 0.34;
  ctx.strokeStyle = "#526070";
  ctx.lineWidth = 1;
  const linkCount = Math.min(50, indices.length);
  for (let index = linkCount - 1; index >= 0; index -= 1) {
    const trainPoint = pointAt(args.context.points.train_points, indices[index], args.context.trainDim);
    const [sx, sy] = projectPointToViewport(trainPoint[0], trainPoint[1], args.context.bounds, viewport);
    ctx.beginPath();
    ctx.moveTo(rowSx, rowSy);
    ctx.lineTo(sx, sy);
    ctx.stroke();
  }
  ctx.restore();

  for (let index = indices.length - 1; index >= 0; index -= 1) {
    const trainPoint = pointAt(args.context.points.train_points, indices[index], args.context.trainDim);
    const [sx, sy] = projectPointToViewport(trainPoint[0], trainPoint[1], args.context.bounds, viewport);
    ctx.beginPath();
    ctx.arc(sx, sy, radius(Math.abs(values[index])), 0, Math.PI * 2);
    ctx.fillStyle = color(values[index]);
    ctx.fill();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = "#182230";
    ctx.stroke();
  }
  drawPointMarker(ctx, rowSx, rowSy, 7);
  return maxAbs;
}

export function renderRegionalInfluencePlot(args: {
  canvas: HTMLCanvasElement;
  svg: SVGSVGElement;
  context: PlotContext;
  aggregate: InfluenceAggregate | null;
  k: number;
}): number {
  const { ctx, width, height } = prepareCanvas(args.canvas);
  clearCanvas(ctx, width, height);
  const viewport = plotViewport(args.context.bounds, width, height);
  renderAxes(args.svg, args.context.bounds, width, height, viewport);
  drawPointCloudLayer(ctx, args.context.points.train_points, args.context.trainDim, args.context.bounds, viewport, {
    color: "#526070",
    alpha: 0.18,
    maxPoints: 10000,
    size: 2,
  });

  if (!args.aggregate) return 0;
  const values = args.aggregate.values.subarray(0, Math.min(args.k, args.aggregate.values.length));
  const indices = args.aggregate.indices.subarray(0, values.length);
  const maxAbs = Math.max(0, ...Array.from(values, (value) => Math.abs(value)));
  const radius = scaleSqrt().domain([0, maxAbs || 1]).range([3, 15]);
  const color = divergingColorScale([-maxAbs || -1, maxAbs || 1]);

  for (let index = indices.length - 1; index >= 0; index -= 1) {
    const trainPoint = pointAt(args.context.points.train_points, indices[index], args.context.trainDim);
    const [sx, sy] = projectPointToViewport(trainPoint[0], trainPoint[1], args.context.bounds, viewport);
    ctx.beginPath();
    ctx.arc(sx, sy, radius(Math.abs(values[index])), 0, Math.PI * 2);
    ctx.fillStyle = color(values[index]);
    ctx.fill();
    ctx.lineWidth = 1.2;
    ctx.strokeStyle = "#182230";
    ctx.stroke();
  }
  return maxAbs;
}

export function renderGlobalPlot(args: {
  canvas: HTMLCanvasElement;
  svg: SVGSVGElement;
  context: PlotContext;
  values: Float32Array | null;
  diverging: boolean;
}): [number, number] {
  const { ctx, width, height } = prepareCanvas(args.canvas);
  clearCanvas(ctx, width, height);
  const viewport = plotViewport(args.context.bounds, width, height);
  renderAxes(args.svg, args.context.bounds, width, height, viewport);
  const count = Math.floor(args.context.points.train_points.length / args.context.trainDim);
  const values = args.values ?? new Float32Array(count);
  const domain = args.diverging ? finiteExtent(values) : finiteExtent(values);
  const color = args.diverging ? divergingColorScale(domain) : sequentialColorScale(domain);
  const size = Math.max(2.5, Math.min(7, Math.sqrt((viewport.width * viewport.height) / Math.max(1, count)) * 0.75));
  for (let index = 0; index < count; index += 1) {
    const [x, y] = pointAt(args.context.points.train_points, index, args.context.trainDim);
    const [sx, sy] = projectPointToViewport(x, y, args.context.bounds, viewport);
    const kind = args.context.points.train_kind[index] ?? 0;
    ctx.beginPath();
    if (kind) {
      ctx.rect(sx - size / 2, sy - size / 2, size, size);
    } else {
      ctx.arc(sx, sy, size / 2, 0, Math.PI * 2);
    }
    ctx.fillStyle = color(values[index] ?? 0);
    ctx.globalAlpha = 0.82;
    ctx.fill();
    if (kind) {
      ctx.globalAlpha = 0.9;
      ctx.lineWidth = 0.8;
      ctx.strokeStyle = "#182230";
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
  return domain;
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
