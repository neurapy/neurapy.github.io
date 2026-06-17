import { area, axisBottom, axisLeft, line, scaleLinear, select } from "d3";
import type {
  LossDecompositionData,
  LossDecompositionOutput,
  LossDecompositionTermSeries,
  ModelQuality,
} from "../types";
import { formatDisplayLabel, formatNumber } from "../ui/dom";

export interface LossDecompositionRenderArgs {
  svg: SVGSVGElement;
  datasets: Partial<Record<ModelQuality, LossDecompositionData>>;
  outputId: string | null;
}

interface StackedPoint {
  x: number;
  y0: number;
  y1: number;
}

interface PanelDatum {
  quality: ModelQuality;
  label: string;
  data: LossDecompositionData | null;
  output: LossDecompositionOutput | null;
}

const QUALITY_LABELS: Record<ModelQuality, string> = {
  good: "Good",
  bad: "Bad",
};
const QUALITY_ORDER: ModelQuality[] = ["good", "bad"];
const TERM_COLORS = [
  "#1f82c0",
  "#66bfac",
  "#7353ba",
  "#e39d25",
  "#d14f3f",
  "#8bbf45",
  "#7c6f64",
  "#c05a9d",
  "#4f8f8d",
  "#9d7b33",
];

export function renderLossDecompositionPlot(args: LossDecompositionRenderArgs): void {
  const svg = args.svg;
  const width = Math.max(0, Math.floor(svg.clientWidth || svg.getBoundingClientRect().width));
  const height = Math.max(0, Math.floor(svg.clientHeight || svg.getBoundingClientRect().height));
  const root = select(svg);
  root.selectAll("*").remove();
  svg.setAttribute("viewBox", `0 0 ${Math.max(1, width)} ${Math.max(1, height)}`);

  if (width < 260 || height < 150) return;

  const panels = QUALITY_ORDER.map((quality) => {
    const data = args.datasets[quality] ?? null;
    return {
      quality,
      label: QUALITY_LABELS[quality],
      data,
      output: selectedOutput(data, args.outputId),
    };
  });
  if (!panels.some((panel) => panel.output)) {
    drawEmpty(root, width, height, "Loss decomposition unavailable");
    return;
  }

  const terms = sortedTerms(panels);
  if (!terms.length) {
    drawEmpty(root, width, height, "Loss decomposition unavailable");
    return;
  }

  const firstOutputPanel = panels.find((panel) => panel.output);
  const axisLabel = firstOutputPanel?.data?.axis.label ?? "Bin center";
  const xDomain = sharedXDomain(panels);
  const cancellationMax = sharedCancellationMax(panels);
  const colorForTerm = new Map(terms.map((term, index) => [term, TERM_COLORS[index % TERM_COLORS.length]]));
  const layout = plotLayout(width, height, terms.length);
  drawLegend(root, terms, colorForTerm, layout.legend);

  const group = root.append("g").attr("class", "loss-decomposition-panels");
  panels.forEach((panel, index) => {
    const frame = layout.frames[index];
    renderPanel(group, {
      panel,
      terms,
      colorForTerm,
      xDomain,
      cancellationMax,
      axisLabel,
      x: frame.x,
      y: frame.y,
      width: frame.width,
      height: frame.height,
      showYAxis: true,
    });
  });
}

function selectedOutput(
  data: LossDecompositionData | null,
  outputId: string | null,
): LossDecompositionOutput | null {
  if (!data?.outputs.length) return null;
  return (
    (outputId ? data.outputs.find((output) => output.id === outputId) : null) ??
    data.outputs[0] ??
    null
  );
}

function sortedTerms(panels: PanelDatum[]): string[] {
  const terms = new Set<string>();
  for (const panel of panels) {
    for (const term of panel.output?.terms ?? []) {
      terms.add(term.id);
    }
  }
  return Array.from(terms).sort(termSortKey);
}

function termSortKey(a: string, b: string): number {
  const pa = termParts(a);
  const pb = termParts(b);
  return pa.group - pb.group || pa.index - pb.index || a.localeCompare(b);
}

function termParts(term: string): { group: number; index: number } {
  const [prefix, suffix] = term.split("_");
  const group = prefix === "pde" ? 0 : prefix === "bc" ? 1 : 2;
  const index = suffix && /^\d+$/.test(suffix) ? Number(suffix) : 999;
  return { group, index };
}

function sharedXDomain(panels: PanelDatum[]): [number, number] {
  const values = panels.flatMap((panel) => panel.output?.bin_centers ?? []);
  const finite = values.filter(Number.isFinite);
  if (!finite.length) return [0, 1];
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  return min < max ? [min, max] : [min - 0.5, max + 0.5];
}

function sharedCancellationMax(panels: PanelDatum[]): number {
  const values = panels.flatMap((panel) =>
    (panel.output?.bin_centers ?? []).map((_, index) =>
      cancellationAt(panel.output, index),
    ),
  );
  const max = Math.max(0, ...values.filter(Number.isFinite));
  if (max <= 0.08) return 0.08;
  return Math.min(1, Math.ceil(max * 10) / 10);
}

function plotLayout(width: number, height: number, termCount: number) {
  const margin = { top: 10, right: 12, bottom: 10, left: 12 };
  const legendItemWidth = 58;
  const legendColumns = Math.max(1, Math.floor((width - margin.left - margin.right) / legendItemWidth));
  const legendRows = Math.max(1, Math.ceil(termCount / legendColumns));
  const legendHeight = Math.min(42, 8 + legendRows * 15);
  const vertical = width < 560;
  const gap = vertical ? 10 : 14;
  const plotTop = margin.top + legendHeight;
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = Math.max(80, height - plotTop - margin.bottom);
  const frameWidth = vertical ? plotWidth : (plotWidth - gap) / 2;
  const frameHeight = vertical ? (plotHeight - gap) / 2 : plotHeight;
  return {
    legend: {
      x: margin.left,
      y: margin.top,
      width: plotWidth,
      itemWidth: legendItemWidth,
      columns: legendColumns,
    },
    frames: QUALITY_ORDER.map((_, index) => ({
      x: margin.left + (vertical ? 0 : index * (frameWidth + gap)),
      y: plotTop + (vertical ? index * (frameHeight + gap) : 0),
      width: frameWidth,
      height: frameHeight,
    })),
  };
}

function drawLegend(
  root: ReturnType<typeof select<SVGSVGElement, unknown>>,
  terms: string[],
  colorForTerm: Map<string, string>,
  legend: { x: number; y: number; itemWidth: number; columns: number },
): void {
  const group = root.append("g").attr("class", "loss-decomposition-legend");
  terms.forEach((term, index) => {
    const col = index % legend.columns;
    const row = Math.floor(index / legend.columns);
    const item = group
      .append("g")
      .attr("transform", `translate(${legend.x + col * legend.itemWidth},${legend.y + row * 15})`);
    item
      .append("rect")
      .attr("width", 10)
      .attr("height", 10)
      .attr("y", 1)
      .attr("fill", colorForTerm.get(term) ?? "#999999");
    item
      .append("text")
      .attr("x", 14)
      .attr("y", 10)
      .text(shortTermLabel(term));
  });
}

function renderPanel(
  parent: ReturnType<typeof select<SVGGElement, unknown>>,
  args: {
    panel: PanelDatum;
    terms: string[];
    colorForTerm: Map<string, string>;
    xDomain: [number, number];
    cancellationMax: number;
    axisLabel: string;
    x: number;
    y: number;
    width: number;
    height: number;
    showYAxis: boolean;
  },
): void {
  const group = parent
    .append("g")
    .attr("class", `loss-decomposition-panel loss-decomposition-${args.panel.quality}`)
    .attr("transform", `translate(${args.x},${args.y})`);
  group.append("rect").attr("class", "loss-decomposition-frame").attr("width", args.width).attr("height", args.height);
  group
    .append("text")
    .attr("class", "loss-decomposition-panel-title")
    .attr("x", 6)
    .attr("y", 14)
    .text(panelTitle(args.panel));

  if (!args.panel.output) {
    group
      .append("text")
      .attr("class", "loss-decomposition-empty")
      .attr("x", args.width / 2)
      .attr("y", args.height / 2)
      .attr("text-anchor", "middle")
      .text(`${args.panel.label} unavailable`);
    return;
  }

  const left = args.showYAxis ? 34 : 12;
  const right = 8;
  const top = 22;
  const bottom = 28;
  const gap = 14;
  const innerWidth = Math.max(1, args.width - left - right);
  const innerHeight = Math.max(1, args.height - top - bottom);
  const fractionHeight = Math.max(44, Math.round(innerHeight * 0.62));
  const cancellationHeight = Math.max(34, innerHeight - fractionHeight - gap);
  const x = scaleLinear().domain(args.xDomain).range([0, innerWidth]);
  const yFraction = scaleLinear().domain([0, 1]).range([fractionHeight, 0]);
  const yCancellation = scaleLinear()
    .domain([0, args.cancellationMax || 0.08])
    .range([cancellationHeight, 0]);
  const plot = group.append("g").attr("transform", `translate(${left},${top})`);

  const stacks = stackedTermData(args.panel.output, args.terms);
  const areaPath = area<StackedPoint>()
    .defined((point) => Number.isFinite(point.x) && Number.isFinite(point.y0) && Number.isFinite(point.y1))
    .x((point) => x(point.x))
    .y0((point) => yFraction(point.y0))
    .y1((point) => yFraction(point.y1));
  for (const series of stacks) {
    plot
      .append("path")
      .datum(series.points)
      .attr("class", "loss-fraction-area")
      .attr("fill", args.colorForTerm.get(series.term) ?? "#999999")
      .attr("d", areaPath);
  }

  plot
    .append("g")
    .attr("class", "axis loss-fraction-y")
    .call(axisLeft(yFraction).tickValues([0, 0.5, 1]).tickSizeOuter(0).tickFormat((value) => `${value}`));
  plot
    .append("text")
    .attr("class", "axis-label loss-fraction-label")
    .attr("x", 4)
    .attr("y", 10)
    .text("Fraction");

  const cancellationY = top + fractionHeight + gap;
  const cancellationGroup = group
    .append("g")
    .attr("class", "loss-cancellation-plot")
    .attr("transform", `translate(${left},${cancellationY})`);
  const cancellationLine = line<[number, number]>()
    .defined(([xValue, yValue]) => Number.isFinite(xValue) && Number.isFinite(yValue))
    .x(([xValue]) => x(xValue))
    .y(([, yValue]) => yCancellation(yValue));
  const cancellation = args.panel.output.bin_centers.map(
    (center, index) => [center, cancellationAt(args.panel.output, index)] as [number, number],
  );
  cancellationGroup
    .append("path")
    .datum(cancellation)
    .attr("class", "loss-cancellation-line")
    .attr("d", cancellationLine);
  const meanCancellation = Math.max(0, 1 - args.panel.output.mean_coherence);
  cancellationGroup
    .append("line")
    .attr("class", "loss-cancellation-mean")
    .attr("x1", 0)
    .attr("x2", innerWidth)
    .attr("y1", yCancellation(meanCancellation))
    .attr("y2", yCancellation(meanCancellation));
  cancellationGroup
    .append("g")
    .attr("class", "axis loss-cancellation-y")
    .call(axisLeft(yCancellation).ticks(3).tickSizeOuter(0));
  cancellationGroup
    .append("g")
    .attr("class", "axis loss-cancellation-x")
    .attr("transform", `translate(0,${cancellationHeight})`)
    .call(axisBottom(x).ticks(Math.max(2, Math.min(5, Math.floor(innerWidth / 70)))).tickSizeOuter(0));
  cancellationGroup
    .append("text")
    .attr("class", "axis-label loss-cancellation-label")
    .attr("x", 4)
    .attr("y", 10)
    .text("κ");
  cancellationGroup
    .append("text")
    .attr("class", "axis-label loss-x-label")
    .attr("x", innerWidth)
    .attr("y", cancellationHeight + 24)
    .attr("text-anchor", "end")
    .text(args.axisLabel);
}

function panelTitle(panel: PanelDatum): string {
  if (!panel.output) return panel.label;
  const meanCancellation = 1 - panel.output.mean_coherence;
  return `${panel.label} · mean κ ${formatNumber(meanCancellation)}`;
}

function stackedTermData(output: LossDecompositionOutput, terms: string[]): Array<{ term: string; points: StackedPoint[] }> {
  const termMap = new Map(output.terms.map((term) => [term.id, term]));
  const accumulators = new Array(output.bin_centers.length).fill(0);
  return terms.map((termId) => {
    const term = termMap.get(termId);
    const values = termValues(term, output.bin_centers.length);
    const points = output.bin_centers.map((x, index) => {
      const y0 = accumulators[index];
      const value = Math.max(0, Number.isFinite(values[index]) ? values[index] : 0);
      const y1 = Math.min(1.05, y0 + value);
      accumulators[index] = y1;
      return { x, y0, y1 };
    });
    return { term: termId, points };
  });
}

function termValues(term: LossDecompositionTermSeries | undefined, expectedLength: number): number[] {
  if (!term) return new Array(expectedLength).fill(0);
  if (term.binned_fraction.length !== expectedLength) return new Array(expectedLength).fill(0);
  return term.binned_fraction;
}

function cancellationAt(output: LossDecompositionOutput | null, index: number): number {
  if (!output) return Number.NaN;
  const coherence = output.mean_coherence;
  const value = 1 - (Number.isFinite(coherence) ? outputMeanOrBinnedCoherence(output, index) : coherence);
  return Math.max(0, Math.min(1, value));
}

function outputMeanOrBinnedCoherence(output: LossDecompositionOutput, index: number): number {
  const value = output.binned_coherence[index];
  if (value != null && Number.isFinite(value)) return value;
  return output.mean_coherence;
}

function shortTermLabel(term: string): string {
  const [prefix, suffix] = term.split("_");
  if ((prefix === "pde" || prefix === "bc") && suffix) return `${prefix.toUpperCase()} ${suffix}`;
  return formatDisplayLabel(term.replace(/_/g, " "));
}

function drawEmpty(
  root: ReturnType<typeof select<SVGSVGElement, unknown>>,
  width: number,
  height: number,
  message: string,
): void {
  root
    .append("text")
    .attr("class", "loss-decomposition-empty")
    .attr("x", width / 2)
    .attr("y", height / 2)
    .attr("text-anchor", "middle")
    .text(message);
}
