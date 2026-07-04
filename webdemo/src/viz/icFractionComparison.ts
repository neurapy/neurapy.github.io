import { area, axisBottom, axisLeft, line, scaleLinear, select } from "d3";
import type {
  LossDecompositionTermSeries,
  ModelQuality,
  ResultsData,
} from "../types";

interface SeriesDatum {
  problem: string;
  label: string;
  quality: ModelQuality;
  centers: number[];
  fractions: number[];
  std: number[];
}

interface BandPoint {
  x: number;
  y0: number;
  y1: number;
}

interface LinePoint {
  x: number;
  y: number;
}

const QUALITY_ORDER: ModelQuality[] = ["good", "bad"];
const QUALITY_LABELS: Record<ModelQuality, string> = {
  good: "Well-Trained",
  bad: "Poorly-Trained",
};
const PROBLEM_ORDER = ["allen_cahn", "burgers", "diffusion", "wave", "drift_diffusion"];
const PROBLEM_LABELS: Record<string, string> = {
  allen_cahn: "Allen-Cahn",
  burgers: "Burgers",
  diffusion: "Diffusion",
  wave: "Wave",
  drift_diffusion: "Drift-Diffusion",
};
const PROBLEM_COLORS: Record<string, string> = {
  allen_cahn: "#1f77b4",
  burgers: "#ff7f0e",
  diffusion: "#2ca02c",
  wave: "#d62728",
  drift_diffusion: "#9467bd",
};

export function renderIcFractionComparisonPlot(svg: SVGSVGElement, data: ResultsData): void {
  const width = Math.max(0, Math.floor(svg.clientWidth || svg.getBoundingClientRect().width));
  const height = Math.max(0, Math.floor(svg.clientHeight || svg.getBoundingClientRect().height));
  const root = select(svg);
  root.selectAll("*").remove();
  svg.setAttribute("viewBox", `0 0 ${Math.max(1, width)} ${Math.max(1, height)}`);
  if (width < 260 || height < 150) return;

  const series = icSeries(data);
  if (!series.length) {
    drawEmpty(root, width, height, "IC fraction unavailable");
    return;
  }

  const layout = plotLayout(width, height);
  drawLegend(root, layout.legend);
  const xDomain = sharedXDomain(series);
  const group = root.append("g").attr("class", "results-ic-panels");
  QUALITY_ORDER.forEach((quality, index) => {
    const frame = layout.frames[index];
    renderPanel(group, {
      quality,
      series: series.filter((entry) => entry.quality === quality),
      xDomain,
      x: frame.x,
      y: frame.y,
      width: frame.width,
      height: frame.height,
    });
  });
}

function icSeries(data: ResultsData): SeriesDatum[] {
  return PROBLEM_ORDER.flatMap((problem) =>
    QUALITY_ORDER.flatMap((quality) => {
      const record = data.loss_decompositions.find(
        (entry) => entry.problem === problem && entry.quality === quality,
      );
      const output = record?.outputs[0];
      const term = output ? initialConditionTerm(output.terms) : null;
      if (!record || !output || !term) return [];
      return [
        {
          problem,
          label: PROBLEM_LABELS[problem] ?? record.display_name,
          quality,
          centers: output.bin_centers,
          fractions: term.binned_fraction,
          std: term.binned_fraction_std,
        },
      ];
    }),
  );
}

function initialConditionTerm(terms: LossDecompositionTermSeries[]): LossDecompositionTermSeries | null {
  return (
    terms.find((term) => /\bIC\b/i.test(term.label)) ??
    terms.find((term) => term.id === "bc_0") ??
    null
  );
}

function sharedXDomain(series: SeriesDatum[]): [number, number] {
  const finite = series.flatMap((entry) => entry.centers).filter(Number.isFinite);
  if (!finite.length) return [0, 1];
  const min = Math.min(...finite);
  const max = Math.max(...finite);
  return min < max ? [min, max] : [min - 0.5, max + 0.5];
}

function plotLayout(width: number, height: number) {
  const margin = { top: 10, right: 13, bottom: 14, left: 13 };
  const legendItemWidth = 120;
  const legendColumns = Math.max(1, Math.floor((width - margin.left - margin.right) / legendItemWidth));
  const legendRows = Math.max(1, Math.ceil(PROBLEM_ORDER.length / legendColumns));
  const legendHeight = Math.min(42, 8 + legendRows * 16);
  const vertical = width < 640;
  const gap = vertical ? 20 : 22;
  const plotTop = margin.top + legendHeight;
  const plotWidth = Math.max(1, width - margin.left - margin.right);
  const plotHeight = Math.max(100, height - plotTop - margin.bottom);
  const frameWidth = vertical ? plotWidth : (plotWidth - gap) / 2;
  const frameHeight = vertical ? (plotHeight - gap) / 2 : plotHeight;
  return {
    legend: {
      x: margin.left,
      y: margin.top,
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
  legend: { x: number; y: number; itemWidth: number; columns: number },
): void {
  const group = root.append("g").attr("class", "results-ic-legend");
  PROBLEM_ORDER.forEach((problem, index) => {
    const col = index % legend.columns;
    const row = Math.floor(index / legend.columns);
    const item = group
      .append("g")
      .attr("transform", `translate(${legend.x + col * legend.itemWidth},${legend.y + row * 16})`);
    item
      .append("line")
      .attr("class", "results-ic-legend-line")
      .attr("x1", 0)
      .attr("x2", 16)
      .attr("y1", 7)
      .attr("y2", 7)
      .attr("stroke", PROBLEM_COLORS[problem]);
    item.append("text").attr("x", 22).attr("y", 10).text(PROBLEM_LABELS[problem]);
  });
}

function renderPanel(
  parent: ReturnType<typeof select<SVGGElement, unknown>>,
  args: {
    quality: ModelQuality;
    series: SeriesDatum[];
    xDomain: [number, number];
    x: number;
    y: number;
    width: number;
    height: number;
  },
): void {
  const group = parent
    .append("g")
    .attr("class", `results-ic-panel results-ic-${args.quality}`)
    .attr("transform", `translate(${args.x},${args.y})`);
  group.append("rect").attr("class", "loss-decomposition-frame").attr("width", args.width).attr("height", args.height);
  group
    .append("text")
    .attr("class", "loss-decomposition-panel-title")
    .attr("x", args.width / 2)
    .attr("y", args.height - 4)
    .attr("text-anchor", "middle")
    .text(QUALITY_LABELS[args.quality]);

  const margin = { top: 10, right: 10, bottom: 34, left: 43 };
  const innerWidth = Math.max(1, args.width - margin.left - margin.right);
  const innerHeight = Math.max(1, args.height - margin.top - margin.bottom);
  const x = scaleLinear().domain(args.xDomain).range([0, innerWidth]);
  const y = scaleLinear().domain([0, 1]).range([innerHeight, 0]);
  const plot = group.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  plot
    .append("g")
    .attr("class", "axis-grid")
    .call(axisLeft(y).tickValues([0, 0.25, 0.5, 0.75, 1]).tickSize(-innerWidth).tickFormat(() => ""));
  plot
    .append("g")
    .attr("class", "axis-grid")
    .attr("transform", `translate(0,${innerHeight})`)
    .call(axisBottom(x).ticks(5).tickSize(-innerHeight).tickFormat(() => ""));

  const bandPath = area<BandPoint>()
    .defined((point) => Number.isFinite(point.x) && Number.isFinite(point.y0) && Number.isFinite(point.y1))
    .x((point) => x(point.x))
    .y0((point) => y(point.y0))
    .y1((point) => y(point.y1));
  const linePath = line<LinePoint>()
    .defined((point) => Number.isFinite(point.x) && Number.isFinite(point.y))
    .x((point) => x(point.x))
    .y((point) => y(point.y));

  for (const entry of args.series) {
    const color = PROBLEM_COLORS[entry.problem] ?? "#999999";
    plot
      .append("path")
      .datum(bandData(entry))
      .attr("class", "results-ic-band")
      .attr("fill", color)
      .attr("d", bandPath);
    plot
      .append("path")
      .datum(lineData(entry))
      .attr("class", "results-ic-line")
      .attr("stroke", color)
      .attr("d", linePath);
  }

  plot
    .append("g")
    .attr("class", "axis")
    .attr("transform", `translate(0,${innerHeight})`)
    .call(axisBottom(x).ticks(5).tickSizeOuter(0));
  plot.append("g").attr("class", "axis").call(axisLeft(y).tickValues([0, 0.5, 1]).tickSizeOuter(0));
  plot
    .append("text")
    .attr("class", "axis-label")
    .attr("transform", `translate(-31,${innerHeight / 2}) rotate(-90)`)
    .attr("text-anchor", "middle")
    .text("IC Fraction");
  plot
    .append("text")
    .attr("class", "axis-label")
    .attr("x", innerWidth)
    .attr("y", innerHeight + 27)
    .attr("text-anchor", "end")
    .text("t");
}

function bandData(entry: SeriesDatum): BandPoint[] {
  return entry.centers.map((x, index) => {
    const value = entry.fractions[index] ?? Number.NaN;
    const std = entry.std[index] ?? 0;
    return { x, y0: clamp01(value - std), y1: clamp01(value + std) };
  });
}

function lineData(entry: SeriesDatum): LinePoint[] {
  return entry.centers.map((x, index) => ({ x, y: clamp01(entry.fractions[index] ?? Number.NaN) }));
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return Number.NaN;
  return Math.max(0, Math.min(1, value));
}

function drawEmpty(
  root: ReturnType<typeof select<SVGSVGElement, unknown>>,
  width: number,
  height: number,
  message: string,
): void {
  root
    .append("text")
    .attr("class", "results-empty-svg")
    .attr("x", width / 2)
    .attr("y", height / 2)
    .attr("text-anchor", "middle")
    .text(message);
}
