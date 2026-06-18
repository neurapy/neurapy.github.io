import { area, axisBottom, axisLeft, line, scaleLinear, select } from "d3";
import type {
  LossDecompositionData,
  LossDecompositionOutput,
  LossDecompositionTermSeries,
  ModelQuality,
} from "../types";
import { formatDisplayLabel } from "../ui/dom";

export interface LossDecompositionRenderArgs {
  svg: SVGSVGElement;
  datasets: Partial<Record<ModelQuality, LossDecompositionData>>;
  outputId: string | null;
  problem: string | null;
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

interface PanelDatum {
  quality: ModelQuality;
  label: string;
  data: LossDecompositionData | null;
  output: LossDecompositionOutput | null;
}

const QUALITY_LABELS: Record<ModelQuality, string> = {
  good: "Well-trained",
  bad: "Poorly-Trained",
};
const QUALITY_ORDER: ModelQuality[] = ["good", "bad"];
const LOSS_TERM_ALIASES: Record<string, Record<string, string>> = {
  allen_cahn: {
    pde_loss: "PDE",
    pde_0: "PDE",
    bc_0: "IC",
    bc_1: "BC: x=1",
    bc_2: "BC: x=-1",
  },
  burgers: {
    pde_loss: "PDE",
    pde_0: "PDE",
    bc_0: "IC",
    bc_1: "BC",
  },
  diffusion: {
    pde_loss: "PDE",
    pde_0: "PDE",
    bc_0: "IC",
    bc_1: "BC",
  },
  drift_diffusion: {
    pde_loss: "PDE",
    pde_0: "PDE",
    bc_0: "IC",
    bc_1: "BC: x=0",
    bc_2: "BC: x=2π",
  },
  wave: {
    pde_loss: "PDE",
    pde_0: "PDE",
    bc_0: "IC",
    bc_1: "BC: u(0,t)=0",
    bc_2: "BC: u(1,t)=0",
    bc_3: "IC: ∂u/∂t",
  },
  poisson_disk: {
    pde_loss: "PDE",
    pde_0: "PDE",
    bc_loss: "BC",
    bc_0: "BC",
  },
  navier_stokes_nd: {
    pde_0: "Continuity",
    pde_1: "PDE: x-momentum",
    pde_2: "PDE: y-momentum",
    bc_0: "No-slip u",
    bc_1: "No-slip v",
    bc_2: "Inflow u",
    bc_3: "Inflow v",
    bc_4: "Outflow u",
    bc_5: "Outflow v",
  },
};
const TERM_COLORS = [
  "#1f77b4",
  "#ff7f0e",
  "#2ca02c",
  "#d62728",
  "#9467bd",
  "#8c564b",
  "#e377c2",
  "#7f7f7f",
  "#bcbd22",
  "#17becf",
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

  const axisLabel = panels.find((panel) => panel.output)?.data?.axis.label ?? "Bin center";
  const xDomain = sharedXDomain(panels);
  const colorForTerm = new Map(terms.map((term, index) => [term, TERM_COLORS[index % TERM_COLORS.length]]));
  const layout = plotLayout(width, height, terms.length);
  drawLegend(root, terms, colorForTerm, layout.legend, panels, args.problem);

  const group = root.append("g").attr("class", "loss-decomposition-panels");
  panels.forEach((panel, index) => {
    const frame = layout.frames[index];
    renderPanel(group, {
      panel,
      terms,
      colorForTerm,
      xDomain,
      axisLabel,
      x: frame.x,
      y: frame.y,
      width: frame.width,
      height: frame.height,
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
    for (const term of panel.output?.terms ?? []) terms.add(term.id);
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

function plotLayout(width: number, height: number, termCount: number) {
  const margin = { top: 11, right: 13, bottom: 14, left: 13 };
  const legendItemWidth = 118;
  const legendColumns = Math.max(1, Math.floor((width - margin.left - margin.right) / legendItemWidth));
  const legendRows = Math.max(1, Math.ceil((termCount + 1) / legendColumns));
  const legendHeight = Math.min(58, 8 + legendRows * 16);
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
  terms: string[],
  colorForTerm: Map<string, string>,
  legend: { x: number; y: number; itemWidth: number; columns: number },
  panels: PanelDatum[],
  problem: string | null,
): void {
  const group = root.append("g").attr("class", "loss-decomposition-legend");
  [...terms, "cancellation"].forEach((term, index) => {
    const col = index % legend.columns;
    const row = Math.floor(index / legend.columns);
    const item = group
      .append("g")
      .attr("transform", `translate(${legend.x + col * legend.itemWidth},${legend.y + row * 16})`);
    if (term === "cancellation") {
      item
        .append("line")
        .attr("class", "loss-cancellation-legend-line")
        .attr("x1", 0)
        .attr("x2", 16)
        .attr("y1", 7)
        .attr("y2", 7);
      item.append("text").attr("x", 22).attr("y", 10).text("Cancellation κ");
      return;
    }
    item
      .append("line")
      .attr("class", "loss-fraction-legend-line")
      .attr("x1", 0)
      .attr("x2", 16)
      .attr("y1", 7)
      .attr("y2", 7)
      .attr("stroke", colorForTerm.get(term) ?? "#999999");
    item.append("text").attr("x", 22).attr("y", 10).text(lossTermLegendLabel(problem, term, panels));
  });
}

function renderPanel(
  parent: ReturnType<typeof select<SVGGElement, unknown>>,
  args: {
    panel: PanelDatum;
    terms: string[];
    colorForTerm: Map<string, string>;
    xDomain: [number, number];
    axisLabel: string;
    x: number;
    y: number;
    width: number;
    height: number;
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
    .attr("x", args.width / 2)
    .attr("y", args.height - 4)
    .attr("text-anchor", "middle")
    .text(args.panel.label);

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

  for (const termId of args.terms) {
    const term = args.panel.output.terms.find((entry) => entry.id === termId);
    if (!term) continue;
    const color = args.colorForTerm.get(termId) ?? "#999999";
    plot
      .append("path")
      .datum(termBandData(args.panel.output, term))
      .attr("class", "loss-fraction-area")
      .attr("fill", color)
      .attr("d", bandPath);
    plot
      .append("path")
      .datum(termLineData(args.panel.output, term))
      .attr("class", "loss-fraction-line")
      .attr("stroke", color)
      .attr("d", linePath);
  }

  plot
    .append("path")
    .datum(cancellationData(args.panel.output))
    .attr("class", "loss-cancellation-line")
    .attr("d", linePath);
  plot
    .append("g")
    .attr("class", "axis")
    .attr("transform", `translate(0,${innerHeight})`)
    .call(axisBottom(x).ticks(5).tickSizeOuter(0));
  plot.append("g").attr("class", "axis").call(axisLeft(y).tickValues([0, 0.5, 1]).tickSizeOuter(0));
  plot
    .append("text")
    .attr("class", "axis-label loss-fraction-label")
    .attr("transform", `translate(-31,${innerHeight / 2}) rotate(-90)`)
    .attr("text-anchor", "middle")
    .text("Loss Fraction");
  plot
    .append("text")
    .attr("class", "axis-label loss-x-label")
    .attr("x", innerWidth)
    .attr("y", innerHeight + 27)
    .attr("text-anchor", "end")
    .text(args.axisLabel);
}

function termLineData(output: LossDecompositionOutput, term: LossDecompositionTermSeries): LinePoint[] {
  return output.bin_centers.map((x, index) => ({
    x,
    y: clamp01(term.binned_fraction[index] ?? Number.NaN),
  }));
}

function termBandData(output: LossDecompositionOutput, term: LossDecompositionTermSeries): BandPoint[] {
  return output.bin_centers.map((x, index) => {
    const value = term.binned_fraction[index] ?? Number.NaN;
    const std = term.binned_fraction_std[index] ?? 0;
    return {
      x,
      y0: clamp01(value - std),
      y1: clamp01(value + std),
    };
  });
}

function cancellationData(output: LossDecompositionOutput): LinePoint[] {
  return output.bin_centers.map((x, index) => ({
    x,
    y: cancellationAt(output, index),
  }));
}

function cancellationAt(output: LossDecompositionOutput | null, index: number): number {
  if (!output) return Number.NaN;
  const value = output.binned_coherence[index];
  const coherence = value != null && Number.isFinite(value) ? value : output.mean_coherence;
  return clamp01(1 - coherence);
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return Number.NaN;
  return Math.max(0, Math.min(1, value));
}

function lossTermLegendLabel(problem: string | null, termId: string, panels: PanelDatum[]): string {
  const alias = problem ? LOSS_TERM_ALIASES[problem]?.[termId] : undefined;
  if (alias) return alias;
  const metadataLabel = panels
    .flatMap((panel) => panel.output?.terms ?? [])
    .find((term) => term.id === termId)?.label;
  return formatDisplayLabel(metadataLabel ?? termId.replace(/_/g, " "));
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
