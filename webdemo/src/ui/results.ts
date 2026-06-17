import { axisBottom, axisLeft, line, scaleBand, scaleLinear, select } from "d3";
import type {
  DirectionalityIndicatorEntry,
  LossDecompositionData,
  LossDecompositionTermSeries,
  ModelQuality,
  ResultsData,
  TemporalIndicatorEntry,
} from "../types";
import { renderLossDecompositionPlot } from "../viz/lossDecomposition";

const QUALITY_ORDER: ModelQuality[] = ["good", "bad"];
const QUALITY_LABEL: Record<ModelQuality, string> = { good: "Good", bad: "Bad" };
const QUALITY_COLOR: Record<ModelQuality, string> = { good: "#1f82c0", bad: "#d14f3f" };

export class ResultsDashboard {
  private data: ResultsData | null = null;
  private problem: string | null = null;
  private outputByProblem = new Map<string, string>();

  constructor(private readonly root: HTMLElement) {}

  setLoading(): void {
    this.root.innerHTML = `<div class="results-shell" data-state="loading"><div class="results-empty">Loading results</div></div>`;
  }

  setError(message: string): void {
    this.root.innerHTML = `<div class="results-shell" data-state="error"><div class="results-empty">${escapeHtml(message)}</div></div>`;
  }

  render(data: ResultsData, problem: string | null): void {
    this.data = data;
    this.problem = problem;
    this.renderCurrent();
  }

  rerender(): void {
    if (!this.data) return;
    this.renderCurrent();
  }

  private renderCurrent(): void {
    if (!this.data) return;
    const problem = this.activeProblemId();
    if (!problem) {
      this.setError("Results data is unavailable for the selected problem");
      return;
    }
    const lossDatasets = this.lossDatasets(problem);
    const outputIds = outputIdsFor(lossDatasets);
    const outputId = this.selectedOutput(problem, outputIds);
    const temporal = this.data.indicators.temporal.find((entry) => entry.problem === problem);
    const directionality = this.data.indicators.directionality.filter(
      (entry) => entry.problem === problem,
    );

    this.root.innerHTML = `
      <div class="results-shell" data-state="ready">
        <section class="results-grid">
          <article class="results-panel results-panel-wide">
            <div class="results-panel-header">
              <div>
                <h3>Loss Component Decomposition</h3>
                <p>Aggregate summary · fractions by PDE, boundary, and initial constraints.</p>
              </div>
              <div class="results-output-buttons" role="group" aria-label="Output selection">
                ${outputButtons(outputIds, outputId)}
              </div>
            </div>
            <svg class="results-chart results-loss-chart" data-results-chart="loss"></svg>
          </article>

          <article class="results-panel">
            <div class="results-panel-header">
              <div>
                <h3>${temporal ? "Temporal Indicator" : "Directionality Indicator"}</h3>
                <p>Paper aggregate · Good/Bad model comparison against baseline.</p>
              </div>
            </div>
            <svg class="results-chart" data-results-chart="indicator"></svg>
          </article>

          <article class="results-panel">
            <div class="results-panel-header">
              <div>
                <h3>Constraint Dominance</h3>
                <p>Aggregate summary · selected loss term over the problem axis.</p>
              </div>
            </div>
            <svg class="results-chart" data-results-chart="dominance"></svg>
          </article>
        </section>
      </div>
    `;

    this.root.querySelectorAll<HTMLButtonElement>("[data-results-output]").forEach((button) => {
      button.addEventListener("click", () => {
        const nextOutput = button.dataset.resultsOutput;
        if (!nextOutput) return;
        this.outputByProblem.set(problem, nextOutput);
        this.renderCurrent();
      });
    });

    const lossSvg = this.root.querySelector<SVGSVGElement>('[data-results-chart="loss"]');
    if (lossSvg) renderLossDecompositionPlot({ svg: lossSvg, datasets: lossDatasets, outputId });
    const indicatorSvg = this.root.querySelector<SVGSVGElement>('[data-results-chart="indicator"]');
    if (indicatorSvg) renderIndicatorChart(indicatorSvg, temporal, directionality, outputId);
    const dominanceSvg = this.root.querySelector<SVGSVGElement>('[data-results-chart="dominance"]');
    if (dominanceSvg) renderDominanceChart(dominanceSvg, lossDatasets, outputId);
  }

  private activeProblemId(): string | null {
    if (!this.data) return null;
    if (this.problem && hasResultsForProblem(this.data, this.problem)) return this.problem;
    return firstResultsProblemId(this.data);
  }

  private lossDatasets(problem: string): Partial<Record<ModelQuality, LossDecompositionData>> {
    const datasets: Partial<Record<ModelQuality, LossDecompositionData>> = {};
    if (!this.data) return datasets;
    for (const quality of QUALITY_ORDER) {
      const data = this.data.loss_decompositions.find(
        (entry) => entry.problem === problem && entry.quality === quality,
      );
      if (data) datasets[quality] = data;
    }
    return datasets;
  }

  private selectedOutput(problem: string, outputIds: string[]): string | null {
    if (!outputIds.length) return null;
    const selected = this.outputByProblem.get(problem);
    if (selected && outputIds.includes(selected)) return selected;
    const fallback = outputIds[0];
    this.outputByProblem.set(problem, fallback);
    return fallback;
  }
}

function hasResultsForProblem(data: ResultsData, problem: string): boolean {
  return (
    data.loss_decompositions.some((entry) => entry.problem === problem) ||
    data.indicators.temporal.some((entry) => entry.problem === problem) ||
    data.indicators.directionality.some((entry) => entry.problem === problem)
  );
}

function firstResultsProblemId(data: ResultsData): string | null {
  return (
    data.loss_decompositions[0]?.problem ??
    data.indicators.temporal[0]?.problem ??
    data.indicators.directionality[0]?.problem ??
    null
  );
}

function outputIdsFor(datasets: Partial<Record<ModelQuality, LossDecompositionData>>): string[] {
  const ids = new Set<string>();
  for (const data of Object.values(datasets)) {
    for (const output of data?.outputs ?? []) ids.add(output.id);
  }
  return Array.from(ids).sort();
}

function outputButtons(outputIds: string[], selected: string | null): string {
  if (outputIds.length <= 1) return "";
  return outputIds
    .map(
      (id) =>
        `<button type="button" data-results-output="${escapeHtml(id)}" class="${id === selected ? "active" : ""}">${escapeHtml(outputLabel(id))}</button>`,
    )
    .join("");
}

function outputLabel(id: string): string {
  const suffix = id.match(/(\d+)$/)?.[1];
  return suffix ? `Output ${suffix}` : id.replace(/_/g, " ");
}

function renderIndicatorChart(
  svg: SVGSVGElement,
  temporal: TemporalIndicatorEntry | undefined,
  directionality: DirectionalityIndicatorEntry[],
  outputId: string | null,
): void {
  const entries = temporal
    ? [
        {
          label: temporal.display_name,
          baseline: temporal.baseline,
          badBaseline: temporal.bad_baseline,
          values: temporal.values,
        },
      ]
    : directionality
        .filter((entry) => !outputId || entry.output_id === outputId || directionality.length <= 1)
        .map((entry) => ({
          label: entry.output_label,
          baseline: entry.baseline,
          values: entry.values,
        }));
  renderGroupedBarChart(svg, entries, "indicator");
}

function renderGroupedBarChart(
  svg: SVGSVGElement,
  entries: Array<{
    label: string;
    baseline: number;
    badBaseline?: number;
    values: Record<ModelQuality, { mean: number; std: number }>;
  }>,
  emptyContext: string,
): void {
  const { root, width, height } = prepareSvg(svg);
  if (!entries.length) {
    drawSvgEmpty(root, width, height, `${emptyContext} unavailable`);
    return;
  }
  const margin = { top: 18, right: 14, bottom: 42, left: 42 };
  const innerWidth = Math.max(1, width - margin.left - margin.right);
  const innerHeight = Math.max(1, height - margin.top - margin.bottom);
  const labels = entries.map((entry) => entry.label);
  const maxValue = Math.max(
    0.55,
    ...entries.flatMap((entry) =>
      QUALITY_ORDER.map((quality) => {
        const value = entry.values[quality];
        return value ? value.mean + value.std : 0;
      }),
    ),
  );
  const x = scaleBand<string>().domain(labels).range([0, innerWidth]).padding(0.28);
  const xQuality = scaleBand<ModelQuality>().domain(QUALITY_ORDER).range([0, x.bandwidth()]).padding(0.12);
  const y = scaleLinear().domain([0, Math.min(1, Math.ceil(maxValue * 10) / 10)]).nice().range([innerHeight, 0]);
  const plot = root.append("g").attr("transform", `translate(${margin.left},${margin.top})`);

  plot
    .append("g")
    .attr("class", "axis")
    .attr("transform", `translate(0,${innerHeight})`)
    .call(axisBottom(x).tickSizeOuter(0));
  plot.append("g").attr("class", "axis").call(axisLeft(y).ticks(4).tickSizeOuter(0));

  for (const entry of entries) {
    const groupX = x(entry.label);
    if (groupX == null) continue;
    const baselineY = y(entry.baseline);
    plot
      .append("line")
      .attr("class", "results-baseline")
      .attr("x1", groupX)
      .attr("x2", groupX + x.bandwidth())
      .attr("y1", baselineY)
      .attr("y2", baselineY);
    if (entry.badBaseline != null) {
      plot
        .append("line")
        .attr("class", "results-baseline results-baseline-alt")
        .attr("x1", groupX)
        .attr("x2", groupX + x.bandwidth())
        .attr("y1", y(entry.badBaseline))
        .attr("y2", y(entry.badBaseline));
    }
    for (const quality of QUALITY_ORDER) {
      const value = entry.values[quality];
      if (!value) continue;
      const barX = groupX + (xQuality(quality) ?? 0);
      const barWidth = xQuality.bandwidth();
      plot
        .append("rect")
        .attr("class", `results-bar results-bar-${quality}`)
        .attr("x", barX)
        .attr("y", y(value.mean))
        .attr("width", barWidth)
        .attr("height", Math.max(0, innerHeight - y(value.mean)))
        .attr("fill", QUALITY_COLOR[quality]);
      plot
        .append("line")
        .attr("class", "results-errorbar")
        .attr("x1", barX + barWidth / 2)
        .attr("x2", barX + barWidth / 2)
        .attr("y1", y(value.mean - value.std))
        .attr("y2", y(value.mean + value.std));
    }
  }
  drawChartLegend(root, width, 8);
}

function renderDominanceChart(
  svg: SVGSVGElement,
  datasets: Partial<Record<ModelQuality, LossDecompositionData>>,
  outputId: string | null,
): void {
  const { root, width, height } = prepareSvg(svg);
  const series = QUALITY_ORDER.flatMap((quality) => {
    const output = selectedOutput(datasets[quality], outputId);
    const term = selectedDominanceTerm(output?.terms ?? []);
    if (!output || !term) return [];
    return [{ quality, output, term }];
  });
  if (!series.length) {
    drawSvgEmpty(root, width, height, "dominance unavailable");
    return;
  }
  const margin = { top: 20, right: 16, bottom: 38, left: 42 };
  const innerWidth = Math.max(1, width - margin.left - margin.right);
  const innerHeight = Math.max(1, height - margin.top - margin.bottom);
  const xValues = series.flatMap((entry) => entry.output.bin_centers);
  const xDomain: [number, number] = [Math.min(...xValues), Math.max(...xValues)];
  const yMax = Math.max(0.1, ...series.flatMap((entry) => entry.term.binned_fraction));
  const x = scaleLinear().domain(xDomain).range([0, innerWidth]);
  const y = scaleLinear().domain([0, Math.min(1, Math.ceil(yMax * 10) / 10)]).nice().range([innerHeight, 0]);
  const plot = root.append("g").attr("transform", `translate(${margin.left},${margin.top})`);
  plot
    .append("g")
    .attr("class", "axis")
    .attr("transform", `translate(0,${innerHeight})`)
    .call(axisBottom(x).ticks(5).tickSizeOuter(0));
  plot.append("g").attr("class", "axis").call(axisLeft(y).ticks(4).tickSizeOuter(0));
  const linePath = line<[number, number]>()
    .defined(([xValue, yValue]) => Number.isFinite(xValue) && Number.isFinite(yValue))
    .x(([xValue]) => x(xValue))
    .y(([, yValue]) => y(yValue));
  for (const entry of series) {
    const points = entry.output.bin_centers.map(
      (center, index) => [center, entry.term.binned_fraction[index] ?? 0] as [number, number],
    );
    plot
      .append("path")
      .datum(points)
      .attr("class", `results-line results-line-${entry.quality}`)
      .attr("stroke", QUALITY_COLOR[entry.quality])
      .attr("d", linePath);
  }
  const label = series[0]?.term.label ?? "constraint";
  root
    .append("text")
    .attr("class", "results-chart-note")
    .attr("x", margin.left)
    .attr("y", 14)
    .text(label);
  drawChartLegend(root, width, 8);
}

function selectedOutput(
  data: LossDecompositionData | undefined,
  outputId: string | null,
) {
  if (!data?.outputs.length) return null;
  return (
    (outputId ? data.outputs.find((output) => output.id === outputId) : null) ??
    data.outputs[0]
  );
}

function selectedDominanceTerm(terms: LossDecompositionTermSeries[]): LossDecompositionTermSeries | null {
  return (
    terms.find((term) => term.id === "bc_0") ??
    terms.find((term) => term.id.startsWith("bc_")) ??
    terms.find((term) => term.id.startsWith("pde_")) ??
    terms[0] ??
    null
  );
}

function prepareSvg(svg: SVGSVGElement) {
  const width = Math.max(1, Math.floor(svg.clientWidth || svg.getBoundingClientRect().width || 320));
  const height = Math.max(1, Math.floor(svg.clientHeight || svg.getBoundingClientRect().height || 220));
  const root = select(svg);
  root.selectAll("*").remove();
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  return { root, width, height };
}

function drawSvgEmpty(
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

function drawChartLegend(
  root: ReturnType<typeof select<SVGSVGElement, unknown>>,
  width: number,
  y: number,
): void {
  const legend = root.append("g").attr("class", "results-legend").attr("transform", `translate(${Math.max(48, width - 150)},${y})`);
  QUALITY_ORDER.forEach((quality, index) => {
    const item = legend.append("g").attr("transform", `translate(${index * 66},0)`);
    item.append("rect").attr("width", 10).attr("height", 10).attr("fill", QUALITY_COLOR[quality]);
    item.append("text").attr("x", 14).attr("y", 10).text(QUALITY_LABEL[quality]);
  });
}

function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
