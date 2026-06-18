import type {
  DirectionalityIndicatorEntry,
  LossDecompositionData,
  ModelQuality,
  PaperIndicatorValue,
  ResultsData,
  TemporalIndicatorEntry,
} from "../types";
import { renderIcFractionComparisonPlot } from "../viz/icFractionComparison";
import { renderLossDecompositionPlot } from "../viz/lossDecomposition";

const QUALITY_ORDER: ModelQuality[] = ["good", "bad"];

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

    this.root.innerHTML = `
      <div class="results-shell" data-state="ready">
        <section class="results-grid">
          <article class="results-panel results-panel-full">
            <div class="results-panel-header">
              <h3>Loss Component Decomposition</h3>
              <div class="results-output-buttons" role="group" aria-label="Output selection">
                ${outputButtons(outputIds, outputId)}
              </div>
            </div>
            <svg class="results-chart results-loss-chart" data-results-chart="loss"></svg>
          </article>
          <article class="results-panel results-panel-wide">
            <div class="results-panel-header">
              <h3>IC Fraction Across Problems</h3>
            </div>
            <svg class="results-chart results-ic-chart" data-results-chart="ic"></svg>
          </article>
          <article class="results-panel">
            <div class="results-panel-header">
              <h3>Influence Indicator</h3>
            </div>
            <div class="results-indicator-table" data-results-table="indicator">
              ${indicatorTable(this.data, problem, outputId)}
            </div>
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
    const icSvg = this.root.querySelector<SVGSVGElement>('[data-results-chart="ic"]');
    if (icSvg) renderIcFractionComparisonPlot(icSvg, this.data);
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
  return data.loss_decompositions.some((entry) => entry.problem === problem);
}

function firstResultsProblemId(data: ResultsData): string | null {
  return data.loss_decompositions[0]?.problem ?? null;
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

function indicatorTable(data: ResultsData, problem: string, outputId: string | null): string {
  const temporal = data.indicators.temporal.find((entry) => entry.problem === problem);
  if (temporal) return temporalIndicatorTable(temporal);
  const directionality =
    (outputId
      ? data.indicators.directionality.find(
          (entry) => entry.problem === problem && entry.output_id === outputId,
        )
      : null) ??
    data.indicators.directionality.find((entry) => entry.problem === problem) ??
    null;
  if (directionality) return directionalityIndicatorTable(directionality);
  return `<div class="results-empty">Indicator unavailable</div>`;
}

function temporalIndicatorTable(entry: TemporalIndicatorEntry): string {
  return paperIndicatorTable("Temporal η", [
    ["Sampling baseline", formatScalar(entry.baseline)],
    ...(entry.bad_baseline == null ? [] : [["Poor-model baseline", formatScalar(entry.bad_baseline)] as const]),
    ["Well-trained", formatMeanStd(entry.values.good)],
    ["Poorly-Trained", formatMeanStd(entry.values.bad)],
  ]);
}

function directionalityIndicatorTable(entry: DirectionalityIndicatorEntry): string {
  return paperIndicatorTable(`${escapeHtml(entry.output_label)} η`, [
    ["Spatial baseline", formatScalar(entry.baseline)],
    ["Well-trained", formatMeanStd(entry.values.good)],
    ["Poorly-Trained", formatMeanStd(entry.values.bad)],
  ]);
}

function paperIndicatorTable(label: string, rows: readonly (readonly [string, string])[]): string {
  return `
    <table>
      <thead>
        <tr><th>${escapeHtml(label)}</th><th>Value</th></tr>
      </thead>
      <tbody>
        ${rows
          .map(([name, value]) => `<tr><td>${escapeHtml(name)}</td><td>${escapeHtml(value)}</td></tr>`)
          .join("")}
      </tbody>
    </table>
  `;
}

function formatMeanStd(value: PaperIndicatorValue | undefined): string {
  if (!value) return "-";
  return `${formatScalar(value.mean)} ± ${formatScalar(value.std)}`;
}

function formatScalar(value: number): string {
  return Number.isFinite(value) ? value.toFixed(2) : "-";
}

function escapeHtml(value: unknown): string {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}
