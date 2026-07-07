import type {
  DirectionalityIndicatorEntry,
  LossDecompositionData,
  LossDecompositionOutput,
  ModelQuality,
  PaperIndicatorValue,
  ResultsData,
  TemporalIndicatorEntry,
} from "../types";
import { formatFieldSelectLabel } from "./dom";
import { renderIcFractionComparisonPlot } from "../viz/icFractionComparison";
import { renderLossDecompositionPlot } from "../viz/lossDecomposition";

const QUALITY_ORDER: ModelQuality[] = ["good", "bad"];
const ETA_FORMULA = `
  <math display="block" aria-label="eta of R equals one minus the average upstream absolute influence fraction over candidate points z in R">
    <mrow>
      <mi>η</mi>
      <mo>(</mo><mi>R</mi><mo>)</mo>
      <mo>=</mo>
      <mn>1</mn>
      <mo>−</mo>
      <mfrac>
        <mn>1</mn>
        <mrow><mo>|</mo><mi>R</mi><mo>|</mo></mrow>
      </mfrac>
      <msub>
        <mo>∑</mo>
        <mrow><mi>z</mi><mo>∈</mo><mi>R</mi></mrow>
      </msub>
      <mfrac>
        <mrow>
          <msub>
            <mo>∑</mo>
            <mrow><mi>x</mi><mo>∈</mo><mi>U</mi><mo>(</mo><mi>z</mi><mo>)</mo></mrow>
          </msub>
          <mo>|</mo><mi>I</mi><mo>(</mo><mi>x</mi><mo>,</mo><mi>z</mi><mo>)</mo><mo>|</mo>
        </mrow>
        <mrow>
          <msub>
            <mo>∑</mo>
            <mrow><mi>x</mi><mo>∈</mo><msub><mi>X</mi><mtext>train</mtext></msub></mrow>
          </msub>
          <mo>|</mo><mi>I</mi><mo>(</mo><mi>x</mi><mo>,</mo><mi>z</mi><mo>)</mo><mo>|</mo>
        </mrow>
      </mfrac>
    </mrow>
  </math>
`;
const RESULTS_HINTS = {
  loss: {
    title: "Loss Component Decomposition",
    points: [
      "Contribution of training loss terms to absolute influence, averaged within intervals along the x-axis.",
      "Cancellation κ rises when positive and negative signed influences of loss terms offset each other.",
      "Shaded bands show sample-to-sample variation inside a bin.",
    ],
  },
  ic: {
    title: "IC Fraction Across Problems",
    points: [
      "Share of absolute influence assigned to initial-condition terms as time changes.",
      "High IC influence near t=0 is expected. Slow decay or late peaks suggest the model remains dependent on initial-condition data.",
      "Shaded bands show sample-to-sample variation inside a bin.",
    ],
  },
  indicator: {
    title: "Influence Indicator",
    formula: ETA_FORMULA,
    points: [
      "η compresses directional influence into one number for the selected problem.",
      "R is the set of candidate points being summarized; U(z) is the set of earlier or upstream training points for candidate point z.",
      "The fraction is computed for each z and then averaged over R; because of the 1 - term, lower η means stronger earlier/upstream influence.",
      "Temporal η uses earlier times; spatial η uses the analogous upstream direction in space.",
      "Baseline is the value expected from the training-point layout alone, before considering the model's learned influence pattern.",
      "Well-Trained and Poorly-Trained rows report mean ± standard deviation across runs.",
      "Compare Well-Trained and Poorly-Trained values against the listed baseline before judging a model.",
      "Values above baseline are prompts for inspection, not automatic proof that a model is better.",
    ],
  },
} as const;
type ResultsHintId = keyof typeof RESULTS_HINTS;
interface ResultsOutputOption {
  id: string;
  label: string;
}

export class ResultsDashboard {
  private data: ResultsData | null = null;
  private problem: string | null = null;
  private outputByProblem = new Map<string, string>();

  constructor(private readonly root: HTMLElement) {
    this.root.addEventListener("click", this.handleRootClick);
    document.addEventListener("click", this.handleDocumentClick);
    document.addEventListener("keydown", this.handleDocumentKeydown);
  }

  setLoading(): void {
    this.root.innerHTML = `<div class="results-shell" data-state="loading"><div class="results-empty">Loading indicators</div></div>`;
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
      this.setError("Indicator data is unavailable for the selected problem");
      return;
    }
    const lossDatasets = this.lossDatasets(problem);
    const outputOptions = outputOptionsFor(lossDatasets);
    const outputIds = outputOptions.map((output) => output.id);
    const outputId = this.selectedOutput(problem, outputIds);

    this.root.innerHTML = `
      <div class="results-shell" data-state="ready">
        <section class="results-grid">
          <article class="results-panel results-panel-full">
            <div class="results-panel-header">
              ${panelTitle("loss")}
              <div class="results-output-buttons" role="group" aria-label="Output selection">
                ${outputButtons(outputOptions, outputId)}
              </div>
              ${hintPopover("loss")}
            </div>
            <svg class="results-chart results-loss-chart" data-results-chart="loss"></svg>
          </article>
          <article class="results-panel results-panel-wide">
            <div class="results-panel-header">
              ${panelTitle("ic")}
              ${hintPopover("ic")}
            </div>
            <svg class="results-chart results-ic-chart" data-results-chart="ic"></svg>
          </article>
          <article class="results-panel">
            <div class="results-panel-header">
              ${panelTitle("indicator")}
              ${hintPopover("indicator")}
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
    if (lossSvg) renderLossDecompositionPlot({ svg: lossSvg, datasets: lossDatasets, outputId, problem });
    const icSvg = this.root.querySelector<SVGSVGElement>('[data-results-chart="ic"]');
    if (icSvg) renderIcFractionComparisonPlot(icSvg, this.data);
  }

  private readonly handleRootClick = (event: MouseEvent): void => {
    const target = event.target;
    if (!(target instanceof Element)) return;
    const button = target.closest<HTMLButtonElement>("[data-results-hint]");
    if (button && this.root.contains(button)) {
      this.toggleHint(button);
      return;
    }
    if (!target.closest(".results-hint-popover")) this.closeHints();
  };

  private readonly handleDocumentClick = (event: MouseEvent): void => {
    const target = event.target;
    if (target instanceof Node && !this.root.contains(target)) this.closeHints();
  };

  private readonly handleDocumentKeydown = (event: KeyboardEvent): void => {
    if (event.key === "Escape") this.closeHints();
  };

  private toggleHint(button: HTMLButtonElement): void {
    const controls = button.getAttribute("aria-controls");
    if (!button.dataset.resultsHint || !controls) return;
    const popover = document.getElementById(controls);
    if (!(popover instanceof HTMLElement) || !this.root.contains(popover)) return;
    const shouldOpen = button.getAttribute("aria-expanded") !== "true";
    this.closeHints();
    button.setAttribute("aria-expanded", String(shouldOpen));
    popover.hidden = !shouldOpen;
  }

  private closeHints(): void {
    this.root.querySelectorAll<HTMLButtonElement>("[data-results-hint]").forEach((button) => {
      button.setAttribute("aria-expanded", "false");
    });
    this.root.querySelectorAll<HTMLElement>(".results-hint-popover").forEach((popover) => {
      popover.hidden = true;
    });
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

function panelTitle(id: ResultsHintId): string {
  const hint = RESULTS_HINTS[id];
  return `
    <div class="results-panel-title-row">
      <h3>${escapeHtml(hint.title)}</h3>
      <button
        type="button"
        class="results-hint-button"
        data-results-hint="${id}"
        aria-label="Explain ${escapeHtml(hint.title)}"
        aria-expanded="false"
        aria-controls="results-hint-${id}"
      >?</button>
    </div>
  `;
}

function hintPopover(id: ResultsHintId): string {
  const hint = RESULTS_HINTS[id];
  return `
    <div class="results-hint-popover" id="results-hint-${id}" role="note" aria-label="${escapeHtml(hint.title)} hint" hidden>
      <div class="results-hint-title">${escapeHtml(hint.title)}</div>
      ${"formula" in hint ? `<div class="results-hint-formula">${hint.formula}</div>` : ""}
      <ul>
        ${hint.points.map((point) => `<li>${escapeHtml(point)}</li>`).join("")}
      </ul>
    </div>
  `;
}

function hasResultsForProblem(data: ResultsData, problem: string): boolean {
  return data.loss_decompositions.some((entry) => entry.problem === problem);
}

function firstResultsProblemId(data: ResultsData): string | null {
  return data.loss_decompositions[0]?.problem ?? null;
}

function outputOptionsFor(
  datasets: Partial<Record<ModelQuality, LossDecompositionData>>,
): ResultsOutputOption[] {
  const outputsById = new Map<string, ResultsOutputOption>();
  for (const quality of QUALITY_ORDER) {
    for (const output of datasets[quality]?.outputs ?? []) {
      if (!outputsById.has(output.id)) {
        outputsById.set(output.id, { id: output.id, label: formatResultsOutputLabel(output) });
      }
    }
  }
  return Array.from(outputsById.values()).sort((a, b) =>
    a.id.localeCompare(b.id, undefined, { numeric: true }),
  );
}

function outputButtons(outputOptions: readonly ResultsOutputOption[], selected: string | null): string {
  if (outputOptions.length <= 1) return "";
  return outputOptions
    .map((output) => {
      const id = escapeHtml(output.id);
      const label = escapeHtml(output.label);
      return `<button type="button" data-results-output="${id}" class="${output.id === selected ? "active" : ""}">${label}</button>`;
    })
    .join("");
}

export function formatResultsOutputLabel(output: Pick<LossDecompositionOutput, "id" | "label">): string {
  return formatFieldSelectLabel(output.label) || outputLabelFromId(output.id);
}

function outputLabelFromId(id: string): string {
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
  return paperIndicatorTable("Temporal influence", [
    ["Sampling baseline", formatScalar(entry.baseline)],
    ...(entry.bad_baseline == null ? [] : [["Poorly-Trained baseline", formatScalar(entry.bad_baseline)] as const]),
    ["Well-Trained", formatMeanStd(entry.values.good)],
    ["Poorly-Trained", formatMeanStd(entry.values.bad)],
  ]);
}

function directionalityIndicatorTable(entry: DirectionalityIndicatorEntry): string {
  return paperIndicatorTable(formatDirectionalityIndicatorLabel(entry), [
    ["Spatial baseline", formatScalar(entry.baseline)],
    ["Well-Trained", formatMeanStd(entry.values.good)],
    ["Poorly-Trained", formatMeanStd(entry.values.bad)],
  ]);
}

export function formatDirectionalityIndicatorLabel(
  entry: Pick<DirectionalityIndicatorEntry, "problem" | "output_id" | "output_label">,
): string {
  if (entry.problem === "poisson_disk" && entry.output_id === "output_0") {
    return "|x| directional influence (distance from [0, 0])";
  }
  return `${entry.output_label} directional influence`;
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
