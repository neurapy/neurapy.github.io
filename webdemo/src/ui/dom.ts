import type { InfluenceMatrixManifest } from "../types";

export interface DomRefs {
  viewButtons: HTMLElement;
  runMeta: HTMLElement;
  problemSelect: HTMLSelectElement;
  qualityButtons: HTMLElement;
  resetButton: HTMLButtonElement;
  message: HTMLElement;
  playgroundWorkspace: HTMLElement;
  resultsWorkspace: HTMLElement;
  fieldSelect: HTMLSelectElement;
  matrixSelect: HTMLSelectElement;
  signButtons: HTMLElement;
  backgroundButtons: HTMLElement;
  kControl: HTMLElement;
  kSlider: HTMLInputElement;
  kOutput: HTMLOutputElement;
  modelPanel: HTMLElement;
  selectedPointLabel: HTMLElement;
  selectedPoint: HTMLElement;
  selectedValueLabel: HTMLElement;
  selectedValue: HTMLElement;
  plotGrid: HTMLElement;
  mainTitle: HTMLElement;
  trainTitle: HTMLElement;
  trainRange: HTMLElement;
  mainCanvas: HTMLCanvasElement;
  trainCanvas: HTMLCanvasElement;
  mainSvg: SVGSVGElement;
  trainSvg: SVGSVGElement;
  modelInteractionHint: HTMLElement;
  trainPanel: HTMLElement;
}

function required<T extends Element>(selector: string, ctor: new (...args: never[]) => T): T {
  const element = document.querySelector(selector);
  if (!element || !(element instanceof ctor)) {
    throw new Error(`Missing required element ${selector}`);
  }
  return element;
}

export function getDomRefs(): DomRefs {
  return {
    viewButtons: required("#viewButtons", HTMLElement),
    runMeta: required("#runMeta", HTMLElement),
    problemSelect: required("#problemSelect", HTMLSelectElement),
    qualityButtons: required("#qualityButtons", HTMLElement),
    resetButton: required("#resetButton", HTMLButtonElement),
    message: required("#message", HTMLElement),
    playgroundWorkspace: required("#playgroundWorkspace", HTMLElement),
    resultsWorkspace: required("#resultsWorkspace", HTMLElement),
    fieldSelect: required("#fieldSelect", HTMLSelectElement),
    matrixSelect: required("#matrixSelect", HTMLSelectElement),
    signButtons: required("#signButtons", HTMLElement),
    backgroundButtons: required("#backgroundButtons", HTMLElement),
    kControl: required("#kControl", HTMLElement),
    kSlider: required("#kSlider", HTMLInputElement),
    kOutput: required("#kOutput", HTMLOutputElement),
    modelPanel: required("#modelPanel", HTMLElement),
    selectedPointLabel: required("#selectedPointLabel", HTMLElement),
    selectedPoint: required("#selectedPoint", HTMLElement),
    selectedValueLabel: required("#selectedValueLabel", HTMLElement),
    selectedValue: required("#selectedValue", HTMLElement),
    plotGrid: required("#plotGrid", HTMLElement),
    mainTitle: required("#mainTitle", HTMLElement),
    trainTitle: required("#trainTitle", HTMLElement),
    trainRange: required("#trainRange", HTMLElement),
    mainCanvas: required("#mainCanvas", HTMLCanvasElement),
    trainCanvas: required("#trainCanvas", HTMLCanvasElement),
    mainSvg: required("#mainSvg", SVGSVGElement),
    trainSvg: required("#trainSvg", SVGSVGElement),
    modelInteractionHint: required("#modelInteractionHint", HTMLElement),
    trainPanel: required("#trainPanel", HTMLElement),
  };
}

export function showMessage(element: HTMLElement, text: string | null): void {
  if (!text) {
    element.hidden = true;
    element.textContent = "";
    return;
  }
  element.hidden = false;
  element.textContent = text;
}

export function formatNumber(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  if ((abs > 0 && abs < 0.001) || abs >= 10000) {
    return value.toExponential(3);
  }
  return value.toLocaleString(undefined, { maximumSignificantDigits: 5 });
}

export function formatReadoutNumber(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  if ((abs > 0 && abs < 0.001) || abs >= 10000) {
    return value.toExponential(2);
  }
  return value.toLocaleString(undefined, { maximumSignificantDigits: 4 });
}

export function formatRegionReadoutNumber(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  if ((abs > 0 && abs < 0.01) || abs >= 1000) {
    return value.toExponential(1);
  }
  return value.toLocaleString(undefined, { maximumSignificantDigits: 3 });
}

function formatLatexExpression(expression: string): string {
  return expression
    .replace(
      /\\frac\s*\{\s*\\partial\s+([^{}]+?)\s*\}\s*\{\s*\\partial\s+([^{}]+?)\s*\}/g,
      (_match, numerator: string, denominator: string) =>
        `∂${formatLatexExpression(numerator)}/∂${formatLatexExpression(denominator)}`,
    )
    .replace(/\\hat\s*\{?([A-Za-z])\}?/g, (_match, variable: string) => `${variable}\u0302`)
    .replace(/\\partial/g, "∂")
    .replace(/\\pi/g, "π")
    .replace(/\\cdot/g, "·")
    .replace(/\\times/g, "×")
    .replace(/\\_/g, "_")
    .replace(/[{}]/g, "")
    .replace(/\\/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

export function formatDisplayLabel(label: string | null | undefined): string {
  if (!label) return "";
  return label
    .replace(/\$([^$]*)\$/g, (_match, expression: string) => formatLatexExpression(expression))
    .replace(/\s+/g, " ")
    .trim();
}

const LOSS_SYMBOL = "ℒ";
const OUTPUT_FALLBACK_SYMBOL = "ŷ";
const TERM_ARROW = "→";

function subscriptDigits(value: string): string {
  const subscripts: Record<string, string> = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
  };
  return value.replace(/\d/g, (digit) => subscripts[digit] ?? digit);
}

function normalizeMathLabel(label: string): string {
  return formatDisplayLabel(label).replace(/_/g, " ").replace(/\s+/g, " ").trim();
}

function outputFallbackSymbol(source: string): string {
  const outputIndex = source.match(/output\s*([0-9]+)/i)?.[1];
  return outputIndex ? `${OUTPUT_FALLBACK_SYMBOL}${subscriptDigits(outputIndex)}` : OUTPUT_FALLBACK_SYMBOL;
}

function compactMathTerm(label: string, source = label): string {
  const display = normalizeMathLabel(label);
  const lowerDisplay = display.toLowerCase();
  const lowerSource = source.replace(/_/g, " ").toLowerCase();
  if (lowerDisplay.includes("loss") || lowerSource.includes("loss")) return LOSS_SYMBOL;

  const predictionSymbol = display.replace(/^prediction\s+/i, "").trim();
  if (predictionSymbol && predictionSymbol !== display) {
    return predictionSymbol.toLowerCase().includes("output")
      ? outputFallbackSymbol(`${source} ${predictionSymbol}`)
      : predictionSymbol;
  }

  if (lowerDisplay.includes("output") || lowerSource.includes("output")) {
    return outputFallbackSymbol(`${source} ${display}`);
  }
  return display;
}

function termLabelSource(term: string, termLabels?: Record<string, string>): string {
  const normalizedTerm = term.replace(/\s+/g, "_");
  return termLabels?.[normalizedTerm] ?? termLabels?.[term] ?? term;
}

function compactInfluenceTermLabel(term: string, termLabels?: Record<string, string>): string {
  const source = termLabelSource(term, termLabels);
  return compactMathTerm(source, source === term ? term : source);
}

export function formatFieldSelectLabel(label: string | null | undefined): string {
  if (!label) return "";
  return compactMathTerm(label);
}

export function formatInfluenceMatrixLabel(
  matrix: Pick<InfluenceMatrixManifest, "id" | "label" | "display_label" | "left_term" | "right_term">,
  termLabels?: Record<string, string>,
): string {
  const display = formatDisplayLabel(matrix.display_label || matrix.label || "")
    .replace(/^PINNfluence\s*(?:\/|:)\s*/i, "")
    .replace(/\s*\([^)]*\)\s*$/g, "")
    .trim();
  const displayParts = display.split(/\s*->\s*/);
  if (displayParts.length === 2 && displayParts[0] && displayParts[1]) {
    return `${compactInfluenceTermLabel(displayParts[0], termLabels)} ${TERM_ARROW} ${compactInfluenceTermLabel(displayParts[1], termLabels)}`;
  }
  if (matrix.id.includes("total_loss_output")) {
    return `${LOSS_SYMBOL} ${TERM_ARROW} ${compactInfluenceTermLabel(matrix.left_term, termLabels)}`;
  }
  if (matrix.id.includes("total_loss_total_loss")) return `${LOSS_SYMBOL} ${TERM_ARROW} ${LOSS_SYMBOL}`;
  return `${compactInfluenceTermLabel(matrix.left_term, termLabels)} ${TERM_ARROW} ${compactInfluenceTermLabel(matrix.right_term, termLabels)}`;
}
