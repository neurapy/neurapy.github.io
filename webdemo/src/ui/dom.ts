import type { InfluenceMatrixManifest } from "../types";

export interface DomRefs {
  runMeta: HTMLElement;
  problemSelect: HTMLSelectElement;
  qualityButtons: HTMLElement;
  resetButton: HTMLButtonElement;
  message: HTMLElement;
  fieldSelect: HTMLSelectElement;
  matrixSelect: HTMLSelectElement;
  signButtons: HTMLElement;
  backgroundButtons: HTMLElement;
  kControl: HTMLElement;
  kSlider: HTMLInputElement;
  kOutput: HTMLOutputElement;
  modelPanel: HTMLElement;
  modelActions: HTMLElement;
  modelMenuButton: HTMLButtonElement;
  modelMenu: HTMLElement;
  trainActions: HTMLElement;
  trainMenuButton: HTMLButtonElement;
  trainMenu: HTMLElement;
  selectedPointLabel: HTMLElement;
  selectedPoint: HTMLElement;
  selectedValueLabel: HTMLElement;
  selectedValue: HTMLElement;
  plotGrid: HTMLElement;
  mainTitle: HTMLElement;
  mainRange: HTMLElement;
  trainTitle: HTMLElement;
  trainRange: HTMLElement;
  mainCanvas: HTMLCanvasElement;
  trainCanvas: HTMLCanvasElement;
  mainSvg: SVGSVGElement;
  trainSvg: SVGSVGElement;
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
    runMeta: required("#runMeta", HTMLElement),
    problemSelect: required("#problemSelect", HTMLSelectElement),
    qualityButtons: required("#qualityButtons", HTMLElement),
    resetButton: required("#resetButton", HTMLButtonElement),
    message: required("#message", HTMLElement),
    fieldSelect: required("#fieldSelect", HTMLSelectElement),
    matrixSelect: required("#matrixSelect", HTMLSelectElement),
    signButtons: required("#signButtons", HTMLElement),
    backgroundButtons: required("#backgroundButtons", HTMLElement),
    kControl: required("#kControl", HTMLElement),
    kSlider: required("#kSlider", HTMLInputElement),
    kOutput: required("#kOutput", HTMLOutputElement),
    modelPanel: required("#modelPanel", HTMLElement),
    modelActions: required(".model-actions", HTMLElement),
    modelMenuButton: required("#modelMenuButton", HTMLButtonElement),
    modelMenu: required("#modelMenu", HTMLElement),
    trainActions: required(".train-actions", HTMLElement),
    trainMenuButton: required("#trainMenuButton", HTMLButtonElement),
    trainMenu: required("#trainMenu", HTMLElement),
    selectedPointLabel: required("#selectedPointLabel", HTMLElement),
    selectedPoint: required("#selectedPoint", HTMLElement),
    selectedValueLabel: required("#selectedValueLabel", HTMLElement),
    selectedValue: required("#selectedValue", HTMLElement),
    plotGrid: required("#plotGrid", HTMLElement),
    mainTitle: required("#mainTitle", HTMLElement),
    mainRange: required("#mainRange", HTMLElement),
    trainTitle: required("#trainTitle", HTMLElement),
    trainRange: required("#trainRange", HTMLElement),
    mainCanvas: required("#mainCanvas", HTMLCanvasElement),
    trainCanvas: required("#trainCanvas", HTMLCanvasElement),
    mainSvg: required("#mainSvg", SVGSVGElement),
    trainSvg: required("#trainSvg", SVGSVGElement),
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

function compactInfluenceTermLabel(term: string): string {
  const label = formatDisplayLabel(term).replace(/_/g, " ").toLowerCase();
  if (label.includes("loss")) return "loss";
  if (label.includes("output")) return "output";
  return label.trim();
}

export function formatInfluenceMatrixLabel(
  matrix: Pick<InfluenceMatrixManifest, "id" | "label" | "display_label" | "left_term" | "right_term">,
): string {
  const display = formatDisplayLabel(matrix.display_label || matrix.label || "")
    .replace(/^PINNfluence\s*(?:\/|:)\s*/i, "")
    .replace(/\s*\([^)]*\)\s*$/g, "")
    .trim();
  const displayParts = display.split(/\s*->\s*/);
  if (displayParts.length === 2 && displayParts[0] && displayParts[1]) {
    return `${compactInfluenceTermLabel(displayParts[0])} -> ${compactInfluenceTermLabel(displayParts[1])}`;
  }
  if (matrix.id.includes("total_loss_output")) return "loss -> output";
  if (matrix.id.includes("total_loss_total_loss")) return "loss -> loss";
  return `${compactInfluenceTermLabel(matrix.left_term)} -> ${compactInfluenceTermLabel(matrix.right_term)}`;
}
