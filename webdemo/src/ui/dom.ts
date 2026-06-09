export interface DomRefs {
  runMeta: HTMLElement;
  runSelect: HTMLSelectElement;
  resetButton: HTMLButtonElement;
  message: HTMLElement;
  fieldSelect: HTMLSelectElement;
  matrixSelect: HTMLSelectElement;
  fieldKindButtons: HTMLElement;
  signButtons: HTMLElement;
  kSlider: HTMLInputElement;
  kOutput: HTMLOutputElement;
  summarySelect: HTMLSelectElement;
  summaryControl: HTMLElement;
  trainModeButtons: HTMLElement;
  modelMenuButton: HTMLButtonElement;
  modelMenu: HTMLElement;
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
    runSelect: required("#runSelect", HTMLSelectElement),
    resetButton: required("#resetButton", HTMLButtonElement),
    message: required("#message", HTMLElement),
    fieldSelect: required("#fieldSelect", HTMLSelectElement),
    matrixSelect: required("#matrixSelect", HTMLSelectElement),
    fieldKindButtons: required("#fieldKindButtons", HTMLElement),
    signButtons: required("#signButtons", HTMLElement),
    kSlider: required("#kSlider", HTMLInputElement),
    kOutput: required("#kOutput", HTMLOutputElement),
    summarySelect: required("#summarySelect", HTMLSelectElement),
    summaryControl: required("#summaryControl", HTMLElement),
    trainModeButtons: required("#trainModeButtons", HTMLElement),
    modelMenuButton: required("#modelMenuButton", HTMLButtonElement),
    modelMenu: required("#modelMenu", HTMLElement),
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
