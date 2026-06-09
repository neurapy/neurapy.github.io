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
  mobileTabs: HTMLElement;
  selectedPoint: HTMLElement;
  selectedValue: HTMLElement;
  trainCount: HTMLElement;
  candidateCount: HTMLElement;
  mainTitle: HTMLElement;
  mainRange: HTMLElement;
  influenceRange: HTMLElement;
  globalRange: HTMLElement;
  mainCanvas: HTMLCanvasElement;
  influenceCanvas: HTMLCanvasElement;
  globalCanvas: HTMLCanvasElement;
  mainSvg: SVGSVGElement;
  influenceSvg: SVGSVGElement;
  globalSvg: SVGSVGElement;
  localPanel: HTMLElement;
  globalPanel: HTMLElement;
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
    mobileTabs: required("#mobileTabs", HTMLElement),
    selectedPoint: required("#selectedPoint", HTMLElement),
    selectedValue: required("#selectedValue", HTMLElement),
    trainCount: required("#trainCount", HTMLElement),
    candidateCount: required("#candidateCount", HTMLElement),
    mainTitle: required("#mainTitle", HTMLElement),
    mainRange: required("#mainRange", HTMLElement),
    influenceRange: required("#influenceRange", HTMLElement),
    globalRange: required("#globalRange", HTMLElement),
    mainCanvas: required("#mainCanvas", HTMLCanvasElement),
    influenceCanvas: required("#influenceCanvas", HTMLCanvasElement),
    globalCanvas: required("#globalCanvas", HTMLCanvasElement),
    mainSvg: required("#mainSvg", SVGSVGElement),
    influenceSvg: required("#influenceSvg", SVGSVGElement),
    globalSvg: required("#globalSvg", SVGSVGElement),
    localPanel: required("#localPanel", HTMLElement),
    globalPanel: required("#globalPanel", HTMLElement),
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
