import type { PlotViewport } from "../types";

export const REFERENCE_PLOT_SHORT_SIDE_PX = 600;
export const PLOT_VISUAL_SCALE_MULTIPLIER = 2;

function positiveCssPixel(value: unknown): number {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : 1;
}

export function plotVisualScale(viewport: Pick<PlotViewport, "width" | "height">): number {
  const width = positiveCssPixel(viewport.width);
  const height = positiveCssPixel(viewport.height);
  return PLOT_VISUAL_SCALE_MULTIPLIER * Math.min(width, height) / REFERENCE_PLOT_SHORT_SIDE_PX;
}

export function scaledPlotPx(
  value: number,
  viewport: Pick<PlotViewport, "width" | "height">,
): number {
  const safeValue = Number.isFinite(value) ? value : 0;
  return Math.max(0, safeValue * plotVisualScale(viewport));
}
