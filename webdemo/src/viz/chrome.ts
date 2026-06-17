import type { Bounds } from "../types";
import { domainAspectRatio, type PlotInsets } from "./geometry";

export type PlotChromeMode = "regular" | "compact" | "tight";

export interface PlotChrome {
  mode: PlotChromeMode;
  padding: PlotInsets;
  showColorbar: boolean;
}

export const REGULAR_PLOT_PADDING: PlotInsets = {
  top: 8,
  right: 46,
  bottom: 36,
  left: 38,
};

export const COMPACT_PLOT_PADDING: PlotInsets = {
  top: 6,
  right: 14,
  bottom: 28,
  left: 30,
};

export const TIGHT_PLOT_PADDING: PlotInsets = {
  top: 4,
  right: 10,
  bottom: 24,
  left: 24,
};

const COMPACT_PLOT_WIDTH_PX = 560;
const TIGHT_PLOT_WIDTH_PX = 340;

function positiveCssPixel(value: unknown): number {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : 1;
}

export function plotChromeForSize(
  width: number,
  _height: number,
  _aspect = 1,
): PlotChrome {
  const safeWidth = positiveCssPixel(width);
  if (safeWidth < TIGHT_PLOT_WIDTH_PX) {
    return { mode: "tight", padding: TIGHT_PLOT_PADDING, showColorbar: false };
  }
  if (safeWidth < COMPACT_PLOT_WIDTH_PX) {
    return { mode: "compact", padding: COMPACT_PLOT_PADDING, showColorbar: false };
  }
  return { mode: "regular", padding: REGULAR_PLOT_PADDING, showColorbar: true };
}

export function plotChromeForBounds(
  width: number,
  height: number,
  bounds: Partial<Bounds> | null | undefined,
): PlotChrome {
  return plotChromeForSize(width, height, domainAspectRatio(bounds));
}
