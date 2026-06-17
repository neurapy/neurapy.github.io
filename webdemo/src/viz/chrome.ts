import type { Bounds } from "../types";
import { domainAspectRatio, plotViewport, type PlotInsets } from "./geometry";

export type PlotChromeMode = "regular" | "compact" | "tight";
export type PlotColorbarPlacement = "right" | "bottom";

export interface PlotChrome {
  mode: PlotChromeMode;
  padding: PlotInsets;
  colorbarPlacement: PlotColorbarPlacement;
}

export const REGULAR_RIGHT_COLORBAR_PADDING: PlotInsets = {
  top: 8,
  right: 46,
  bottom: 36,
  left: 38,
};

export const REGULAR_BOTTOM_COLORBAR_PADDING: PlotInsets = {
  top: 8,
  right: 14,
  bottom: 64,
  left: 38,
};

export const COMPACT_RIGHT_COLORBAR_PADDING: PlotInsets = {
  top: 6,
  right: 46,
  bottom: 28,
  left: 30,
};

export const COMPACT_BOTTOM_COLORBAR_PADDING: PlotInsets = {
  top: 6,
  right: 14,
  bottom: 56,
  left: 30,
};

export const TIGHT_RIGHT_COLORBAR_PADDING: PlotInsets = {
  top: 4,
  right: 42,
  bottom: 24,
  left: 24,
};

export const TIGHT_BOTTOM_COLORBAR_PADDING: PlotInsets = {
  top: 4,
  right: 10,
  bottom: 50,
  left: 24,
};

const COMPACT_PLOT_WIDTH_PX = 560;
const TIGHT_PLOT_WIDTH_PX = 340;

function positiveNumber(value: unknown, fallback: number): number {
  const number = Number(value);
  return Number.isFinite(number) && number > 0 ? number : fallback;
}

function viewportArea(width: number, height: number, aspect: number, padding: PlotInsets): number {
  const viewport = plotViewport(
    { minX: 0, maxX: aspect, minY: 0, maxY: 1 },
    width,
    height,
    padding,
  );
  return viewport.width * viewport.height;
}

function paddingCandidates(mode: PlotChromeMode): Array<{
  colorbarPlacement: PlotColorbarPlacement;
  padding: PlotInsets;
}> {
  if (mode === "tight") {
    return [
      { colorbarPlacement: "right", padding: TIGHT_RIGHT_COLORBAR_PADDING },
      { colorbarPlacement: "bottom", padding: TIGHT_BOTTOM_COLORBAR_PADDING },
    ];
  }
  if (mode === "compact") {
    return [
      { colorbarPlacement: "right", padding: COMPACT_RIGHT_COLORBAR_PADDING },
      { colorbarPlacement: "bottom", padding: COMPACT_BOTTOM_COLORBAR_PADDING },
    ];
  }
  return [
    { colorbarPlacement: "right", padding: REGULAR_RIGHT_COLORBAR_PADDING },
    { colorbarPlacement: "bottom", padding: REGULAR_BOTTOM_COLORBAR_PADDING },
  ];
}

export function plotChromeForSize(
  width: number,
  height: number,
  aspect = 1,
): PlotChrome {
  const safeWidth = positiveNumber(width, 1);
  const safeHeight = positiveNumber(height, 1);
  const safeAspect = positiveNumber(aspect, 1);
  const mode =
    safeWidth < TIGHT_PLOT_WIDTH_PX
      ? "tight"
      : safeWidth < COMPACT_PLOT_WIDTH_PX
        ? "compact"
        : "regular";
  const [right, bottom] = paddingCandidates(mode);
  const rightArea = viewportArea(safeWidth, safeHeight, safeAspect, right.padding);
  const bottomArea = viewportArea(safeWidth, safeHeight, safeAspect, bottom.padding);
  const candidate = bottomArea > rightArea ? bottom : right;
  return { mode, ...candidate };
}

export function plotChromeForBounds(
  width: number,
  height: number,
  bounds: Partial<Bounds> | null | undefined,
): PlotChrome {
  return plotChromeForSize(width, height, domainAspectRatio(bounds));
}
