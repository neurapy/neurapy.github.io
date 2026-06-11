import { describe, expect, it } from "vitest";

import {
  PLOT_VISUAL_SCALE_MULTIPLIER,
  REFERENCE_PLOT_SHORT_SIDE_PX,
  plotVisualScale,
  scaledPlotPx,
} from "../src/viz/scale";

describe("plot visual scaling", () => {
  it("uses the multiplier at the reference short side", () => {
    expect(plotVisualScale({ width: 900, height: REFERENCE_PLOT_SHORT_SIDE_PX })).toBe(
      PLOT_VISUAL_SCALE_MULTIPLIER,
    );
  });

  it("scales proportionally with the viewport short side", () => {
    expect(plotVisualScale({ width: 1200, height: 1600 })).toBe(
      2 * PLOT_VISUAL_SCALE_MULTIPLIER,
    );
    expect(scaledPlotPx(7, { width: 300, height: 900 })).toBe(
      3.5 * PLOT_VISUAL_SCALE_MULTIPLIER,
    );
  });

  it("keeps tiny or invalid viewports finite and non-negative", () => {
    expect(plotVisualScale({ width: 0, height: Number.NaN })).toBeGreaterThan(0);
    expect(Number.isFinite(plotVisualScale({ width: 0, height: Number.NaN }))).toBe(true);
    expect(Number.isFinite(plotVisualScale({ width: Infinity, height: Infinity }))).toBe(true);
    expect(scaledPlotPx(-4, { width: 0, height: Number.NaN })).toBe(0);
  });
});
