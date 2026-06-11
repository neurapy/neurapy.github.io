import { describe, expect, it } from "vitest";

import {
  referenceAspectPlotArea,
  PLOT_VISUAL_SCALE_MULTIPLIER,
  REFERENCE_PLOT_ASPECT_RATIO,
  REFERENCE_PLOT_SHORT_SIDE_PX,
  plotVisualScale,
  scaledPlotPx,
} from "../src/viz/scale";

describe("plot visual scaling", () => {
  it("uses the multiplier for a 2:1 plot at the reference short side", () => {
    expect(plotVisualScale({
      width: REFERENCE_PLOT_SHORT_SIDE_PX * REFERENCE_PLOT_ASPECT_RATIO,
      height: REFERENCE_PLOT_SHORT_SIDE_PX,
    })).toBe(
      PLOT_VISUAL_SCALE_MULTIPLIER,
    );
  });

  it("normalizes non-2:1 plots to the 2:1-equivalent short side", () => {
    expect(plotVisualScale({ width: 2400, height: 1200 })).toBe(
      2 * PLOT_VISUAL_SCALE_MULTIPLIER,
    );
    expect(plotVisualScale({ width: 1200, height: 1200 })).toBe(
      PLOT_VISUAL_SCALE_MULTIPLIER,
    );
    expect(plotVisualScale({ width: 600, height: 1200 })).toBe(
      PLOT_VISUAL_SCALE_MULTIPLIER,
    );
    expect(scaledPlotPx(7, { width: 300, height: 900 })).toBeCloseTo(10.5);
  });

  it("uses the same 2:1-equivalent area for automatic point sizing", () => {
    expect(referenceAspectPlotArea({ width: 1200, height: 600 })).toBe(720000);
    expect(referenceAspectPlotArea({ width: 1200, height: 1200 })).toBe(720000);
    expect(referenceAspectPlotArea({ width: 600, height: 1200 })).toBe(
      referenceAspectPlotArea({ width: 1200, height: 600 }),
    );
  });

  it("keeps tiny or invalid viewports finite and non-negative", () => {
    expect(plotVisualScale({ width: 0, height: Number.NaN })).toBeGreaterThan(0);
    expect(Number.isFinite(plotVisualScale({ width: 0, height: Number.NaN }))).toBe(true);
    expect(Number.isFinite(plotVisualScale({ width: Infinity, height: Infinity }))).toBe(true);
    expect(scaledPlotPx(-4, { width: 0, height: Number.NaN })).toBe(0);
  });
});
