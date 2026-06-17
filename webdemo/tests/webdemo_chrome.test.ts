import { describe, expect, it } from "vitest";

import {
  COMPACT_BOTTOM_COLORBAR_PADDING,
  REGULAR_BOTTOM_COLORBAR_PADDING,
  REGULAR_RIGHT_COLORBAR_PADDING,
  TIGHT_RIGHT_COLORBAR_PADDING,
  plotChromeForSize,
} from "../src/viz/chrome";

describe("plot chrome", () => {
  it("uses the bottom colorbar when vertical gutter preserves a larger plot", () => {
    expect(plotChromeForSize(720, 420, 2)).toEqual({
      mode: "regular",
      padding: REGULAR_BOTTOM_COLORBAR_PADDING,
      colorbarPlacement: "bottom",
    });
  });

  it("uses the right colorbar when height is the limiting dimension", () => {
    expect(plotChromeForSize(720, 260, 2)).toEqual({
      mode: "regular",
      padding: REGULAR_RIGHT_COLORBAR_PADDING,
      colorbarPlacement: "right",
    });
  });

  it("uses compact bottom colorbars on mobile-sized plot bodies", () => {
    expect(plotChromeForSize(390, 289, 2)).toEqual({
      mode: "compact",
      padding: COMPACT_BOTTOM_COLORBAR_PADDING,
      colorbarPlacement: "bottom",
    });
  });

  it("uses tight right colorbars when that preserves more area", () => {
    expect(plotChromeForSize(320, 220, 1)).toEqual({
      mode: "tight",
      padding: TIGHT_RIGHT_COLORBAR_PADDING,
      colorbarPlacement: "right",
    });
  });
});
