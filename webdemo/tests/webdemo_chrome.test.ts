import { describe, expect, it } from "vitest";

import {
  COMPACT_PLOT_PADDING,
  REGULAR_PLOT_PADDING,
  TIGHT_PLOT_PADDING,
  plotChromeForSize,
} from "../src/viz/chrome";

describe("plot chrome", () => {
  it("keeps colorbars and regular insets on wide plot bodies", () => {
    expect(plotChromeForSize(720, 420, 2)).toEqual({
      mode: "regular",
      padding: REGULAR_PLOT_PADDING,
      showColorbar: true,
    });
  });

  it("uses compact insets and hides colorbars on mobile-sized plot bodies", () => {
    expect(plotChromeForSize(390, 289, 2)).toEqual({
      mode: "compact",
      padding: COMPACT_PLOT_PADDING,
      showColorbar: false,
    });
  });

  it("uses tight insets for very narrow plot bodies", () => {
    expect(plotChromeForSize(320, 220, 1)).toEqual({
      mode: "tight",
      padding: TIGHT_PLOT_PADDING,
      showColorbar: false,
    });
  });
});
