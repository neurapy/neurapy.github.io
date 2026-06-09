import { describe, expect, it } from "vitest";

import { formatDisplayLabel } from "../src/ui/dom";

describe("UI display labels", () => {
  it("formats common manifest math labels without exposing raw LaTeX", () => {
    expect(formatDisplayLabel("Prediction $\\hat u$")).toBe("Prediction û");
    expect(formatDisplayLabel("Periodic BC ($x=2\\pi$)")).toBe("Periodic BC (x=2π)");
    expect(formatDisplayLabel("Operator BC ($\\frac{\\partial u}{\\partial t}$ at $t=0$)")).toBe(
      "Operator BC (∂u/∂t at t=0)",
    );
  });
});
