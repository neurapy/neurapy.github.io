import { describe, expect, it } from "vitest";

import { formatDisplayLabel, formatInfluenceMatrixLabel } from "../src/ui/dom";

describe("UI display labels", () => {
  it("formats common manifest math labels without exposing raw LaTeX", () => {
    expect(formatDisplayLabel("Prediction $\\hat u$")).toBe("Prediction û");
    expect(formatDisplayLabel("Periodic BC ($x=2\\pi$)")).toBe("Periodic BC (x=2π)");
    expect(formatDisplayLabel("Operator BC ($\\frac{\\partial u}{\\partial t}$ at $t=0$)")).toBe(
      "Operator BC (∂u/∂t at t=0)",
    );
  });

  it("compacts PINNfluence matrix labels to term arrows", () => {
    expect(formatInfluenceMatrixLabel({
      id: "influences_total_loss_output_0",
      label: "PINNfluence: output_0 -> total_loss",
      display_label: "PINNfluence / total loss -> output 0",
      left_term: "output_0",
      right_term: "total_loss",
    })).toBe("loss -> output");
    expect(formatInfluenceMatrixLabel({
      id: "influences_total_loss_total_loss",
      label: "PINNfluence: total_loss -> total_loss",
      display_label: "PINNfluence / total loss -> total loss",
      left_term: "total_loss",
      right_term: "total_loss",
    })).toBe("loss -> loss");
  });
});
