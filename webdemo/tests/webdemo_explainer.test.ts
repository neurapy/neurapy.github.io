import { describe, expect, it } from "vitest";

import { EXPLAINER_TOKENS } from "../src/ui/explainer";

describe("explainer token metadata", () => {
  it("defines stable, complete formula tokens", () => {
    const ids = EXPLAINER_TOKENS.map((token) => token.id);
    expect(new Set(ids).size).toBe(ids.length);
    expect(ids).toEqual(
      expect.arrayContaining([
        "score",
        "loss",
        "quantity",
        "train-point",
        "test-point",
        "sign",
        "grad-f",
        "hessian",
        "grad-loss",
        "loss-fraction",
        "loss-normalizer",
        "cancellation",
      ]),
    );

    for (const token of EXPLAINER_TOKENS) {
      expect(token.title.trim().length).toBeGreaterThan(0);
      expect(token.body.trim().length).toBeGreaterThan(24);
    }
  });
});
