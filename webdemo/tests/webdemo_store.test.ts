import { describe, expect, it } from "vitest";

import { initialState, reduceState } from "../src/state/store";

describe("app state reducer", () => {
  it("clears region state on reset and run changes", () => {
    const region = { minX: 0.1, maxX: 0.4, minY: 0.2, maxY: 0.6 };
    const selected = reduceState(
      reduceState(initialState, { type: "selectionMode", selectionMode: "region" }),
      { type: "regionSelection", region, candidateIndices: [1, 2] },
    );

    const reset = reduceState(selected, { type: "resetSelection" });
    expect(reset.selectionMode).toBe("point");
    expect(reset.selectedRegion).toBeNull();
    expect(reset.selectedRegionCandidateIndices).toEqual([]);

    const changedRun = reduceState(selected, { type: "run", runId: "next" });
    expect(changedRun.selectionMode).toBe("point");
    expect(changedRun.selectedRegion).toBeNull();
    expect(changedRun.selectedRegionCandidateIndices).toEqual([]);
  });
});
