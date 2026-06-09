import { describe, expect, it } from "vitest";

import { initialState, reduceState } from "../src/state/store";

describe("app state reducer", () => {
  it("defaults the Train plot to Local and switches mode explicitly", () => {
    expect(initialState.trainPlotMode).toBe("local");

    const state = reduceState(initialState, { type: "trainPlotMode", trainPlotMode: "global" });

    expect(state.trainPlotMode).toBe("global");
  });

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

  it("lets gesture actions switch between point and region selection", () => {
    const region = { minX: 0.1, maxX: 0.4, minY: 0.2, maxY: 0.6 };
    const regional = reduceState(initialState, {
      type: "regionSelection",
      region,
      candidateIndices: [1, 2],
    });

    expect(regional.selectionMode).toBe("region");
    expect(regional.selectedRegion).toEqual(region);
    expect(regional.selectedRegionCandidateIndices).toEqual([1, 2]);

    const point = reduceState(regional, {
      type: "selection",
      candidateIndex: 3,
      trainIndex: 4,
      coord: [0.25, 0.75],
    });

    expect(point.selectionMode).toBe("point");
    expect(point.selectedCandidateIndex).toBe(3);
    expect(point.selectedTrainIndex).toBe(4);
    expect(point.selectedCoord).toEqual([0.25, 0.75]);
    expect(point.selectedRegion).toBeNull();
    expect(point.selectedRegionCandidateIndices).toEqual([]);
  });
});
