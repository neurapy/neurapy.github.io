import { describe, expect, it } from "vitest";

import { MAX_TOP_K, initialState, reduceState } from "../src/state/store";

describe("app state reducer", () => {
  it("defaults to the Well-Trained model quality", () => {
    expect(initialState.modelQuality).toBe("good");
    expect(initialState.appView).toBe("playground");
  });

  it("switches among views without resetting selection state", () => {
    const selected = reduceState(initialState, {
      type: "selection",
      candidateIndex: 4,
      trainIndex: 7,
      coord: [0.2, 0.8],
    });
    const indicators = reduceState(selected, { type: "view", appView: "indicators" });
    const explainer = reduceState(indicators, { type: "view", appView: "explainer" });
    const playground = reduceState(explainer, { type: "view", appView: "playground" });

    expect(indicators.appView).toBe("indicators");
    expect(explainer.appView).toBe("explainer");
    expect(indicators.selectedCandidateIndex).toBe(4);
    expect(indicators.selectedTrainIndex).toBe(7);
    expect(indicators.selectedCoord).toEqual([0.2, 0.8]);
    expect(playground.appView).toBe("playground");
  });

  it("switches the influence background explicitly", () => {
    expect(initialState.backgroundMode).toBe("points");

    const smooth = reduceState(initialState, {
      type: "backgroundMode",
      backgroundMode: "smooth",
    });
    const cell = reduceState(smooth, {
      type: "backgroundMode",
      backgroundMode: "cell",
    });

    expect(smooth.backgroundMode).toBe("smooth");
    expect(cell.backgroundMode).toBe("cell");
  });

  it("clamps top-k to the exported control range", () => {
    expect(reduceState(initialState, { type: "k", k: -4 }).k).toBe(0);
    expect(reduceState(initialState, { type: "k", k: 0 }).k).toBe(0);
    expect(reduceState(initialState, { type: "k", k: 25.9 }).k).toBe(25);
    expect(reduceState(initialState, { type: "k", k: MAX_TOP_K + 20 }).k).toBe(MAX_TOP_K);

    const unchanged = reduceState({ ...initialState, k: 17 }, { type: "k", k: Number.NaN });
    expect(unchanged.k).toBe(17);
  });

  it("preserves loaded-manifest state on problem and quality changes", () => {
    const region = { minX: 0.1, maxX: 0.4, minY: 0.2, maxY: 0.6 };
    const selected = reduceState(
      reduceState(
        reduceState(
          reduceState(
            reduceState(
              reduceState(initialState, { type: "field", fieldId: "pred_output_0" }),
              { type: "matrix", matrixId: "m0" },
            ),
            { type: "sign", sign: "pos" },
          ),
          { type: "k", k: 17 },
        ),
        { type: "backgroundMode", backgroundMode: "cell" },
      ),
      { type: "regionSelection", region, candidateIndices: [1, 2] },
    );

    const reset = reduceState(selected, { type: "resetSelection" });
    expect(reset.fieldId).toBe("pred_output_0");
    expect(reset.matrixId).toBe("m0");
    expect(reset.selectionMode).toBe("point");
    expect(reset.selectedRegion).toBeNull();
    expect(reset.selectedRegionCandidateIndices).toEqual([]);

    const changedProblem = reduceState(selected, { type: "problem", problem: "next" });
    expect(changedProblem.problem).toBe("next");
    expect(changedProblem.fieldId).toBe("pred_output_0");
    expect(changedProblem.matrixId).toBe("m0");
    expect(changedProblem.sign).toBe("pos");
    expect(changedProblem.k).toBe(17);
    expect(changedProblem.backgroundMode).toBe("cell");
    expect(changedProblem.selectionMode).toBe("region");
    expect(changedProblem.selectedRegion).toEqual(region);
    expect(changedProblem.selectedRegionCandidateIndices).toEqual([1, 2]);

    const changedQuality = reduceState(selected, { type: "modelQuality", modelQuality: "bad" });
    expect(changedQuality.modelQuality).toBe("bad");
    expect(changedQuality.fieldId).toBe("pred_output_0");
    expect(changedQuality.matrixId).toBe("m0");
    expect(changedQuality.sign).toBe("pos");
    expect(changedQuality.k).toBe(17);
    expect(changedQuality.backgroundMode).toBe("cell");
    expect(changedQuality.selectionMode).toBe("region");
    expect(changedQuality.selectedRegion).toEqual(region);
    expect(changedQuality.selectedRegionCandidateIndices).toEqual([1, 2]);
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
