import { describe, expect, it } from "vitest";

import {
  adaptivePlotLayoutCandidates,
  chooseAdaptivePlotLayout,
} from "../src/viz/layout";

function close(actual: number, expected: number, tolerance = 1e-9) {
  expect(Math.abs(actual - expected)).toBeLessThan(tolerance);
}

describe("adaptive plot layout", () => {
  it("chooses a row layout for wide containers", () => {
    const layout = chooseAdaptivePlotLayout({
      width: 1200,
      height: 600,
      gap: 20,
      headerHeight: 44,
      modelAspect: 1.5,
      trainAspect: 1,
    });

    expect(layout.orientation).toBe("row");
  });

  it("chooses a column layout for tall narrow containers", () => {
    const layout = chooseAdaptivePlotLayout({
      width: 480,
      height: 1100,
      gap: 20,
      headerHeight: 44,
      modelAspect: 1.5,
      trainAspect: 1,
    });

    expect(layout.orientation).toBe("column");
  });

  it("keeps the Model track first in both orientations", () => {
    const row = chooseAdaptivePlotLayout({
      width: 1000,
      height: 500,
      gap: 20,
      headerHeight: 44,
      modelAspect: 2,
      trainAspect: 1,
      padding: 0,
    });
    const column = chooseAdaptivePlotLayout({
      width: 420,
      height: 1000,
      gap: 20,
      headerHeight: 44,
      modelAspect: 2,
      trainAspect: 1,
      padding: 0,
    });

    expect(row.orientation).toBe("row");
    expect(row.modelTrackPx).toBeGreaterThan(row.trainTrackPx);
    expect(column.orientation).toBe("column");
    expect(column.modelTrackPx).toBeLessThan(column.trainTrackPx);
  });

  it("uses aspect-weighted tracks", () => {
    const [row, column] = adaptivePlotLayoutCandidates({
      width: 1000,
      height: 800,
      gap: 20,
      headerHeight: 50,
      modelAspect: 2,
      trainAspect: 1,
      padding: 0,
    });

    close(row.modelTrackPx / row.trainTrackPx, 2);
    close((column.modelTrackPx - 50) / (column.trainTrackPx - 50), 0.5);
  });

  it("selects the orientation with the larger fitted viewport area", () => {
    const input = {
      width: 480,
      height: 1100,
      gap: 20,
      headerHeight: 44,
      modelAspect: 1.5,
      trainAspect: 1,
    };
    const [row, column] = adaptivePlotLayoutCandidates(input);
    const layout = chooseAdaptivePlotLayout(input);

    expect(column.score).toBeGreaterThan(row.score);
    expect(layout.orientation).toBe("column");
    expect(layout.score).toBe(column.score);
  });
});
