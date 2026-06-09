import { describe, expect, it } from "vitest";

import {
  containsViewportPoint,
  domainAspectRatio,
  fitScales,
  plotViewport,
  projectPointToViewport,
  unprojectPointFromViewport,
} from "../src/viz/geometry";

const EPS = 1e-9;

function close(actual: number, expected: number) {
  expect(Math.abs(actual - expected)).toBeLessThan(EPS);
}

function closePoint(actual: [number, number], expected: [number, number]) {
  close(actual[0], expected[0]);
  close(actual[1], expected[1]);
}

describe("D3 plot geometry", () => {
  it("keeps wide physical domains centered in square canvases", () => {
    const bounds = { minX: 0, maxX: 4, minY: 0, maxY: 2 };
    const viewport = plotViewport(bounds, 400, 400);

    close(viewport.x, 28);
    close(viewport.y, 114);
    close(viewport.width, 344);
    close(viewport.height, 172);
    close(viewport.width / viewport.height, domainAspectRatio(bounds));
  });

  it("keeps tall physical domains centered in wide canvases", () => {
    const bounds = { minX: 0, maxX: 2, minY: 0, maxY: 4 };
    const viewport = plotViewport(bounds, 600, 300);

    close(viewport.x, 239);
    close(viewport.y, 28);
    close(viewport.width, 122);
    close(viewport.height, 244);
    close(viewport.width / viewport.height, domainAspectRatio(bounds));
  });

  it("projects and unprojects with the fitted D3 scales", () => {
    const bounds = { minX: -2, maxX: 6, minY: 10, maxY: 14 };
    const { x, y, viewport } = fitScales(bounds, 800, 500);
    const point: [number, number] = [3.25, 11.75];

    const screen = projectPointToViewport(point[0], point[1], bounds, viewport);
    close(screen[0], x(point[0]));
    close(screen[1], y(point[1]));
    closePoint(unprojectPointFromViewport(screen[0], screen[1], bounds, viewport), point);
  });

  it("treats letterboxed margins as outside the plot", () => {
    const viewport = plotViewport({ minX: 0, maxX: 4, minY: 0, maxY: 2 }, 400, 400);

    expect(containsViewportPoint(200, 50, viewport)).toBe(false);
    expect(containsViewportPoint(20, 200, viewport)).toBe(false);
    expect(containsViewportPoint(200, 200, viewport)).toBe(true);
  });

  it("preserves zero as a valid physical bound", () => {
    close(domainAspectRatio({ minX: -1, maxX: 0, minY: -1, maxY: 1 }), 0.5);
  });
});
