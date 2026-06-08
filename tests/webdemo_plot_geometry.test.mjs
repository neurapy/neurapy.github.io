import assert from "node:assert/strict";
import test from "node:test";

import {
  containsViewportPoint,
  domainAspectRatio,
  plotViewport,
  projectPointToViewport,
  unprojectPointFromViewport,
} from "../webdemo/plotGeometry.mjs";

const EPS = 1e-9;

function close(actual, expected, message) {
  assert.ok(Math.abs(actual - expected) < EPS, `${message}: expected ${expected}, got ${actual}`);
}

function closePoint(actual, expected, message) {
  close(actual[0], expected[0], `${message} x`);
  close(actual[1], expected[1], `${message} y`);
}

test("wide domains keep physical ratio in a square canvas", () => {
  const bounds = { minX: 0, maxX: 4, minY: 0, maxY: 2 };
  const viewport = plotViewport(bounds, 400, 400);

  close(viewport.x, 28, "viewport x");
  close(viewport.y, 114, "viewport y");
  close(viewport.width, 344, "viewport width");
  close(viewport.height, 172, "viewport height");
  close(viewport.width / viewport.height, domainAspectRatio(bounds), "viewport ratio");
});

test("tall domains are centered horizontally inside wide canvases", () => {
  const bounds = { minX: 0, maxX: 2, minY: 0, maxY: 4 };
  const viewport = plotViewport(bounds, 600, 300);

  close(viewport.x, 239, "viewport x");
  close(viewport.y, 28, "viewport y");
  close(viewport.width, 122, "viewport width");
  close(viewport.height, 244, "viewport height");
  close(viewport.width / viewport.height, domainAspectRatio(bounds), "viewport ratio");
});

test("square domains fill a square padded canvas", () => {
  const bounds = { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  const viewport = plotViewport(bounds, 500, 500);

  close(viewport.x, 28, "viewport x");
  close(viewport.y, 28, "viewport y");
  close(viewport.width, 444, "viewport width");
  close(viewport.height, 444, "viewport height");
});

test("project and unproject are inverses inside the viewport", () => {
  const bounds = { minX: -2, maxX: 6, minY: 10, maxY: 14 };
  const viewport = plotViewport(bounds, 800, 500);
  const point = [3.25, 11.75];

  const screen = projectPointToViewport(point[0], point[1], bounds, viewport);
  const domain = unprojectPointFromViewport(screen[0], screen[1], bounds, viewport);

  closePoint(domain, point, "round trip");
});

test("letterboxed margins are outside the interactive plot area", () => {
  const bounds = { minX: 0, maxX: 4, minY: 0, maxY: 2 };
  const viewport = plotViewport(bounds, 400, 400);

  assert.equal(containsViewportPoint(200, 50, viewport), false);
  assert.equal(containsViewportPoint(20, 200, viewport), false);
  assert.equal(containsViewportPoint(200, 200, viewport), true);
});

test("zero is preserved as a valid physical bound", () => {
  const bounds = { minX: -1, maxX: 0, minY: -1, maxY: 1 };

  close(domainAspectRatio(bounds), 0.5, "domain ratio");
});
