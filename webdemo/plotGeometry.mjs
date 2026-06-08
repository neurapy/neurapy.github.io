export const DEFAULT_PLOT_PADDING = 28;

function positiveSpan(a, b) {
  const span = Number(b) - Number(a);
  return Number.isFinite(span) && span > 0 ? span : 1;
}

function finiteNumber(value, fallback) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function normalizedBounds(bounds) {
  return {
    minX: finiteNumber(bounds?.minX, 0),
    maxX: finiteNumber(bounds?.maxX, 1),
    minY: finiteNumber(bounds?.minY, 0),
    maxY: finiteNumber(bounds?.maxY, 1),
  };
}

export function domainAspectRatio(bounds) {
  const safeBounds = normalizedBounds(bounds);
  return positiveSpan(safeBounds.minX, safeBounds.maxX) / positiveSpan(safeBounds.minY, safeBounds.maxY);
}

export function plotViewport(
  bounds,
  canvasWidth,
  canvasHeight,
  padding = DEFAULT_PLOT_PADDING,
) {
  const width = Math.max(1, Number(canvasWidth) || 1);
  const height = Math.max(1, Number(canvasHeight) || 1);
  const inset = Math.min(
    Math.max(0, Number(padding) || 0),
    Math.max(0, (width - 1) / 2),
    Math.max(0, (height - 1) / 2),
  );
  const availableWidth = Math.max(1, width - inset * 2);
  const availableHeight = Math.max(1, height - inset * 2);
  const targetRatio = domainAspectRatio(bounds);
  const availableRatio = availableWidth / availableHeight;

  let viewportWidth = availableWidth;
  let viewportHeight = availableHeight;
  if (availableRatio > targetRatio) {
    viewportWidth = availableHeight * targetRatio;
  } else {
    viewportHeight = availableWidth / targetRatio;
  }

  const x = inset + (availableWidth - viewportWidth) / 2;
  const y = inset + (availableHeight - viewportHeight) / 2;
  return {
    x,
    y,
    width: viewportWidth,
    height: viewportHeight,
    right: x + viewportWidth,
    bottom: y + viewportHeight,
  };
}

export function projectPointToViewport(x, y, bounds, viewport) {
  const safeBounds = normalizedBounds(bounds);
  const spanX = positiveSpan(safeBounds.minX, safeBounds.maxX);
  const spanY = positiveSpan(safeBounds.minY, safeBounds.maxY);
  return [
    viewport.x + ((x - safeBounds.minX) / spanX) * viewport.width,
    viewport.y + viewport.height - ((y - safeBounds.minY) / spanY) * viewport.height,
  ];
}

export function containsViewportPoint(sx, sy, viewport, tolerance = 0) {
  return (
    sx >= viewport.x - tolerance &&
    sx <= viewport.right + tolerance &&
    sy >= viewport.y - tolerance &&
    sy <= viewport.bottom + tolerance
  );
}

export function unprojectPointFromViewport(sx, sy, bounds, viewport) {
  const safeBounds = normalizedBounds(bounds);
  const spanX = positiveSpan(safeBounds.minX, safeBounds.maxX);
  const spanY = positiveSpan(safeBounds.minY, safeBounds.maxY);
  const nx = Math.max(0, Math.min(1, (sx - viewport.x) / Math.max(1, viewport.width)));
  const ny = Math.max(0, Math.min(1, (viewport.y + viewport.height - sy) / Math.max(1, viewport.height)));
  return [safeBounds.minX + nx * spanX, safeBounds.minY + ny * spanY];
}
