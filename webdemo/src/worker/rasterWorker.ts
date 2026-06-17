import { contours, interpolateTurbo } from "d3";
import type { LinearEncoding } from "../types";
import { dequantizeUint16Linear } from "../data/dequantize";

export interface RasterWorkerRequest {
  requestId: number;
  width: number;
  height: number;
  values: Uint16Array;
  mask: Uint8Array;
  encoding: LinearEncoding;
  displayDomain: [number, number];
}

export interface RasterWorkerResponse {
  requestId: number;
  width: number;
  height: number;
  rgba: Uint8ClampedArray;
  decoded: Float32Array;
  contourValues: number[];
  contourPaths: string[];
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

function colorize(decoded: Float32Array, request: RasterWorkerRequest): Uint8ClampedArray {
  const rgba = new Uint8ClampedArray(decoded.length * 4);
  const [min, max] = request.displayDomain;
  const span = max - min || 1;
  for (let index = 0; index < decoded.length; index += 1) {
    const value = decoded[index];
    const offset = index * 4;
    if (!Number.isFinite(value)) {
      rgba[offset + 3] = 0;
      continue;
    }
    const color = interpolateTurbo(clamp01((value - min) / span));
    const match = /rgb\((\d+), (\d+), (\d+)\)/.exec(color);
    rgba[offset] = match ? Number(match[1]) : 0;
    rgba[offset + 1] = match ? Number(match[2]) : 0;
    rgba[offset + 2] = match ? Number(match[3]) : 0;
    rgba[offset + 3] = 255;
  }
  return rgba;
}

function contourThresholds(domain: [number, number], count = 7): number[] {
  const [min, max] = domain;
  if (!Number.isFinite(min) || !Number.isFinite(max) || min === max) return [];
  const step = (max - min) / (count + 1);
  return Array.from({ length: count }, (_, index) => min + (index + 1) * step);
}

function pathFromRing(ring: number[][]): string {
  return ring
    .map(([x, y], index) => `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`)
    .join(" ")
    .concat(" Z");
}

function buildContourPaths(decoded: Float32Array, request: RasterWorkerRequest): [number[], string[]] {
  const thresholds = contourThresholds(request.displayDomain);
  if (!thresholds.length) return [[], []];
  const filled = Array.from(decoded, (value) => (Number.isFinite(value) ? value : request.displayDomain[0]));
  const generated = contours().size([request.width, request.height]).thresholds(thresholds)(filled);
  const values: number[] = [];
  const paths: string[] = [];
  for (const contour of generated) {
    const rings = contour.coordinates.flat(1);
    const path = rings.map(pathFromRing).join(" ");
    if (!path) continue;
    values.push(Number(contour.value));
    paths.push(path);
  }
  return [values, paths];
}

self.onmessage = (event: MessageEvent<RasterWorkerRequest>) => {
  const request = event.data;
  const decoded = dequantizeUint16Linear(request.values, request.encoding, request.mask);
  const rgba = colorize(decoded, request);
  const [contourValues, contourPaths] = buildContourPaths(decoded, request);
  const response: RasterWorkerResponse = {
    requestId: request.requestId,
    width: request.width,
    height: request.height,
    rgba,
    decoded,
    contourValues,
    contourPaths,
  };
  self.postMessage(response, [rgba.buffer, decoded.buffer]);
};
