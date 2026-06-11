import {
  interpolateCividis,
  interpolateTurbo,
  scaleDiverging,
  scaleLinear,
  scaleSequential,
} from "d3";

export function finiteExtent(values: ArrayLike<number>): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (!Number.isFinite(value)) continue;
    min = Math.min(min, value);
    max = Math.max(max, value);
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
  if (min === max) return [min - 1, max + 1];
  return [min, max];
}

export function symmetricExtent(values: ArrayLike<number>): [number, number] {
  let maxAbs = 0;
  for (let index = 0; index < values.length; index += 1) {
    const value = values[index];
    if (Number.isFinite(value)) maxAbs = Math.max(maxAbs, Math.abs(value));
  }
  return maxAbs > 0 ? [-maxAbs, maxAbs] : [-1, 1];
}

export function sequentialColorScale(domain: [number, number]) {
  return scaleSequential(domain, interpolateCividis);
}

export function fieldColorScale(domain: [number, number]) {
  return scaleSequential(domain, interpolateTurbo);
}

export function divergingColorScale(domain: [number, number]) {
  const maxAbs = Math.max(Math.abs(domain[0]), Math.abs(domain[1])) || 1;
  return scaleLinear<string>()
    .domain([-maxAbs, 0, maxAbs])
    .range(["#b2182b", "#f7f7f7", "#2166ac"])
    .clamp(true);
}
