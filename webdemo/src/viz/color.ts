import { scaleLinear } from "d3";

export function divergingColorScale(domain: [number, number]) {
  const maxAbs = Math.max(Math.abs(domain[0]), Math.abs(domain[1])) || 1;
  return scaleLinear<string>()
    .domain([-maxAbs, 0, maxAbs])
    .range(["#b2182b", "#f7f7f7", "#2166ac"])
    .clamp(true);
}
