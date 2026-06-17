import type { LinearEncoding } from "../types";

export function dequantizeUint16Linear(
  values: Uint16Array,
  encoding: LinearEncoding,
  mask?: Uint8Array,
): Float32Array {
  const decoded = new Float32Array(values.length);
  const missing = encoding.missing ?? 65535;
  const span = encoding.max - encoding.min || 1;
  for (let index = 0; index < values.length; index += 1) {
    const q = values[index];
    decoded[index] =
      q === missing || mask?.[index] === 0 ? Number.NaN : encoding.min + (q / 65534) * span;
  }
  return decoded;
}
