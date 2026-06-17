import type { ArraySpec, DType, TypedArray } from "../types";

const DTYPE_CTORS = {
  float32: Float32Array,
  uint32: Uint32Array,
  uint16: Uint16Array,
  uint8: Uint8Array,
  int16: Int16Array,
} satisfies Record<DType, TypedArrayConstructor>;

type TypedArrayConstructor =
  | Float32ArrayConstructor
  | Uint32ArrayConstructor
  | Uint16ArrayConstructor
  | Uint8ArrayConstructor
  | Int16ArrayConstructor;

export function arrayElementCount(shape: number[]): number {
  return shape.reduce((acc, value) => acc * value, 1);
}

export function typedArrayFromBuffer(spec: ArraySpec, buffer: ArrayBuffer): TypedArray {
  const Ctor = DTYPE_CTORS[spec.dtype];
  const expected = arrayElementCount(spec.shape);
  if (buffer.byteLength !== expected * Ctor.BYTES_PER_ELEMENT) {
    throw new Error(
      `${spec.path}: expected ${expected * Ctor.BYTES_PER_ELEMENT} bytes for ${expected} ${spec.dtype} values, got ${buffer.byteLength}`,
    );
  }
  return new Ctor(buffer) as TypedArray;
}

export function assertDType(spec: ArraySpec, dtype: DType): void {
  if (spec.dtype !== dtype) {
    throw new Error(`${spec.path}: expected ${dtype}, got ${spec.dtype}`);
  }
}
