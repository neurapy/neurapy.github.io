import type {
  ArraySpec,
  InfluenceAggregate,
  InfluenceChunkSpec,
  InfluenceMatrixManifest,
  InfluenceRow,
  InfluenceSign,
  PointArrays,
  Priority,
  RasterData,
  RasterFieldManifest,
  RunManifest,
  SummaryName,
  TypedArray,
} from "../types";
import { assertDType, typedArrayFromBuffer } from "./dtypes";
import { LruCache } from "./cache";
import { PriorityLoader } from "./loader";

export class DataRepository {
  private readonly loader: PriorityLoader;
  private readonly cache: LruCache<TypedArray>;

  constructor(
    private readonly manifestUrl: URL,
    private readonly manifest: RunManifest,
    cacheBytes = 128 * 1024 * 1024,
    loader = new PriorityLoader(),
  ) {
    this.loader = loader;
    this.cache = new LruCache<TypedArray>(cacheBytes);
  }

  abortBackground(): void {
    this.loader.abortBackground();
  }

  hasArray(spec: ArraySpec): boolean {
    return this.cache.has(this.cacheKey(spec));
  }

  async prefetchArray(spec: ArraySpec): Promise<void> {
    await this.loadArray(spec, "background");
  }

  async loadArray<T extends TypedArray>(
    spec: ArraySpec,
    priority: Priority = "foreground",
    signal?: AbortSignal,
  ): Promise<T> {
    const url = this.arrayUrl(spec);
    const key = this.cacheKey(spec);
    const cached = this.cache.get(key);
    if (cached) return cached as T;
    const buffer = await this.loader.load(url, priority, signal);
    const array = typedArrayFromBuffer(spec, buffer);
    this.cache.set(key, array, buffer.byteLength);
    return array as T;
  }

  async loadPointArrays(signal?: AbortSignal): Promise<PointArrays> {
    const arrays = this.manifest.arrays;
    assertDType(arrays.candidate_points, "float32");
    assertDType(arrays.train_points, "float32");
    assertDType(arrays.train_kind, "uint8");
    assertDType(arrays.train_bc_id, "int16");
    const [candidatePoints, trainPoints, trainKind, trainBcId] = await Promise.all([
      this.loadArray<Float32Array>(arrays.candidate_points, "foreground", signal),
      this.loadArray<Float32Array>(arrays.train_points, "foreground", signal),
      this.loadArray<Uint8Array>(arrays.train_kind, "foreground", signal),
      this.loadArray<Int16Array>(arrays.train_bc_id, "foreground", signal),
    ]);
    return {
      candidate_points: candidatePoints,
      train_points: trainPoints,
      train_kind: trainKind,
      train_bc_id: trainBcId,
    };
  }

  async loadRaster(fieldId: string, priority: Priority = "foreground"): Promise<RasterData> {
    const field = this.manifest.fields[fieldId];
    const raster = this.manifest.field_raster;
    if (!field || !raster) {
      throw new Error(`Raster field ${fieldId} is unavailable`);
    }
    assertDType(field.raster, "uint16");
    assertDType(raster.mask, "uint8");
    const [values, mask] = await Promise.all([
      this.loadArray<Uint16Array>(field.raster, priority),
      this.loadArray<Uint8Array>(raster.mask, priority),
    ]);
    return {
      fieldId,
      values,
      mask,
      encoding: field.encoding,
      displayDomain: field.display_domain,
      width: raster.width,
      height: raster.height,
    };
  }

  async loadSummary(
    matrix: InfluenceMatrixManifest,
    name: SummaryName,
    priority: Priority = "foreground",
  ): Promise<Float32Array> {
    const spec = matrix.summary[name];
    if (!spec) throw new Error(`${matrix.id}: summary ${name} is unavailable`);
    assertDType(spec, "float32");
    return this.loadArray<Float32Array>(spec, priority);
  }

  async loadInfluenceRow(
    matrix: InfluenceMatrixManifest,
    sign: InfluenceSign,
    rowIndex: number,
    priority: Priority = "foreground",
  ): Promise<InfluenceRow> {
    const group = matrix.top_chunks[sign];
    if (!group) throw new Error(`${matrix.id}: ${sign} chunks are unavailable`);
    const chunk = group.chunks.find(
      (candidate) =>
        rowIndex >= candidate.row_start && rowIndex < candidate.row_start + candidate.row_count,
    );
    if (!chunk) {
      throw new Error(`${matrix.id}: row ${rowIndex} is outside exported chunks`);
    }
    const localRow = rowIndex - chunk.row_start;
    const [indicesArray, rawArray] = await Promise.all([
      this.loadArray<Uint16Array | Uint32Array>(chunk.indices, priority),
      this.loadArray<Int16Array>(chunk.values, priority),
    ]);
    const offset = localRow * chunk.k;
    const indices = indicesArray.subarray(offset, offset + chunk.k) as Uint16Array | Uint32Array;
    const rawValues = rawArray.subarray(offset, offset + chunk.k);
    const values = dequantizeInt16Values(rawValues, chunk.value_scale);
    return {
      rowIndex,
      indices,
      values,
      rawValues,
      valueScale: chunk.value_scale,
    };
  }

  async loadInfluenceAggregate(
    matrix: InfluenceMatrixManifest,
    sign: InfluenceSign,
    rowIndices: number[],
    priority: Priority = "foreground",
  ): Promise<InfluenceAggregate> {
    const group = matrix.top_chunks[sign];
    if (!group) throw new Error(`${matrix.id}: ${sign} chunks are unavailable`);
    const rowsByChunk = new Map<InfluenceChunkSpec, number[]>();
    const validRows: number[] = [];
    for (const rowIndex of rowIndices) {
      if (!Number.isFinite(rowIndex)) continue;
      const row = Math.trunc(rowIndex);
      if (row < 0 || row >= matrix.row_count) continue;
      const chunk = group.chunks.find(
        (candidate) => row >= candidate.row_start && row < candidate.row_start + candidate.row_count,
      );
      if (!chunk) continue;
      validRows.push(row);
      const rows = rowsByChunk.get(chunk);
      if (rows) {
        rows.push(row);
      } else {
        rowsByChunk.set(chunk, [row]);
      }
    }

    const sums = new Map<number, number>();
    await Promise.all(
      Array.from(rowsByChunk.entries()).map(async ([chunk, rows]) => {
        const [indicesArray, rawArray] = await Promise.all([
          this.loadArray<Uint16Array | Uint32Array>(chunk.indices, priority),
          this.loadArray<Int16Array>(chunk.values, priority),
        ]);
        for (const row of rows) {
          const offset = (row - chunk.row_start) * chunk.k;
          for (let index = 0; index < chunk.k; index += 1) {
            const trainIndex = indicesArray[offset + index];
            const value = rawArray[offset + index] * chunk.value_scale;
            const contribution = aggregateContribution(value, sign);
            if (!contribution) continue;
            sums.set(trainIndex, (sums.get(trainIndex) ?? 0) + contribution);
          }
        }
      }),
    );

    const entries = Array.from(sums.entries()).sort((a, b) => compareAggregateEntries(a, b, sign));
    return {
      rowIndices: validRows,
      indices: Uint32Array.from(entries, ([index]) => index),
      values: Float32Array.from(entries, ([, value]) => value),
    };
  }

  private arrayUrl(spec: ArraySpec): URL {
    return new URL(spec.path, this.manifestUrl);
  }

  private cacheKey(spec: ArraySpec): string {
    return `${spec.dtype}:${this.arrayUrl(spec).toString()}`;
  }
}

export function dequantizeInt16Values(values: Int16Array, scale: number): Float32Array {
  const out = new Float32Array(values.length);
  for (let index = 0; index < values.length; index += 1) {
    out[index] = values[index] * scale;
  }
  return out;
}

function aggregateContribution(value: number, sign: InfluenceSign): number {
  if (sign === "abs") return Math.abs(value);
  if (sign === "pos") return value > 0 ? value : 0;
  return value < 0 ? value : 0;
}

function compareAggregateEntries(
  a: [number, number],
  b: [number, number],
  sign: InfluenceSign,
): number {
  if (sign === "neg") return a[1] - b[1] || a[0] - b[0];
  if (sign === "pos") return b[1] - a[1] || a[0] - b[0];
  return Math.abs(b[1]) - Math.abs(a[1]) || a[0] - b[0];
}

export function dequantizeUint16Raster(values: Uint16Array, field: RasterFieldManifest): Float32Array {
  const out = new Float32Array(values.length);
  const missing = field.encoding.missing ?? 65535;
  const span = field.encoding.max - field.encoding.min || 1;
  for (let index = 0; index < values.length; index += 1) {
    const q = values[index];
    out[index] = q === missing ? Number.NaN : field.encoding.min + (q / 65534) * span;
  }
  return out;
}
