import type {
  ArraySpec,
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
