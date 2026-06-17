import type {
  ArraySpec,
  InfluenceAggregate,
  InfluenceMatrixManifest,
  InfluenceRow,
  InfluenceSign,
  PointArrays,
  Priority,
  RasterData,
  RunManifest,
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
    cacheOrBytes: LruCache<TypedArray> | number = 128 * 1024 * 1024,
    loader = new PriorityLoader(),
  ) {
    this.loader = loader;
    this.cache =
      cacheOrBytes instanceof LruCache ? cacheOrBytes : new LruCache<TypedArray>(cacheOrBytes);
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

  hasInfluenceRows(matrix: InfluenceMatrixManifest, rowStart: number, rowCount: number): boolean {
    if (rowCount <= 0) return true;
    const end = rowStart + rowCount;
    if (rowStart < 0 || end > matrix.row_count) return false;
    for (let rowIndex = rowStart; rowIndex < end; rowIndex += 1) {
      if (!this.cache.has(this.scoreRowCacheKey(matrix, rowIndex))) return false;
    }
    return true;
  }

  async prefetchInfluenceRows(
    matrix: InfluenceMatrixManifest,
    rowStart: number,
    rowCount: number,
  ): Promise<void> {
    await this.loadScoreRows(matrix, rowStart, rowCount, "background");
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

  async loadInfluenceRow(
    matrix: InfluenceMatrixManifest,
    sign: InfluenceSign,
    rowIndex: number,
    priority: Priority = "foreground",
  ): Promise<InfluenceRow> {
    const scoreRow = await this.loadScoreRow(matrix, rowIndex, priority);
    const { indices, values } = topKDenseRow(scoreRow, sign, matrix.k);
    return {
      rowIndex,
      indices,
      values,
      rawValues: values,
    };
  }

  async loadInfluenceAggregate(
    matrix: InfluenceMatrixManifest,
    sign: InfluenceSign,
    rowIndices: number[],
    priority: Priority = "foreground",
  ): Promise<InfluenceAggregate> {
    const validRows: number[] = [];
    for (const rowIndex of rowIndices) {
      if (!Number.isFinite(rowIndex)) continue;
      const row = Math.trunc(rowIndex);
      if (row < 0 || row >= matrix.row_count) continue;
      validRows.push(row);
    }

    await this.loadScoreRowsByIndexList(matrix, validRows, priority);
    const totals = new Map<number, number>();
    let meanTotal = 0;
    let meanCount = 0;
    for (const row of validRows) {
      const scoreRow = this.cachedScoreRow(matrix, row);
      for (const value of scoreRow) {
        if (!Number.isFinite(value)) continue;
        if (sign === "pos" && value <= 0) continue;
        if (sign === "neg" && value >= 0) continue;
        meanTotal += value;
        meanCount += 1;
      }
      const { indices, values } = topKDenseRow(scoreRow, sign, matrix.k);
      for (let index = 0; index < indices.length; index += 1) {
        const trainIndex = indices[index];
        const contribution = aggregateContribution(values[index], sign);
        if (!contribution) continue;
        totals.set(trainIndex, (totals.get(trainIndex) ?? 0) + contribution);
      }
    }

    const selectedRowCount = validRows.length || 1;
    const entries = Array.from(totals.entries(), ([index, total]) => [
      index,
      total / selectedRowCount,
    ] as [number, number]).sort((a, b) => compareAggregateEntries(a, b, sign));
    return {
      rowIndices: validRows,
      indices: Uint32Array.from(entries, ([index]) => index),
      values: Float32Array.from(entries, ([, value]) => value),
      meanValue: meanCount > 0 ? meanTotal / meanCount : Number.NaN,
    };
  }

  private async loadScoreRow(
    matrix: InfluenceMatrixManifest,
    rowIndex: number,
    priority: Priority,
  ): Promise<Float32Array> {
    await this.loadScoreRows(matrix, rowIndex, 1, priority);
    return this.cachedScoreRow(matrix, rowIndex);
  }

  private cachedScoreRow(matrix: InfluenceMatrixManifest, rowIndex: number): Float32Array {
    const cached = this.cache.get(this.scoreRowCacheKey(matrix, rowIndex));
    if (!cached || !(cached instanceof Float32Array)) {
      throw new Error(`${matrix.id}: row ${rowIndex} is not cached`);
    }
    return cached;
  }

  private async loadScoreRowsByIndexList(
    matrix: InfluenceMatrixManifest,
    rowIndices: number[],
    priority: Priority,
  ): Promise<void> {
    const uniqueRows = Array.from(new Set(rowIndices)).sort((a, b) => a - b);
    const groups = contiguousRowGroups(uniqueRows);
    await Promise.all(
      groups.map((group) => this.loadScoreRows(matrix, group.rowStart, group.rowCount, priority)),
    );
  }

  private async loadScoreRows(
    matrix: InfluenceMatrixManifest,
    rowStart: number,
    rowCount: number,
    priority: Priority,
  ): Promise<void> {
    this.assertDenseScores(matrix);
    if (!Number.isFinite(rowStart) || !Number.isFinite(rowCount)) {
      throw new Error(`${matrix.id}: invalid dense row range`);
    }
    const firstRow = Math.trunc(rowStart);
    const count = Math.trunc(rowCount);
    if (count < 1) return;
    const endRow = firstRow + count;
    if (firstRow < 0 || endRow > matrix.row_count) {
      throw new Error(`${matrix.id}: row range ${firstRow}-${endRow} is outside dense scores`);
    }

    const missingGroups: Array<{ rowStart: number; rowCount: number }> = [];
    let missingStart: number | null = null;
    for (let rowIndex = firstRow; rowIndex < endRow; rowIndex += 1) {
      const missing = !this.cache.has(this.scoreRowCacheKey(matrix, rowIndex));
      if (missing && missingStart === null) {
        missingStart = rowIndex;
      } else if (!missing && missingStart !== null) {
        missingGroups.push({ rowStart: missingStart, rowCount: rowIndex - missingStart });
        missingStart = null;
      }
    }
    if (missingStart !== null) {
      missingGroups.push({ rowStart: missingStart, rowCount: endRow - missingStart });
    }

    await Promise.all(
      missingGroups.map(async (group) => {
        const range = this.scoreByteRange(matrix, group.rowStart, group.rowCount);
        const buffer = await this.loader.loadRange(
          this.arrayUrl(matrix.scores),
          range.start,
          range.endExclusive,
          priority,
        );
        const rowStrideBytes = matrix.score_layout.row_stride_bytes;
        const nTrain = matrix.scores.shape[1] ?? 0;
        for (let offset = 0; offset < group.rowCount; offset += 1) {
          const rowOffset = offset * rowStrideBytes;
          const row = new Float32Array(buffer, rowOffset, nTrain);
          const rowIndex = group.rowStart + offset;
          const copy = new Float32Array(row);
          this.cache.set(this.scoreRowCacheKey(matrix, rowIndex), copy, copy.byteLength);
        }
      }),
    );
  }

  private assertDenseScores(matrix: InfluenceMatrixManifest): void {
    assertDType(matrix.scores, "float32");
    if (matrix.score_layout.kind !== "dense_row_major") {
      throw new Error(`${matrix.id}: unsupported score layout ${matrix.score_layout.kind}`);
    }
    const expectedShape = [matrix.row_count, this.manifest.n_train];
    if (matrix.scores.shape.length !== 2 || matrix.scores.shape[0] !== expectedShape[0] || matrix.scores.shape[1] !== expectedShape[1]) {
      throw new Error(`${matrix.id}: dense scores shape ${matrix.scores.shape.join("x")} != ${expectedShape.join("x")}`);
    }
    const expectedRowStride = this.manifest.n_train * Float32Array.BYTES_PER_ELEMENT;
    if (matrix.score_layout.row_stride_bytes !== expectedRowStride) {
      throw new Error(`${matrix.id}: invalid dense score row stride`);
    }
    if (matrix.score_layout.data_offset_bytes !== 0) {
      throw new Error(`${matrix.id}: unsupported dense score data offset`);
    }
  }

  private scoreByteRange(
    matrix: InfluenceMatrixManifest,
    rowStart: number,
    rowCount: number,
  ): { start: number; endExclusive: number } {
    const start =
      matrix.score_layout.data_offset_bytes + rowStart * matrix.score_layout.row_stride_bytes;
    return {
      start,
      endExclusive: start + rowCount * matrix.score_layout.row_stride_bytes,
    };
  }

  private arrayUrl(spec: ArraySpec): URL {
    return new URL(spec.path, this.manifestUrl);
  }

  private cacheKey(spec: ArraySpec): string {
    return `${spec.dtype}:${this.arrayUrl(spec).toString()}`;
  }

  private scoreRowCacheKey(matrix: InfluenceMatrixManifest, rowIndex: number): string {
    return `score-row:${this.arrayUrl(matrix.scores).toString()}:${rowIndex}`;
  }
}

function aggregateContribution(value: number, sign: InfluenceSign): number {
  if (sign === "abs") return value;
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

function contiguousRowGroups(rowIndices: number[]): Array<{ rowStart: number; rowCount: number }> {
  const groups: Array<{ rowStart: number; rowCount: number }> = [];
  let groupStart: number | null = null;
  let previous: number | null = null;
  for (const row of rowIndices) {
    if (groupStart === null || previous === null) {
      groupStart = row;
      previous = row;
      continue;
    }
    if (row === previous + 1) {
      previous = row;
      continue;
    }
    groups.push({ rowStart: groupStart, rowCount: previous - groupStart + 1 });
    groupStart = row;
    previous = row;
  }
  if (groupStart !== null && previous !== null) {
    groups.push({ rowStart: groupStart, rowCount: previous - groupStart + 1 });
  }
  return groups;
}

function topKDenseRow(
  row: Float32Array,
  sign: InfluenceSign,
  k: number,
): { indices: Uint32Array; values: Float32Array } {
  const count = Math.min(Math.max(0, Math.trunc(k)), row.length);
  if (count === 0) {
    return { indices: new Uint32Array(0), values: new Float32Array(0) };
  }
  const topEntries: Array<{ index: number; value: number }> = [];
  for (let index = 0; index < row.length; index += 1) {
    const entry = { index, value: row[index] };
    if (topEntries.length < count) {
      insertDenseEntry(topEntries, entry, sign);
      continue;
    }
    if (compareDenseEntries(entry, topEntries[topEntries.length - 1], sign) < 0) {
      insertDenseEntry(topEntries, entry, sign);
      topEntries.pop();
    }
  }
  return {
    indices: Uint32Array.from(topEntries, (entry) => entry.index),
    values: Float32Array.from(topEntries, (entry) => entry.value),
  };
}

function insertDenseEntry(
  entries: Array<{ index: number; value: number }>,
  entry: { index: number; value: number },
  sign: InfluenceSign,
): void {
  let insertAt = entries.length;
  for (let index = 0; index < entries.length; index += 1) {
    if (compareDenseEntries(entry, entries[index], sign) < 0) {
      insertAt = index;
      break;
    }
  }
  entries.splice(insertAt, 0, entry);
}

function compareDenseEntries(
  a: { index: number; value: number },
  b: { index: number; value: number },
  sign: InfluenceSign,
): number {
  if (sign === "neg") return a.value - b.value || a.index - b.index;
  if (sign === "pos") return b.value - a.value || a.index - b.index;
  return Math.abs(b.value) - Math.abs(a.value) || a.index - b.index;
}
