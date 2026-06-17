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
    const [candidatePoints, trainPoints] = await Promise.all([
      this.loadArray<Float32Array>(arrays.candidate_points, "foreground", signal),
      this.loadArray<Float32Array>(arrays.train_points, "foreground", signal),
    ]);
    return {
      candidate_points: candidatePoints,
      train_points: trainPoints,
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
    let selectionScratch: Uint32Array | undefined;
    for (const row of validRows) {
      const scoreRow = this.cachedScoreRow(matrix, row);
      const topKCount = denseTopKCount(scoreRow, matrix.k);
      const useAllContributions = topKCount >= scoreRow.length;
      for (let trainIndex = 0; trainIndex < scoreRow.length; trainIndex += 1) {
        const value = scoreRow[trainIndex];
        if (!includeDenseValueForSign(value, sign)) continue;
        meanTotal += value;
        meanCount += 1;
        if (useAllContributions) {
          addAggregateContribution(totals, trainIndex, value, sign);
        }
      }
      if (!useAllContributions && topKCount > 0) {
        if (!selectionScratch || selectionScratch.length < scoreRow.length) {
          selectionScratch = new Uint32Array(scoreRow.length);
        }
        const selectedIndices = selectTopKDenseIndices(scoreRow, sign, topKCount, selectionScratch);
        for (const trainIndex of selectedIndices) {
          addAggregateContribution(totals, trainIndex, scoreRow[trainIndex], sign);
        }
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

function includeDenseValueForSign(value: number, sign: InfluenceSign): boolean {
  if (!Number.isFinite(value)) return false;
  if (sign === "pos") return value > 0;
  if (sign === "neg") return value < 0;
  return true;
}

function addAggregateContribution(
  totals: Map<number, number>,
  trainIndex: number,
  value: number,
  sign: InfluenceSign,
): void {
  const contribution = aggregateContribution(value, sign);
  if (!contribution) return;
  totals.set(trainIndex, (totals.get(trainIndex) ?? 0) + contribution);
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
  const count = denseTopKCount(row, k);
  if (count === 0) {
    return { indices: new Uint32Array(0), values: new Float32Array(0) };
  }
  const selectedIndices = selectTopKDenseIndices(row, sign, count);
  sortDenseIndices(selectedIndices, row, sign);
  const indices = new Uint32Array(count);
  const values = new Float32Array(count);
  for (let index = 0; index < count; index += 1) {
    const trainIndex = selectedIndices[index];
    indices[index] = trainIndex;
    values[index] = row[trainIndex];
  }
  return { indices, values };
}

function denseTopKCount(row: Float32Array, k: number): number {
  return Math.min(Math.max(0, Math.trunc(k)), row.length);
}

function selectTopKDenseIndices(
  row: Float32Array,
  sign: InfluenceSign,
  count: number,
  scratch?: Uint32Array,
): Uint32Array {
  if (count <= 0) return new Uint32Array(0);
  const indices = scratch && scratch.length >= row.length
    ? scratch.subarray(0, row.length)
    : new Uint32Array(row.length);
  for (let index = 0; index < row.length; index += 1) {
    indices[index] = index;
  }
  if (count < indices.length) {
    quickselectDenseIndices(row, indices, count - 1, sign);
  }
  return indices.subarray(0, count);
}

function quickselectDenseIndices(
  row: Float32Array,
  indices: Uint32Array,
  target: number,
  sign: InfluenceSign,
): void {
  let left = 0;
  let right = indices.length - 1;
  while (left < right) {
    const pivotIndex = indices[(left + right) >>> 1];
    const pivotValue = row[pivotIndex];
    let i = left;
    let j = right;
    while (i <= j) {
      while (compareDenseEntries(indices[i], row[indices[i]], pivotIndex, pivotValue, sign) < 0) {
        i += 1;
      }
      while (compareDenseEntries(indices[j], row[indices[j]], pivotIndex, pivotValue, sign) > 0) {
        j -= 1;
      }
      if (i <= j) {
        const temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
        i += 1;
        j -= 1;
      }
    }
    if (target <= j) {
      right = j;
    } else if (target >= i) {
      left = i;
    } else {
      break;
    }
  }
}

function sortDenseIndices(
  indices: Uint32Array,
  row: Float32Array,
  sign: InfluenceSign,
): void {
  indices.sort((a, b) => compareDenseEntries(a, row[a], b, row[b], sign));
}

function compareDenseEntries(
  aIndex: number,
  aValue: number,
  bIndex: number,
  bValue: number,
  sign: InfluenceSign,
): number {
  if (sign === "neg") return aValue - bValue || aIndex - bIndex;
  if (sign === "pos") return bValue - aValue || aIndex - bIndex;
  return Math.abs(bValue) - Math.abs(aValue) || aIndex - bIndex;
}
