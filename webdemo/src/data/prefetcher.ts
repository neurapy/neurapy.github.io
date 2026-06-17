import type {
  ArraySpec,
  InfluenceMatrixManifest,
  RunManifest,
  SelectionMode,
} from "../types";
import type { DataRepository } from "./arrays";

const INFLUENCE_PREFETCH_BLOCK_BYTES = 256 * 1024;

export interface PrefetchContext {
  fieldId: string | null;
  matrixId: string | null;
  selectedCandidateIndex: number;
  selectedTrainIndex: number;
  selectionMode?: SelectionMode;
  selectedRegionCandidateIndices?: number[];
}

export type PrefetchTask = ArrayPrefetchTask | InfluenceRowsPrefetchTask;

export interface ArrayPrefetchTask {
  kind: "array";
  key: string;
  spec: ArraySpec;
  rank: number;
  sequence: number;
}

export interface InfluenceRowsPrefetchTask {
  kind: "influence_rows";
  key: string;
  matrix: InfluenceMatrixManifest;
  rowStart: number;
  rowCount: number;
  rank: number;
  sequence: number;
}

export class RunPrefetcher {
  private readonly attempted = new Set<string>();
  private queue: PrefetchTask[] = [];
  private generation = 0;
  private pumping = false;
  private idleResolvers: Array<() => void> = [];

  constructor(
    private readonly repo: DataRepository,
    private readonly manifest: RunManifest,
  ) {}

  update(context: PrefetchContext): void {
    this.queue = planRunPrefetchTasks(this.manifest, context).filter(
      (task) => !this.attempted.has(task.key) && !this.hasTask(task),
    );
    this.pump();
  }

  stop(): void {
    this.generation += 1;
    this.queue = [];
    this.repo.abortBackground();
    this.notifyIdle();
  }

  waitForIdle(): Promise<void> {
    if (!this.pumping && !this.queue.length) return Promise.resolve();
    return new Promise((resolve) => this.idleResolvers.push(resolve));
  }

  private pump(): void {
    if (this.pumping) return;
    this.pumping = true;
    const generation = this.generation;
    void this.run(generation);
  }

  private async run(generation: number): Promise<void> {
    try {
      while (generation === this.generation) {
        const task = this.queue.shift();
        if (!task) break;
        if (this.attempted.has(task.key)) continue;
        this.attempted.add(task.key);
        if (this.hasTask(task)) continue;
        try {
          if (task.kind === "array") {
            await this.repo.prefetchArray(task.spec);
          } else {
            await this.repo.prefetchInfluenceRows(task.matrix, task.rowStart, task.rowCount);
          }
        } catch {
          if (generation !== this.generation) return;
        }
      }
    } finally {
      this.pumping = false;
      if (generation === this.generation && this.queue.length) {
        this.pump();
      } else if (generation === this.generation) {
        this.notifyIdle();
      }
    }
  }

  private notifyIdle(): void {
    const resolvers = this.idleResolvers.splice(0);
    for (const resolve of resolvers) resolve();
  }

  private hasTask(task: PrefetchTask): boolean {
    if (task.kind === "array") return this.repo.hasArray(task.spec);
    return this.repo.hasInfluenceRows(task.matrix, task.rowStart, task.rowCount);
  }
}

export function planRunPrefetchTasks(manifest: RunManifest, context: PrefetchContext): PrefetchTask[] {
  const selectedMatrixId =
    context.matrixId ?? manifest.default_matrix ?? manifest.influence_matrices[0]?.id ?? null;
  const tasks: PrefetchTask[] = [];
  const seen = new Set<string>();
  let sequence = 0;

  const addArray = (spec: ArraySpec | undefined, rank: number) => {
    if (!spec) return;
    const key = arraySpecKey(spec);
    if (seen.has(key)) return;
    seen.add(key);
    tasks.push({ kind: "array", key, spec, rank, sequence: sequence++ });
  };

  const addInfluenceRows = (
    matrix: InfluenceMatrixManifest,
    rowStart: number,
    rowCount: number,
    rank: number,
  ) => {
    if (rowCount <= 0) return;
    const key = `influence:${matrix.id}:${matrix.scores.path}:${rowStart}:${rowCount}`;
    if (seen.has(key)) return;
    seen.add(key);
    tasks.push({
      kind: "influence_rows",
      key,
      matrix,
      rowStart,
      rowCount,
      rank,
      sequence: sequence++,
    });
  };

  addArray(manifest.field_raster?.mask, 0);

  for (const [fieldId, field] of Object.entries(manifest.fields)) {
    const rank = fieldId === context.fieldId ? 1 : 10;
    addArray(field.raster, rank);
  }

  for (const matrix of manifest.influence_matrices) {
    const selectedMatrix = matrix.id === selectedMatrixId;
    addMatrixRowTasks(addInfluenceRows, matrix, context, selectedMatrix);
  }

  return tasks.sort((a, b) => a.rank - b.rank || a.sequence - b.sequence);
}

function addMatrixRowTasks(
  add: (
    matrix: InfluenceMatrixManifest,
    rowStart: number,
    rowCount: number,
    rank: number,
  ) => void,
  matrix: InfluenceMatrixManifest,
  context: PrefetchContext,
  selectedMatrix: boolean,
): void {
  const selectedRow = selectedRowIndex(matrix, context);
  const selectedRows =
    context.selectionMode === "region" && context.selectedRegionCandidateIndices?.length
      ? context.selectedRegionCandidateIndices
      : [selectedRow];
  const rowStrideBytes = Math.max(1, matrix.score_layout.row_stride_bytes);
  const rowsPerBlock = Math.max(1, Math.floor(INFLUENCE_PREFETCH_BLOCK_BYTES / rowStrideBytes));
  for (let rowStart = 0, blockId = 0; rowStart < matrix.row_count; rowStart += rowsPerBlock, blockId += 1) {
    const rowCount = Math.min(rowsPerBlock, matrix.row_count - rowStart);
    const distance = chunkDistanceToRows(rowStart, rowCount, selectedRows);
    const rank = selectedMatrix ? 30 + Math.min(distance, 10_000) : 100 + blockId;
    add(matrix, rowStart, rowCount, rank);
  }
}

function selectedRowIndex(matrix: InfluenceMatrixManifest, context: PrefetchContext): number {
  const index =
    matrix.row_source === "train_points" ? context.selectedTrainIndex : context.selectedCandidateIndex;
  if (!Number.isFinite(index)) return 0;
  return Math.max(0, Math.min(matrix.row_count - 1, Math.trunc(index)));
}

function chunkDistance(rowStart: number, rowCount: number, rowIndex: number): number {
  const rowEnd = rowStart + rowCount - 1;
  if (rowIndex < rowStart) return rowStart - rowIndex;
  if (rowIndex > rowEnd) return rowIndex - rowEnd;
  return 0;
}

function chunkDistanceToRows(rowStart: number, rowCount: number, rowIndices: number[]): number {
  let minDistance = Infinity;
  for (const rowIndex of rowIndices) {
    if (!Number.isFinite(rowIndex)) continue;
    minDistance = Math.min(minDistance, chunkDistance(rowStart, rowCount, Math.trunc(rowIndex)));
  }
  return Number.isFinite(minDistance) ? minDistance : 0;
}

function arraySpecKey(spec: ArraySpec): string {
  return `${spec.dtype}:${spec.path}`;
}
