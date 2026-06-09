import type {
  ArraySpec,
  FieldKind,
  InfluenceMatrixManifest,
  InfluenceSign,
  RunManifest,
  SelectionMode,
  SummaryName,
} from "../types";
import type { DataRepository } from "./arrays";

const INFLUENCE_SIGNS: InfluenceSign[] = ["abs", "pos", "neg"];
const SUMMARY_NAMES: SummaryName[] = [
  "mean_abs",
  "mean_signed",
  "max_abs",
  "positive_mass",
  "negative_mass",
];

export interface PrefetchContext {
  fieldId: string | null;
  fieldKind: FieldKind;
  matrixId: string | null;
  sign: InfluenceSign;
  summary: SummaryName;
  selectedCandidateIndex: number;
  selectedTrainIndex: number;
  selectionMode?: SelectionMode;
  selectedRegionCandidateIndices?: number[];
}

export interface PrefetchTask {
  key: string;
  spec: ArraySpec;
  label: string;
  rank: number;
  sequence: number;
}

export class RunPrefetcher {
  private readonly attempted = new Set<string>();
  private readonly failed = new Set<string>();
  private queue: PrefetchTask[] = [];
  private generation = 0;
  private pumping = false;
  private idleResolvers: Array<() => void> = [];

  constructor(
    private readonly repo: DataRepository,
    private readonly manifest: RunManifest,
  ) {}

  get attemptedCount(): number {
    return this.attempted.size;
  }

  get failedCount(): number {
    return this.failed.size;
  }

  update(context: PrefetchContext): void {
    this.queue = planRunPrefetchTasks(this.manifest, context).filter(
      (task) => !this.attempted.has(task.key) && !this.repo.hasArray(task.spec),
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
        if (this.repo.hasArray(task.spec)) continue;
        try {
          await this.repo.prefetchArray(task.spec);
        } catch {
          if (generation !== this.generation) return;
          this.failed.add(task.key);
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
}

export function planRunPrefetchTasks(manifest: RunManifest, context: PrefetchContext): PrefetchTask[] {
  const selectedMatrixId =
    context.matrixId ?? manifest.default_matrix ?? manifest.influence_matrices[0]?.id ?? null;
  const tasks: PrefetchTask[] = [];
  const seen = new Set<string>();
  let sequence = 0;

  const add = (spec: ArraySpec | undefined, rank: number, label: string) => {
    if (!spec) return;
    const key = arraySpecKey(spec);
    if (seen.has(key)) return;
    seen.add(key);
    tasks.push({ key, spec, label, rank, sequence: sequence++ });
  };

  add(manifest.field_raster?.mask, 0, "field-raster-mask");

  for (const [fieldId, field] of Object.entries(manifest.fields)) {
    const rank =
      fieldId === context.fieldId
        ? 1
        : field.kind === context.fieldKind
          ? 10
          : 60;
    add(field.raster, rank, `field:${fieldId}`);
  }

  for (const matrix of manifest.influence_matrices) {
    const selectedMatrix = matrix.id === selectedMatrixId;
    for (const summaryName of SUMMARY_NAMES) {
      const selectedSummary = summaryName === context.summary;
      add(
        matrix.summary[summaryName],
        selectedMatrix ? (selectedSummary ? 20 : 22) : 80,
        `summary:${matrix.id}:${summaryName}`,
      );
    }
    addMatrixChunkTasks(add, matrix, context, selectedMatrix);
  }

  return tasks.sort((a, b) => a.rank - b.rank || a.sequence - b.sequence);
}

function addMatrixChunkTasks(
  add: (spec: ArraySpec | undefined, rank: number, label: string) => void,
  matrix: InfluenceMatrixManifest,
  context: PrefetchContext,
  selectedMatrix: boolean,
): void {
  const selectedRow = selectedRowIndex(matrix, context);
  const selectedRows =
    context.selectionMode === "region" && context.selectedRegionCandidateIndices?.length
      ? context.selectedRegionCandidateIndices
      : [selectedRow];
  for (const sign of INFLUENCE_SIGNS) {
    const group = matrix.top_chunks[sign];
    if (!group) continue;
    const signOffset = sign === context.sign ? 0 : 10;
    for (const chunk of group.chunks) {
      const distance = chunkDistanceToRows(chunk.row_start, chunk.row_count, selectedRows);
      const rank = selectedMatrix ? 30 + signOffset + Math.min(distance, 10_000) : 100 + signOffset + chunk.id;
      add(chunk.indices, rank, `chunk:${matrix.id}:${sign}:${chunk.id}:indices`);
      add(chunk.values, rank, `chunk:${matrix.id}:${sign}:${chunk.id}:values`);
    }
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
