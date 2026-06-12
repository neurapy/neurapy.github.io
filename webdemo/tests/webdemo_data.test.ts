import { describe, expect, it, vi } from "vitest";

import { DataRepository, dequantizeUint16Raster } from "../src/data/arrays";
import { LruCache } from "../src/data/cache";
import {
  assertV8Index,
  assertV8RunManifest,
  formatProblemLabel,
  resolveProblemVariant,
} from "../src/data/manifest";
import { typedArrayFromBuffer } from "../src/data/dtypes";
import { PriorityLoader } from "../src/data/loader";
import {
  RunPrefetcher,
  planRunPrefetchTasks,
  type PrefetchContext,
} from "../src/data/prefetcher";
import type { InfluenceMatrixManifest, RunManifest } from "../src/types";
import type { DataIndex, TypedArray } from "../src/types";

function bufferFrom<T extends ArrayBufferView>(array: T): ArrayBuffer {
  const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  return copy.buffer as ArrayBuffer;
}

interface DeferredFetchCall {
  url: string;
  signal: AbortSignal | null;
  headers: Headers;
  resolve: (buffer?: ArrayBuffer) => void;
}

async function waitFor(assertion: () => void): Promise<void> {
  let lastError: unknown;
  for (let attempt = 0; attempt < 50; attempt += 1) {
    try {
      assertion();
      return;
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }
  throw lastError;
}

function installDeferredFetch(): DeferredFetchCall[] {
  const calls: DeferredFetchCall[] = [];
  globalThis.fetch = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const signal = init?.signal instanceof AbortSignal ? init.signal : null;
    const headers = new Headers(init?.headers);
    return new Promise<Response>((resolve, reject) => {
      const call: DeferredFetchCall = {
        url: input.toString(),
        signal,
        headers,
        resolve: (buffer = new ArrayBuffer(1)) => resolve(new Response(buffer.slice(0))),
      };
      signal?.addEventListener(
        "abort",
        () => reject(new DOMException("Request aborted", "AbortError")),
        { once: true },
      );
      calls.push(call);
    });
  });
  return calls;
}

const manifest = {
  schema_version: 8,
  problem: "fixture",
  display_name: "Fixture",
  model_quality: "good",
  folder: "fixture_float64_good",
  run_id: "fixture_run_good",
  status: "complete",
  errors: [],
  generated_at: "2026-06-09T00:00:00+0000",
  max_local_influence_points: 2,
  axes: ["x", "y"],
  bounds: { x: [0, 1], y: [0, 1] },
  n_candidate: 2,
  n_train: 3,
  n_outputs: 1,
  num_pdes: 1,
  num_bcs: 0,
  available_terms: ["total_loss"],
  term_labels: { total_loss: "Total loss" },
  default_field: "pred_output_0",
  default_matrix: "m0",
  arrays: {
    candidate_points: { path: "candidate.f32", dtype: "float32", shape: [2, 2] },
    train_points: { path: "train.f32", dtype: "float32", shape: [3, 2] },
    train_kind: { path: "kind.u8", dtype: "uint8", shape: [3] },
    train_bc_id: { path: "bc.i16", dtype: "int16", shape: [3] },
  },
  field_raster: {
    width: 2,
    height: 2,
    shape: [2, 2],
    bounds: { x: [0, 1], y: [0, 1] },
    axes: ["x", "y"],
    mask: { path: "mask.u8", dtype: "uint8", shape: [2, 2] },
  },
  fields: {
    pred_output_0: {
      label: "Prediction",
      kind: "prediction",
      raster: { path: "pred.u16", dtype: "uint16", shape: [2, 2] },
      encoding: { kind: "linear", min: -1, max: 1, missing: 65535 },
      display_domain: [-1, 1],
    },
  },
  influence_matrices: [
    {
      id: "m0",
      method: "PINNfluence",
      left_term: "total_loss",
      right_term: "total_loss",
      num_pdes: 1,
      num_bcs: 0,
      n_outputs: 1,
      self_influence: false,
      scores_shape: [2, 3],
      row_source: "candidate_points",
      row_count: 2,
      k: 2,
      max_local_influence_points: 2,
      label: "m0",
      display_label: "m0",
      scores: { path: "m0/scores.f32", dtype: "float32", shape: [2, 3], bytes: 24 },
      score_layout: {
        kind: "dense_row_major",
        row_stride_bytes: 12,
        data_offset_bytes: 0,
      },
    },
  ],
} satisfies RunManifest;

const goodVariant = {
  run_id: "fixture_run_good",
  display_name: "Fixture",
  problem: "fixture",
  model_quality: "good",
  folder: "fixture_float64_good",
  n_candidate: 2,
  n_train: 3,
  status: "complete",
  manifest: "fixture_float64_good/fixture_run_good/manifest.json",
  default_field: "pred_output_0",
  default_matrix: "m0",
  n_matrices: 1,
  n_fields: 1,
  errors: [],
} satisfies DataIndex["problems"][number]["variants"]["good"];

const badVariant = {
  ...goodVariant,
  run_id: "fixture_run_bad",
  model_quality: "bad",
  folder: "fixture_float64_bad",
  manifest: "fixture_float64_bad/fixture_run_bad/manifest.json",
} satisfies DataIndex["problems"][number]["variants"]["bad"];

const index = {
  schema_version: 8,
  generated_at: "2026-06-09T00:00:00+0000",
  matrix_mode: "core",
  max_local_influence_points: 2,
  bundle_report: "bundle_report.json",
  problems: [
    {
      problem: "fixture",
      display_name: "Fixture",
      variants: {
        good: goodVariant,
        bad: badVariant,
      },
    },
  ],
} satisfies DataIndex;

describe("typed array validation", () => {
  it("rejects buffers that do not match dtype and shape metadata", () => {
    expect(() =>
      typedArrayFromBuffer({ path: "bad.f32", dtype: "float32", shape: [3] }, new ArrayBuffer(4)),
    ).toThrow(/expected 12 bytes/);
  });

  it("parses v8 indexes and manifests and rejects schema v7", () => {
    expect(assertV8Index(index)).toBe(index);
    expect(assertV8RunManifest(manifest)).toBe(manifest);
    expect(() => assertV8RunManifest({ ...manifest, schema_version: 7 } as unknown as RunManifest)).toThrow(
      /expected 8/,
    );
    expect(() =>
      assertV8Index({ ...index, schema_version: 7, runs: [] } as unknown as DataIndex),
    ).toThrow(/expected 8/);
  });

  it("formats problem labels and resolves active Good/Bad variants", () => {
    expect(formatProblemLabel("navier_stokes_nd")).toBe("Navier Stokes");
    expect(formatProblemLabel("navier_stokes_nd_float64_bad")).toBe("Navier Stokes");
    expect(resolveProblemVariant(index, "fixture", "good")).toBe(goodVariant);
    expect(resolveProblemVariant(index, "fixture", "bad")).toBe(badVariant);
  });
});

describe("quantized data helpers", () => {
  it("dequantizes uint16 raster grids and preserves missing values", () => {
    const decoded = dequantizeUint16Raster(new Uint16Array([0, 32767, 65534, 65535]), manifest.fields.pred_output_0);

    expect(decoded[0]).toBeCloseTo(-1);
    expect(decoded[1]).toBeCloseTo(0, 4);
    expect(decoded[2]).toBeCloseTo(1);
    expect(Number.isNaN(decoded[3])).toBe(true);
  });
});

describe("dense influence row lookup", () => {
  it("loads only the selected row byte range", async () => {
    installScoreFetch(new Float32Array([1, -0.5, 0.25, 0.2, -0.1, 0.5]));
    const repo = new DataRepository(new URL("http://example.test/manifest.json"), manifest, 1024);
    const row = await repo.loadInfluenceRow(manifest.influence_matrices[0], "abs", 1);

    expect(Array.from(row.indices)).toEqual([2, 0]);
    expect(Array.from(row.values)).toEqual([expect.closeTo(0.5), expect.closeTo(0.2)]);
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    expect(fetchRangeHeader(0)).toBe("bytes=12-23");
  });

  it("can share decoded score rows across repository instances", async () => {
    installScoreFetch(new Float32Array([1, -0.5, 0.25, 0.2, -0.1, 0.5]));
    const cache = new LruCache<TypedArray>(1024);
    const url = new URL("http://example.test/manifest.json");
    const firstRepo = new DataRepository(url, manifest, cache);
    const secondRepo = new DataRepository(url, manifest, cache);

    await firstRepo.loadInfluenceRow(manifest.influence_matrices[0], "abs", 1);
    await secondRepo.loadInfluenceRow(manifest.influence_matrices[0], "abs", 1);

    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });
});

function aggregateMatrix(): InfluenceMatrixManifest {
  const matrix = manifest.influence_matrices[0];
  return {
    ...matrix,
    scores_shape: [4, 3],
    row_count: 4,
    k: 2,
    max_local_influence_points: 2,
    scores: { path: "m0/scores.f32", dtype: "float32", shape: [4, 3], bytes: 48 },
    score_layout: {
      kind: "dense_row_major",
      row_stride_bytes: 12,
      data_offset_bytes: 0,
    },
  };
}

function aggregateScores(): Float32Array {
  return new Float32Array([
    0.1, -0.2, 0.05,
    0, -0.05, 0.15,
    0.3, 0, -0.1,
    0.02, 0.05, -0.4,
  ]);
}

function installScoreFetch(scores: Float32Array): void {
  const full = bufferFrom(scores);
  globalThis.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    if (input.toString() !== "http://example.test/m0/scores.f32") {
      return new Response(null, { status: 404 });
    }
    const range = new Headers(init?.headers).get("Range");
    if (!range) return new Response(full.slice(0));
    const match = /^bytes=(\d+)-(\d+)$/.exec(range);
    if (!match) return new Response(null, { status: 400 });
    const start = Number(match[1]);
    const endInclusive = Number(match[2]);
    return new Response(full.slice(start, endInclusive + 1), { status: 206 });
  });
}

function fetchRangeHeader(index: number): string | null {
  const calls = vi.mocked(globalThis.fetch).mock.calls;
  return new Headers(calls[index]?.[1]?.headers).get("Range");
}

describe("dense influence row aggregation", () => {
  it("aggregates contiguous rows from one range", async () => {
    installScoreFetch(aggregateScores());
    const matrix = aggregateMatrix();
    const repo = new DataRepository(new URL("http://example.test/manifest.json"), manifest, 1024);

    const aggregate = await repo.loadInfluenceAggregate(matrix, "abs", [0, 1]);

    expect(Array.from(aggregate.rowIndices)).toEqual([0, 1]);
    expect(Array.from(aggregate.indices)).toEqual([1, 2, 0]);
    expect(Array.from(aggregate.values)).toEqual([
      expect.closeTo(-0.25),
      expect.closeTo(0.15),
      expect.closeTo(0.1),
    ]);
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    expect(fetchRangeHeader(0)).toBe("bytes=0-23");
  });

  it("aggregates rows across disjoint ranges", async () => {
    installScoreFetch(aggregateScores());
    const matrix = aggregateMatrix();
    const repo = new DataRepository(new URL("http://example.test/manifest.json"), manifest, 1024);

    const aggregate = await repo.loadInfluenceAggregate(matrix, "abs", [1, 2]);

    expect(Array.from(aggregate.indices)).toEqual([0, 2, 1]);
    expect(Array.from(aggregate.values)).toEqual([
      expect.closeTo(0.3),
      expect.closeTo(0.05),
      expect.closeTo(-0.05),
    ]);
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
    expect(fetchRangeHeader(0)).toBe("bytes=12-35");
  });

  it("filters and sorts aggregate values by selected sign", async () => {
    installScoreFetch(aggregateScores());
    const matrix = aggregateMatrix();
    const repo = new DataRepository(new URL("http://example.test/manifest.json"), manifest, 1024);

    const pos = await repo.loadInfluenceAggregate(matrix, "pos", [0, 1]);
    const neg = await repo.loadInfluenceAggregate(matrix, "neg", [0, 2]);

    expect(Array.from(pos.indices)).toEqual([2, 0]);
    expect(Array.from(pos.values)).toEqual([expect.closeTo(0.2), expect.closeTo(0.1)]);
    expect(Array.from(neg.indices)).toEqual([1, 2]);
    expect(Array.from(neg.values)).toEqual([expect.closeTo(-0.2), expect.closeTo(-0.1)]);
  });
});

describe("priority loader", () => {
  it("aborts queued background requests", async () => {
    globalThis.fetch = vi.fn(async () => new Response(new ArrayBuffer(1)));
    const loader = new PriorityLoader({ foreground: 0, background: 0 });
    const pending = loader.load(new URL("http://example.test/a.bin"), "background");

    loader.abortBackground();

    await expect(pending).rejects.toMatchObject({ name: "AbortError" });
  });

  it("requests byte ranges with a Range header", async () => {
    globalThis.fetch = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
      expect(new Headers(init?.headers).get("Range")).toBe("bytes=2-4");
      return new Response(bufferFrom(new Uint8Array([20, 30, 40])), { status: 206 });
    });
    const loader = new PriorityLoader();

    const buffer = await loader.loadRange(new URL("http://example.test/ranged.bin"), 2, 5);

    expect(Array.from(new Uint8Array(buffer))).toEqual([20, 30, 40]);
  });

  it("slices and reuses full responses when a server ignores Range", async () => {
    globalThis.fetch = vi.fn(async () => new Response(bufferFrom(new Uint8Array([0, 1, 2, 3, 4]))));
    const loader = new PriorityLoader();
    const url = new URL("http://example.test/full.bin");

    const first = await loader.loadRange(url, 1, 4);
    const second = await loader.loadRange(url, 2, 5);

    expect(Array.from(new Uint8Array(first))).toEqual([1, 2, 3]);
    expect(Array.from(new Uint8Array(second))).toEqual([2, 3, 4]);
    expect(globalThis.fetch).toHaveBeenCalledTimes(1);
  });

  it("does not start background requests while foreground work is active", async () => {
    const calls = installDeferredFetch();
    const loader = new PriorityLoader({ foreground: 1, background: 1 });
    const foreground = loader.load(new URL("http://example.test/foreground.bin"), "foreground");

    await waitFor(() => expect(calls.map((call) => call.url)).toEqual(["http://example.test/foreground.bin"]));
    const background = loader.load(new URL("http://example.test/background.bin"), "background");
    await new Promise((resolve) => setTimeout(resolve, 0));

    expect(calls.map((call) => call.url)).toEqual(["http://example.test/foreground.bin"]);

    calls[0].resolve();
    await foreground;
    await waitFor(() =>
      expect(calls.map((call) => call.url)).toEqual([
        "http://example.test/foreground.bin",
        "http://example.test/background.bin",
      ]),
    );
    calls[1].resolve();
    await background;
  });

  it("preempts active background requests and requeues them after foreground completion", async () => {
    const calls = installDeferredFetch();
    const loader = new PriorityLoader({ foreground: 1, background: 1 });
    const background = loader.load(new URL("http://example.test/background.bin"), "background");

    await waitFor(() => expect(calls.map((call) => call.url)).toEqual(["http://example.test/background.bin"]));
    const foreground = loader.load(new URL("http://example.test/foreground.bin"), "foreground");

    await waitFor(() => expect(calls.map((call) => call.url)).toContain("http://example.test/foreground.bin"));
    expect(calls[0].signal?.aborted).toBe(true);

    calls.find((call) => call.url === "http://example.test/foreground.bin")?.resolve();
    await foreground;
    await waitFor(() => expect(calls.filter((call) => call.url === "http://example.test/background.bin")).toHaveLength(2));

    calls.at(-1)?.resolve();
    await background;
  });

  it("promotes queued background requests when the same URL becomes foreground", async () => {
    const calls = installDeferredFetch();
    const loader = new PriorityLoader({ foreground: 1, background: 0 });
    const url = new URL("http://example.test/shared.bin");
    const background = loader.load(url, "background");
    const foreground = loader.load(url, "foreground");

    await waitFor(() => expect(calls.map((call) => call.url)).toEqual(["http://example.test/shared.bin"]));

    calls[0].resolve();
    await expect(background).resolves.toBeInstanceOf(ArrayBuffer);
    await expect(foreground).resolves.toBeInstanceOf(ArrayBuffer);
    expect(calls).toHaveLength(1);
  });
});

describe("run prefetch planner", () => {
  const context: PrefetchContext = {
    fieldId: "pred_output_0",
    matrixId: "m0",
    sign: "abs",
    selectedCandidateIndex: 0,
    selectedTrainIndex: 0,
  };

  it("enumerates run assets once in likely-use order", () => {
    const tasks = planRunPrefetchTasks(manifest, context);
    const arrayPaths = tasks
      .filter((task) => task.kind === "array")
      .map((task) => task.spec.path);
    const influenceTasks = tasks.filter((task) => task.kind === "influence_rows");

    expect(arrayPaths).toContain("mask.u8");
    expect(arrayPaths).toContain("pred.u16");
    expect(influenceTasks).toHaveLength(1);
    expect(influenceTasks[0]).toMatchObject({
      matrix: manifest.influence_matrices[0],
      rowStart: 0,
      rowCount: 2,
    });
    expect(new Set(tasks.map((task) => task.key)).size).toBe(tasks.length);
    expect(tasks.findIndex((task) => task.key.includes("pred.u16"))).toBeLessThan(
      tasks.findIndex((task) => task.kind === "influence_rows"),
    );
  });

  it("records failed background assets and does not retry them forever", async () => {
    globalThis.fetch = vi.fn(async () => new Response(null, { status: 404 }));
    const repo = new DataRepository(new URL("http://example.test/manifest.json"), manifest, 1024);
    const prefetcher = new RunPrefetcher(repo, manifest);

    prefetcher.update(context);
    await prefetcher.waitForIdle();
    const attemptedAfterFirstPass = prefetcher.attemptedCount;
    const failedAfterFirstPass = prefetcher.failedCount;

    prefetcher.update(context);
    await prefetcher.waitForIdle();

    expect(attemptedAfterFirstPass).toBeGreaterThan(0);
    expect(prefetcher.attemptedCount).toBe(attemptedAfterFirstPass);
    expect(prefetcher.failedCount).toBe(failedAfterFirstPass);
  });
});
