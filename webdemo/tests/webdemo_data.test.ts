import { describe, expect, it, vi } from "vitest";

import { DataRepository, dequantizeInt16Values, dequantizeUint16Raster } from "../src/data/arrays";
import { assertV5RunManifest } from "../src/data/manifest";
import { typedArrayFromBuffer } from "../src/data/dtypes";
import { PriorityLoader } from "../src/data/loader";
import {
  RunPrefetcher,
  planRunPrefetchTasks,
  type PrefetchContext,
} from "../src/data/prefetcher";
import type { RunManifest } from "../src/types";

function bufferFrom<T extends ArrayBufferView>(array: T): ArrayBuffer {
  const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  return copy.buffer as ArrayBuffer;
}

interface DeferredFetchCall {
  url: string;
  signal: AbortSignal | null;
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
    return new Promise<Response>((resolve, reject) => {
      const call: DeferredFetchCall = {
        url: input.toString(),
        signal,
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
  schema_version: 5,
  problem: "fixture",
  folder: "fixture_float64",
  run_id: "fixture_run",
  display_name: "Fixture",
  status: "complete",
  errors: [],
  generated_at: "2026-06-09T00:00:00+0000",
  max_local_influence_points: 2,
  row_chunk_size: 2,
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
      row_chunk_size: 2,
      label: "m0",
      display_label: "m0",
      top_chunks: {
        abs: {
          row_chunk_size: 2,
          chunk_count: 1,
          indices_dtype: "uint16",
          values_dtype: "int16",
          value_encoding: { kind: "symmetric_linear", scale_by: "chunk.value_scale" },
          chunks: [
            {
              id: 0,
              row_start: 0,
              row_count: 2,
              k: 2,
              value_scale: 0.01,
              indices: { path: "m0/abs/chunks/0_indices.u16", dtype: "uint16", shape: [2, 2] },
              values: { path: "m0/abs/chunks/0_values.i16", dtype: "int16", shape: [2, 2] },
            },
          ],
        },
        pos: {
          row_chunk_size: 2,
          chunk_count: 1,
          indices_dtype: "uint16",
          values_dtype: "int16",
          value_encoding: { kind: "symmetric_linear", scale_by: "chunk.value_scale" },
          chunks: [],
        },
        neg: {
          row_chunk_size: 2,
          chunk_count: 1,
          indices_dtype: "uint16",
          values_dtype: "int16",
          value_encoding: { kind: "symmetric_linear", scale_by: "chunk.value_scale" },
          chunks: [],
        },
      },
      summary: {
        mean_abs: { path: "summary.f32", dtype: "float32", shape: [3] },
        mean_signed: { path: "summary.f32", dtype: "float32", shape: [3] },
        max_abs: { path: "summary.f32", dtype: "float32", shape: [3] },
        positive_mass: { path: "summary.f32", dtype: "float32", shape: [3] },
        negative_mass: { path: "summary.f32", dtype: "float32", shape: [3] },
      },
    },
  ],
} satisfies RunManifest;

describe("typed array validation", () => {
  it("rejects buffers that do not match dtype and shape metadata", () => {
    expect(() =>
      typedArrayFromBuffer({ path: "bad.f32", dtype: "float32", shape: [3] }, new ArrayBuffer(4)),
    ).toThrow(/expected 12 bytes/);
  });

  it("parses v5 manifests and rejects older versions", () => {
    expect(assertV5RunManifest(manifest)).toBe(manifest);
    expect(() => assertV5RunManifest({ ...manifest, schema_version: 4 } as unknown as RunManifest)).toThrow(
      /expected 5/,
    );
  });
});

describe("quantized data helpers", () => {
  it("dequantizes int16 influence chunks", () => {
    expect(Array.from(dequantizeInt16Values(new Int16Array([-200, 0, 125]), 0.5))).toEqual([
      -100, 0, 62.5,
    ]);
  });

  it("dequantizes uint16 raster grids and preserves missing values", () => {
    const decoded = dequantizeUint16Raster(new Uint16Array([0, 32767, 65534, 65535]), manifest.fields.pred_output_0);

    expect(decoded[0]).toBeCloseTo(-1);
    expect(decoded[1]).toBeCloseTo(0, 4);
    expect(decoded[2]).toBeCloseTo(1);
    expect(Number.isNaN(decoded[3])).toBe(true);
  });
});

describe("chunk row lookup", () => {
  it("loads only the chunk containing the selected row", async () => {
    const buffers = new Map<string, ArrayBuffer>([
      ["http://example.test/m0/abs/chunks/0_indices.u16", bufferFrom(new Uint16Array([2, 1, 1, 0]))],
      ["http://example.test/m0/abs/chunks/0_values.i16", bufferFrom(new Int16Array([100, -50, 20, -10]))],
    ]);
    globalThis.fetch = vi.fn(async (input: RequestInfo | URL) => {
      const key = input.toString();
      const buffer = buffers.get(key);
      if (!buffer) return new Response(null, { status: 404 });
      return new Response(buffer.slice(0));
    });
    const repo = new DataRepository(new URL("http://example.test/manifest.json"), manifest, 1024);
    const row = await repo.loadInfluenceRow(manifest.influence_matrices[0], "abs", 1);

    expect(Array.from(row.indices)).toEqual([1, 0]);
    expect(row.values[0]).toBeCloseTo(0.2);
    expect(row.values[1]).toBeCloseTo(-0.1);
    expect(globalThis.fetch).toHaveBeenCalledTimes(2);
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
    fieldKind: "prediction",
    matrixId: "m0",
    sign: "abs",
    summary: "mean_abs",
    selectedCandidateIndex: 0,
    selectedTrainIndex: 0,
  };

  it("enumerates run assets once in likely-use order", () => {
    const tasks = planRunPrefetchTasks(manifest, context);
    const paths = tasks.map((task) => task.spec.path);

    expect(paths).toContain("mask.u8");
    expect(paths).toContain("pred.u16");
    expect(paths).toContain("summary.f32");
    expect(paths).toContain("m0/abs/chunks/0_indices.u16");
    expect(paths).toContain("m0/abs/chunks/0_values.i16");
    expect(new Set(paths).size).toBe(paths.length);
    expect(paths.indexOf("pred.u16")).toBeLessThan(paths.indexOf("summary.f32"));
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
