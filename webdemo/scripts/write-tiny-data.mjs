import { rm, mkdir, writeFile, stat } from "node:fs/promises";
import { join, dirname } from "node:path";

const root = join(process.cwd(), "public", "fixtures", "tiny-data");
const runRoot = join(root, "fixture_float64", "fixture_run");

async function writeJson(path, payload) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(payload, null, 2)}\n`);
}

async function writeArray(path, typedArray, dtype, shape) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, Buffer.from(typedArray.buffer, typedArray.byteOffset, typedArray.byteLength));
  return {
    path: path.slice(runRoot.length + 1).replaceAll("\\", "/"),
    dtype,
    shape,
    bytes: typedArray.byteLength,
  };
}

async function build() {
  await rm(root, { recursive: true, force: true });
  await mkdir(runRoot, { recursive: true });

  const candidatePoints = await writeArray(
    join(runRoot, "arrays", "candidate_points.f32"),
    new Float32Array([0.15, 0.25, 0.75, 0.2, 0.22, 0.78, 0.82, 0.75]),
    "float32",
    [4, 2],
  );
  const trainPoints = await writeArray(
    join(runRoot, "arrays", "train_points.f32"),
    new Float32Array([0.1, 0.1, 0.35, 0.22, 0.58, 0.5, 0.25, 0.85, 0.88, 0.82]),
    "float32",
    [5, 2],
  );
  const trainKind = await writeArray(
    join(runRoot, "arrays", "train_kind.u8"),
    new Uint8Array([0, 1, 0, 1, 0]),
    "uint8",
    [5],
  );
  const trainBcId = await writeArray(
    join(runRoot, "arrays", "train_bc_id.i16"),
    new Int16Array([-1, 0, -1, 1, -1]),
    "int16",
    [5],
  );
  const mask = await writeArray(
    join(runRoot, "arrays", "field_raster_mask.u8"),
    new Uint8Array(16).fill(1),
    "uint8",
    [4, 4],
  );
  const pred = await writeArray(
    join(runRoot, "fields", "pred_output_0_raster.u16"),
    new Uint16Array([
      0, 6000, 12000, 18000,
      24000, 30000, 36000, 42000,
      46000, 50000, 54000, 58000,
      61000, 63000, 64500, 65534,
    ]),
    "uint16",
    [4, 4],
  );
  const loss = await writeArray(
    join(runRoot, "fields", "loss_total_raster.u16"),
    new Uint16Array([
      65534, 62000, 58000, 54000,
      50000, 46000, 42000, 38000,
      32000, 28000, 24000, 20000,
      15000, 10000, 5000, 0,
    ]),
    "uint16",
    [4, 4],
  );

  const summarySpecs = {};
  for (const [name, values] of Object.entries({
    mean_abs: [0.12, 0.3, 0.08, 0.22, 0.18],
    mean_signed: [0.1, -0.18, 0.03, -0.08, 0.12],
    max_abs: [0.22, 0.4, 0.12, 0.31, 0.25],
    positive_mass: [0.42, 0.15, 0.22, 0.1, 0.35],
    negative_mass: [-0.04, -0.35, -0.01, -0.28, -0.06],
  })) {
    summarySpecs[name] = await writeArray(
      join(runRoot, "influence", "m0", `summary_${name}.f32`),
      new Float32Array(values),
      "float32",
      [5],
    );
  }

  const rowIndices = [
    [1, 3, 0],
    [4, 2, 1],
    [3, 0, 2],
    [4, 1, 3],
  ];
  const rowValues = {
    abs: [
      [300, -220, 100],
      [260, 180, -90],
      [-310, 120, 80],
      [280, -150, 75],
    ],
    pos: [
      [300, 100, 60],
      [260, 180, 40],
      [120, 80, 30],
      [280, 75, 50],
    ],
    neg: [
      [-220, -120, -40],
      [-90, -60, -20],
      [-310, -140, -70],
      [-150, -110, -30],
    ],
  };

  const topChunks = {};
  for (const sign of ["abs", "pos", "neg"]) {
    const chunks = [];
    for (const chunkId of [0, 1]) {
      const rows = rowIndices.slice(chunkId * 2, chunkId * 2 + 2);
      const vals = rowValues[sign].slice(chunkId * 2, chunkId * 2 + 2);
      chunks.push({
        id: chunkId,
        row_start: chunkId * 2,
        row_count: 2,
        k: 3,
        value_scale: 0.001,
        indices: await writeArray(
          join(runRoot, "influence", "m0", sign, "chunks", `${chunkId}_indices.u16`),
          new Uint16Array(rows.flat()),
          "uint16",
          [2, 3],
        ),
        values: await writeArray(
          join(runRoot, "influence", "m0", sign, "chunks", `${chunkId}_values.i16`),
          new Int16Array(vals.flat()),
          "int16",
          [2, 3],
        ),
      });
    }
    topChunks[sign] = {
      row_chunk_size: 2,
      chunk_count: 2,
      indices_dtype: "uint16",
      values_dtype: "int16",
      value_encoding: { kind: "symmetric_linear", scale_by: "chunk.value_scale" },
      chunks,
    };
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
    matrix_mode: "core",
    k_web_max: 3,
    row_chunk_size: 2,
    axes: ["x", "y"],
    bounds: { x: [0, 1], y: [0, 1] },
    n_candidate: 4,
    n_train: 5,
    n_outputs: 1,
    num_pdes: 1,
    num_bcs: 2,
    available_terms: ["output_0", "total_loss"],
    term_labels: { output_0: "Output", total_loss: "Total loss" },
    default_field: "pred_output_0",
    default_matrix: "m0",
    arrays: {
      candidate_points: candidatePoints,
      train_points: trainPoints,
      train_kind: trainKind,
      train_bc_id: trainBcId,
    },
    field_raster: {
      width: 4,
      height: 4,
      shape: [4, 4],
      bounds: { x: [0, 1], y: [0, 1] },
      axes: ["x", "y"],
      max_axis_resolution: 4,
      coordinate_order: {
        columns: "x_ascending",
        rows: "y_descending",
        sample: "pixel_center",
      },
      mask,
    },
    fields: {
      pred_output_0: {
        label: "Prediction output",
        kind: "prediction",
        raster: pred,
        encoding: { kind: "linear", min: -1, max: 1, missing: 65535 },
        display_domain: [-0.9, 0.9],
      },
      loss_total: {
        label: "Total loss",
        kind: "loss",
        raster: loss,
        encoding: { kind: "linear", min: 0, max: 2, missing: 65535 },
        display_domain: [0, 2],
      },
    },
    influence_matrices: [
      {
        id: "m0",
        source_file: "fixtures/source_missing.npz",
        method: "PINNfluence",
        left_term: "total_loss",
        right_term: "total_loss",
        num_pdes: 1,
        num_bcs: 2,
        n_outputs: 1,
        self_influence: false,
        scores_shape: [4, 5],
        candidate_points_shape: [4, 2],
        row_source: "candidate_points",
        row_count: 4,
        k: 3,
        k_web_max: 3,
        row_chunk_size: 2,
        label: "PINNfluence: total_loss -> total_loss",
        display_label: "PINNfluence / total loss -> total loss",
        top_chunks: topChunks,
        summary: summarySpecs,
      },
    ],
    validation: { available: false, counts: {} },
  };

  await writeJson(join(runRoot, "manifest.json"), manifest);

  const index = {
    schema_version: 5,
    generated_at: "2026-06-09T00:00:00+0000",
    matrix_mode: "core",
    k_web_max: 3,
    row_chunk_size: 2,
    bundle_report: "bundle_report.json",
    runs: [
      {
        run_id: "fixture_run",
        display_name: "Fixture",
        problem: "fixture",
        n_candidate: 4,
        n_train: 5,
        status: "complete",
        manifest: "fixture_float64/fixture_run/manifest.json",
        default_field: "pred_output_0",
        default_matrix: "m0",
        n_matrices: 1,
        n_fields: 2,
        errors: [],
      },
    ],
  };
  await writeJson(join(root, "index.json"), index);

  const files = [];
  async function walk(dir) {
    const { readdir } = await import("node:fs/promises");
    for (const entry of await readdir(dir, { withFileTypes: true })) {
      const full = join(dir, entry.name);
      if (entry.isDirectory()) await walk(full);
      else files.push(full);
    }
  }
  await walk(root);
  const total = (
    await Promise.all(files.map(async (file) => (await stat(file)).size))
  ).reduce((sum, size) => sum + size, 0);
  await writeJson(join(root, "bundle_report.json"), {
    schema_version: 5,
    root: ".",
    total_bytes: total,
    budget_bytes: 750 * 1024 * 1024,
    within_budget: true,
    by_kind: {},
    by_suffix: {},
    files: [],
  });
}

await build();
