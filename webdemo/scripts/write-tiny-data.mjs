import { rm, mkdir, writeFile, stat } from "node:fs/promises";
import { join, dirname } from "node:path";

const root = join(process.cwd(), "public", "fixtures", "tiny-data");

async function writeJson(path, payload) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, `${JSON.stringify(payload, null, 2)}\n`);
}

async function writeArray(runRoot, path, typedArray, dtype, shape) {
  await mkdir(dirname(path), { recursive: true });
  await writeFile(path, Buffer.from(typedArray.buffer, typedArray.byteOffset, typedArray.byteLength));
  return {
    path: path.slice(runRoot.length + 1).replaceAll("\\", "/"),
    dtype,
    shape,
    bytes: typedArray.byteLength,
  };
}

function rasterValues(quality) {
  const good = [
    0, 6000, 12000, 18000,
    24000, 30000, 36000, 42000,
    46000, 50000, 54000, 58000,
    61000, 63000, 64500, 65534,
  ];
  if (quality === "good") return good;
  return [
    65534, 64500, 63000, 61000,
    56000, 50000, 44000, 38000,
    32000, 26000, 20000, 14000,
    9000, 5000, 2500, 0,
  ];
}

function lossValues(quality) {
  const good = [
    65534, 62000, 58000, 54000,
    50000, 46000, 42000, 38000,
    32000, 28000, 24000, 20000,
    15000, 10000, 5000, 0,
  ];
  if (quality === "good") return good;
  return [
    0, 5000, 10000, 15000,
    20000, 24000, 28000, 32000,
    38000, 42000, 46000, 50000,
    54000, 58000, 62000, 65534,
  ];
}

function mapUnitPoints(values, bounds) {
  const xSpan = bounds.x[1] - bounds.x[0];
  const ySpan = bounds.y[1] - bounds.y[0];
  const mapped = [];
  for (let index = 0; index < values.length; index += 2) {
    mapped.push(bounds.x[0] + values[index] * xSpan, bounds.y[0] + values[index + 1] * ySpan);
  }
  return mapped;
}

async function buildVariant(quality, config) {
  const folder = `${config.problem}_float64_${quality}`;
  const runId = `${config.problem}_run_${quality}`;
  const runRoot = join(root, folder, runId);
  await mkdir(runRoot, { recursive: true });
  const bounds = config.bounds;
  const lossFieldId = config.lossFieldId ?? "loss_total";
  const lossLabel = config.lossLabel ?? "Total loss";
  const matrixId = config.matrixId ?? "m0";
  const candidateUnitPoints = config.candidateUnitPoints ?? [
    0.15, 0.25, 0.75, 0.2, 0.22, 0.78, 0.82, 0.75,
  ];
  const trainUnitPoints = config.trainUnitPoints ?? [
    0.1, 0.1, 0.35, 0.22, 0.58, 0.5, 0.25, 0.85, 0.88, 0.82,
  ];

  const candidatePoints = await writeArray(
    runRoot,
    join(runRoot, "arrays", "candidate_points.f32"),
    new Float32Array(mapUnitPoints(candidateUnitPoints, bounds)),
    "float32",
    [4, 2],
  );
  const trainPoints = await writeArray(
    runRoot,
    join(runRoot, "arrays", "train_points.f32"),
    new Float32Array(mapUnitPoints(trainUnitPoints, bounds)),
    "float32",
    [5, 2],
  );
  const trainKind = await writeArray(
    runRoot,
    join(runRoot, "arrays", "train_kind.u8"),
    new Uint8Array([0, 1, 0, 1, 0]),
    "uint8",
    [5],
  );
  const trainBcId = await writeArray(
    runRoot,
    join(runRoot, "arrays", "train_bc_id.i16"),
    new Int16Array([-1, 0, -1, 1, -1]),
    "int16",
    [5],
  );
  const mask = await writeArray(
    runRoot,
    join(runRoot, "arrays", "field_raster_mask.u8"),
    new Uint8Array(16).fill(1),
    "uint8",
    [4, 4],
  );
  const pred = await writeArray(
    runRoot,
    join(runRoot, "fields", "pred_output_0_raster.u16"),
    new Uint16Array(rasterValues(quality)),
    "uint16",
    [4, 4],
  );
  const loss = await writeArray(
    runRoot,
    join(runRoot, "fields", `${lossFieldId}_raster.u16`),
    new Uint16Array(lossValues(quality)),
    "uint16",
    [4, 4],
  );

  const rowIndices = [
    [1, 2, 3, 0],
    [4, 2, 1, 3],
    [3, 0, 2, 1],
    [4, 1, 3, 2],
  ];
  const signMultiplier = quality === "good" ? 1 : -1;
  const rowValues = {
    abs: [
      [300, 180, -220, 100],
      [260, 180, -90, 70],
      [-310, 120, 80, -65],
      [280, -150, 75, 55],
    ].map((row) => row.map((value) => value * signMultiplier)),
    pos: [
      [300, 180, 100, 60],
      [260, 180, 70, 40],
      [120, 80, 45, 30],
      [280, 75, 55, 50],
    ].map((row) => row.map((value) => value * signMultiplier)),
    neg: [
      [-220, -120, -80, -40],
      [-90, -60, -35, -20],
      [-310, -140, -95, -70],
      [-150, -110, -65, -30],
    ].map((row) => row.map((value) => value * signMultiplier)),
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
        k: 4,
        value_scale: 0.001,
        indices: await writeArray(
          runRoot,
          join(runRoot, "influence", matrixId, sign, "chunks", `${chunkId}_indices.u16`),
          new Uint16Array(rows.flat()),
          "uint16",
          [2, 4],
        ),
        values: await writeArray(
          runRoot,
          join(runRoot, "influence", matrixId, sign, "chunks", `${chunkId}_values.i16`),
          new Int16Array(vals.flat()),
          "int16",
          [2, 4],
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
    schema_version: 7,
    problem: config.problem,
    display_name: config.displayName,
    model_quality: quality,
    folder,
    run_id: runId,
    status: "complete",
    errors: [],
    generated_at: "2026-06-09T00:00:00+0000",
    matrix_mode: "core",
    max_local_influence_points: 4,
    row_chunk_size: 2,
    axes: ["x", "y"],
    bounds,
    n_candidate: 4,
    n_train: 5,
    n_outputs: 1,
    num_pdes: 1,
    num_bcs: 2,
    available_terms: ["output_0", "total_loss"],
    term_labels: { output_0: "Output", total_loss: "Total loss" },
    default_field: "pred_output_0",
    default_matrix: matrixId,
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
      bounds,
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
      [lossFieldId]: {
        label: lossLabel,
        kind: "loss",
        raster: loss,
        encoding: { kind: "linear", min: 0, max: 2, missing: 65535 },
        display_domain: [0, 2],
      },
    },
    influence_matrices: [
      {
        id: matrixId,
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
        k: 4,
        max_local_influence_points: 4,
        row_chunk_size: 2,
        label: "PINNfluence: total_loss -> total_loss",
        display_label: `PINNfluence / total loss -> total loss (${matrixId})`,
        top_chunks: topChunks,
      },
    ],
    validation: { available: false, counts: {} },
  };

  await writeJson(join(runRoot, "manifest.json"), manifest);

  return {
    run_id: runId,
    display_name: config.displayName,
    problem: config.problem,
    model_quality: quality,
    folder,
    n_candidate: 4,
    n_train: 5,
    status: "complete",
    manifest: `${folder}/${runId}/manifest.json`,
    default_field: "pred_output_0",
    default_matrix: matrixId,
    n_matrices: 1,
    n_fields: 2,
    errors: [],
  };
}

async function build() {
  await rm(root, { recursive: true, force: true });
  const fixtureConfig = {
    problem: "fixture",
    displayName: "Fixture",
    bounds: { x: [0, 1], y: [0, 1] },
    matrixId: "m0",
  };
  const shiftedConfig = {
    problem: "shifted_fixture",
    displayName: "Shifted Fixture",
    bounds: { x: [10, 20], y: [-5, 5] },
    lossFieldId: "loss_residual",
    lossLabel: "Residual loss",
    matrixId: "m_shifted",
  };
  const driftConfig = {
    problem: "drift_diffusion",
    displayName: "Drift Diffusion",
    bounds: { x: [0, 2 * Math.PI], y: [0, 1] },
    matrixId: "m_drift",
    candidateUnitPoints: [0.08, 0.25, 0.97, 0.52, 0.22, 0.78, 0.82, 0.75],
    trainUnitPoints: [0.05, 0.1, 0.34, 0.22, 0.58, 0.5, 0.25, 0.85, 0.95, 0.52],
  };
  const good = await buildVariant("good", fixtureConfig);
  const bad = await buildVariant("bad", fixtureConfig);
  const shiftedGood = await buildVariant("good", shiftedConfig);
  const shiftedBad = await buildVariant("bad", shiftedConfig);
  const driftGood = await buildVariant("good", driftConfig);
  const driftBad = await buildVariant("bad", driftConfig);

  const index = {
    schema_version: 7,
    generated_at: "2026-06-09T00:00:00+0000",
    matrix_mode: "core",
    max_local_influence_points: 4,
    row_chunk_size: 2,
    bundle_report: "bundle_report.json",
    problems: [
      {
        problem: "fixture",
        display_name: "Fixture",
        variants: { good, bad },
      },
      {
        problem: "shifted_fixture",
        display_name: "Shifted Fixture",
        variants: { good: shiftedGood, bad: shiftedBad },
      },
      {
        problem: "drift_diffusion",
        display_name: "Drift Diffusion",
        variants: { good: driftGood, bad: driftBad },
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
    schema_version: 7,
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
