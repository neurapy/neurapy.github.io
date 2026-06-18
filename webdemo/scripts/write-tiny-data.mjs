import { rm, mkdir, writeFile, stat } from "node:fs/promises";
import { join, dirname } from "node:path";

const root = join(process.cwd(), "test-results", "tiny-data");

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

  const signMultiplier = quality === "good" ? 1 : -1;
  const denseScores = await writeArray(
    runRoot,
    join(runRoot, "influence", matrixId, "scores.f32"),
    new Float32Array([
      0.1, 0.3, 0.18, -0.22, 0.06,
      0.04, -0.09, 0.18, 0.07, 0.26,
      0.12, -0.14, 0.08, -0.31, 0.045,
      0.05, -0.15, 0.055, 0.075, 0.28,
    ].map((value) => value * signMultiplier)),
    "float32",
    [4, 5],
  );

  const manifest = {
    schema_version: 9,
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
        label: "PINNfluence: total_loss -> total_loss",
        display_label: `PINNfluence / total loss -> total loss (${matrixId})`,
        scores: denseScores,
        score_layout: {
          kind: "dense_row_major",
          row_stride_bytes: 20,
          data_offset_bytes: 0,
        },
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
  const burgersConfig = {
    problem: "burgers",
    displayName: "Burgers",
    bounds: { x: [-1, 1], y: [0, 1] },
    matrixId: "m_burgers",
  };
  const navierConfig = {
    problem: "navier_stokes_nd",
    displayName: "Navier Stokes",
    bounds: { x: [0, 22], y: [0, 4.1] },
    matrixId: "m_navier",
  };
  const good = await buildVariant("good", fixtureConfig);
  const bad = await buildVariant("bad", fixtureConfig);
  const shiftedGood = await buildVariant("good", shiftedConfig);
  const shiftedBad = await buildVariant("bad", shiftedConfig);
  const driftGood = await buildVariant("good", driftConfig);
  const driftBad = await buildVariant("bad", driftConfig);
  const burgersGood = await buildVariant("good", burgersConfig);
  const burgersBad = await buildVariant("bad", burgersConfig);
  const navierGood = await buildVariant("good", navierConfig);
  const navierBad = await buildVariant("bad", navierConfig);

  const index = {
    schema_version: 9,
    generated_at: "2026-06-09T00:00:00+0000",
    matrix_mode: "core",
    max_local_influence_points: 4,
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
        problem: "burgers",
        display_name: "Burgers",
        variants: { good: burgersGood, bad: burgersBad },
      },
      {
        problem: "drift_diffusion",
        display_name: "Drift Diffusion",
        variants: { good: driftGood, bad: driftBad },
      },
      {
        problem: "navier_stokes_nd",
        display_name: "Navier Stokes",
        variants: { good: navierGood, bad: navierBad },
      },
    ],
  };
  await writeJson(join(root, "index.json"), index);
  await writeJson(join(root, "results", "index.json"), await buildResultsFixture(index));

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
    schema_version: 9,
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

async function buildResultsFixture(index) {
  const lossDecompositions = [];
  for (const problem of index.problems) {
    lossDecompositions.push(
      await tinyLossDecomposition(problem.problem, problem.display_name, "good"),
      await tinyLossDecomposition(problem.problem, problem.display_name, "bad"),
    );
  }
  return {
    schema_version: 2,
    generated_at: "2026-06-09T00:00:00+0000",
    sources: ["tiny fixture"],
    loss_decompositions: lossDecompositions,
    indicators: {
      temporal: [
        tinyTemporal("burgers", "Burgers", 0.43, 0.41, 0.02, 0.28, 0.02),
        tinyTemporal("drift_diffusion", "Drift Diffusion", 0.46, 0.46, 0.04, 0.21, 0.06),
      ],
      directionality: [
        {
          problem: "navier_stokes_nd",
          display_name: "Navier Stokes",
          output_id: "output_0",
          output_label: "x-velocity",
          baseline: 0.48,
          values: {
            good: { mean: 0.15, std: 0.01 },
            bad: { mean: 0.15, std: 0.03 },
          },
        },
      ],
    },
  };
}

async function tinyLossDecomposition(problem, displayName, quality) {
  const bad = quality === "bad";
  const outputId = "output_0";
  const relativeDir = join(problem, quality, outputId);
  const resultsRoot = join(root, "results");
  const binCenters = new Float32Array([0.16, 0.5, 0.84]);
  const hasBoundaryTerm = problem === "burgers";
  const pdeFractions = bad ? [0.45, 0.58, 0.7] : [0.62, 0.72, 0.82];
  const icFractions = hasBoundaryTerm
    ? bad
      ? [0.4, 0.3, 0.2]
      : [0.3, 0.2, 0.1]
    : bad
      ? [0.55, 0.42, 0.3]
      : [0.38, 0.28, 0.18];
  const boundaryFractions = bad ? [0.15, 0.12, 0.1] : [0.08, 0.08, 0.08];
  const terms = [
    {
      id: "pde_0",
      label: "PDE Loss",
      mean_fraction: bad ? 0.58 : 0.72,
      std_fraction: 0.04,
    },
    {
      id: "bc_0",
      label: "IC Loss",
      mean_fraction: bad ? 0.42 : 0.28,
      std_fraction: 0.04,
    },
    ...(hasBoundaryTerm
      ? [
          {
            id: "bc_1",
            label: "Dirichlet BC ($x=-1$ and $x=1$)",
            mean_fraction: bad ? 0.12 : 0.08,
            std_fraction: 0.02,
          },
        ]
      : []),
  ];
  const coherence = bad ? [0.9, 0.91, 0.92] : [0.96, 0.95, 0.97];
  const binnedFractions = hasBoundaryTerm
    ? new Float32Array([...pdeFractions, ...icFractions, ...boundaryFractions])
    : new Float32Array([...pdeFractions, ...icFractions]);
  const binnedFractionStd = hasBoundaryTerm
    ? new Float32Array([0.02, 0.02, 0.03, 0.02, 0.02, 0.03, 0.01, 0.01, 0.01])
    : new Float32Array([0.02, 0.02, 0.03, 0.02, 0.02, 0.03]);
  return {
    problem,
    display_name: displayName,
    quality,
    source_kind: "full_matrix",
    source_dir: `raw_data/${problem}_float64_${quality}/tiny_influence_scores`,
    axis: tinyAxis(problem),
    outputs: [
      {
        id: outputId,
        label: "u",
        mean_coherence: bad ? 0.91 : 0.96,
        std_coherence: 0.02,
        n_bins: 3,
        n_terms: terms.length,
        n_candidate: 4,
        n_train: 5,
        source_matrix_ids: terms.map((term) => `influences_${term.id}_output_0`),
        terms,
        arrays: {
          bin_centers: await writeArray(
            resultsRoot,
            join(resultsRoot, relativeDir, "bin_centers.f32"),
            binCenters,
            "float32",
            [3],
          ),
          binned_fractions: await writeArray(
            resultsRoot,
            join(resultsRoot, relativeDir, "binned_fractions.f32"),
            binnedFractions,
            "float32",
            [terms.length, 3],
          ),
          binned_fractions_std: await writeArray(
            resultsRoot,
            join(resultsRoot, relativeDir, "binned_fractions_std.f32"),
            binnedFractionStd,
            "float32",
            [terms.length, 3],
          ),
          binned_coherence: await writeArray(
            resultsRoot,
            join(resultsRoot, relativeDir, "binned_coherence.f32"),
            new Float32Array(coherence),
            "float32",
            [3],
          ),
          binned_coherence_std: await writeArray(
            resultsRoot,
            join(resultsRoot, relativeDir, "binned_coherence_std.f32"),
            new Float32Array([0.01, 0.01, 0.015]),
            "float32",
            [3],
          ),
        },
      },
    ],
  };
}

function tinyAxis(problem) {
  if (problem === "navier_stokes_nd") return { id: "x", label: "x" };
  return { id: "t", label: "t" };
}

function tinyTemporal(problem, displayName, baseline, good, goodStd, bad, badStd) {
  return {
    problem,
    display_name: displayName,
    baseline,
    values: {
      good: { mean: good, std: goodStd },
      bad: { mean: bad, std: badStd },
    },
  };
}
