const DTYPE_CTORS = {
  float32: Float32Array,
  uint32: Uint32Array,
  uint8: Uint8Array,
  int16: Int16Array,
};

const DEFAULT_MATRIX_ID = "influences_total_loss_total_loss";

const state = {
  indexUrl: new URL("data/index.json", window.location.href),
  index: null,
  manifest: null,
  manifestUrl: null,
  arrays: {},
  fields: {},
  topCache: new Map(),
  summaryCache: new Map(),
  selectedDisplayIndex: 0,
  selectedCandidateIndex: 0,
  selectedTrainIndex: 0,
  selectedField: null,
  selectedKind: "prediction",
  selectedMatrixId: null,
  selectedSign: "abs",
  selectedSummary: "mean_abs",
  k: 25,
  screenPoints: null,
};

const dom = {
  runMeta: document.querySelector("#runMeta"),
  runSelect: document.querySelector("#runSelect"),
  resetButton: document.querySelector("#resetButton"),
  message: document.querySelector("#message"),
  fieldSelect: document.querySelector("#fieldSelect"),
  matrixSelect: document.querySelector("#matrixSelect"),
  fieldKindButtons: document.querySelector("#fieldKindButtons"),
  signButtons: document.querySelector("#signButtons"),
  kSlider: document.querySelector("#kSlider"),
  kOutput: document.querySelector("#kOutput"),
  summarySelect: document.querySelector("#summarySelect"),
  selectedPoint: document.querySelector("#selectedPoint"),
  selectedValue: document.querySelector("#selectedValue"),
  trainCount: document.querySelector("#trainCount"),
  candidateCount: document.querySelector("#candidateCount"),
  mainTitle: document.querySelector("#mainTitle"),
  mainRange: document.querySelector("#mainRange"),
  influenceRange: document.querySelector("#influenceRange"),
  globalRange: document.querySelector("#globalRange"),
  mainCanvas: document.querySelector("#mainCanvas"),
  influenceCanvas: document.querySelector("#influenceCanvas"),
  globalCanvas: document.querySelector("#globalCanvas"),
};

async function fetchJson(url) {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${url}`);
  }
  return response.json();
}

async function fetchArray(spec, baseUrl) {
  const url = new URL(spec.path, baseUrl);
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${url}`);
  }
  const buffer = await response.arrayBuffer();
  const Ctor = DTYPE_CTORS[spec.dtype];
  if (!Ctor) {
    throw new Error(`Unsupported dtype ${spec.dtype}`);
  }
  const expected = spec.shape.reduce((acc, value) => acc * value, 1);
  const values = new Ctor(buffer);
  if (values.length !== expected) {
    throw new Error(`${spec.path}: expected ${expected} values, got ${values.length}`);
  }
  return values;
}

function showMessage(text) {
  if (!text) {
    dom.message.hidden = true;
    dom.message.textContent = "";
    return;
  }
  dom.message.hidden = false;
  dom.message.textContent = text;
}

function formatNumber(value) {
  if (!Number.isFinite(value)) return "-";
  const abs = Math.abs(value);
  if ((abs > 0 && abs < 0.001) || abs >= 10000) {
    return value.toExponential(3);
  }
  return value.toLocaleString(undefined, { maximumSignificantDigits: 5 });
}

function matrixById(id) {
  return state.manifest.influence_matrices.find((matrix) => matrix.id === id);
}

function fieldById(id) {
  return state.manifest.fields[id];
}

function pointAt(points, index, dim) {
  const offset = index * dim;
  return [points[offset], points[offset + 1] ?? 0];
}

function clampIndex(index, count) {
  if (!Number.isFinite(index) || count <= 0) return 0;
  return Math.max(0, Math.min(count - 1, Math.trunc(index)));
}

function matrixRowSource(matrix) {
  return matrix.row_source ?? (matrix.self_influence ? "train_points" : "candidate_points");
}

function matrixRowArrayName(matrix) {
  return matrixRowSource(matrix) === "train_points" ? "train_points" : "candidate_points";
}

function matrixRowPoints(matrix) {
  return state.arrays[matrixRowArrayName(matrix)];
}

function matrixRowDim(matrix) {
  return state.manifest.arrays[matrixRowArrayName(matrix)]?.shape?.[1] ?? 2;
}

function selectedMatrixRowIndex(matrix, rowCount = null) {
  const index =
    matrixRowSource(matrix) === "train_points"
      ? state.selectedTrainIndex
      : state.selectedCandidateIndex;
  return clampIndex(index ?? 0, rowCount ?? matrix.row_count ?? 1);
}

function getBounds(points, dim) {
  const bounds = state.manifest?.bounds ?? {};
  const axes = state.manifest?.axes ?? ["x", "y"];
  const xBounds = bounds[axes[0]];
  const yBounds = bounds[axes[1]];
  if (xBounds && yBounds) {
    return {
      minX: xBounds[0],
      maxX: xBounds[1],
      minY: yBounds[0],
      maxY: yBounds[1],
    };
  }
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  for (let i = 0; i < points.length / dim; i += 1) {
    const x = points[i * dim];
    const y = points[i * dim + 1] ?? 0;
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  }
  return { minX, maxX, minY, maxY };
}

function prepareCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function projectPoint(x, y, bounds, width, height, padding = 28) {
  const spanX = bounds.maxX - bounds.minX || 1;
  const spanY = bounds.maxY - bounds.minY || 1;
  return [
    padding + ((x - bounds.minX) / spanX) * (width - padding * 2),
    height - padding - ((y - bounds.minY) / spanY) * (height - padding * 2),
  ];
}

function quantile(values, q) {
  const clean = [];
  for (let i = 0; i < values.length; i += 1) {
    const value = values[i];
    if (Number.isFinite(value)) clean.push(value);
  }
  if (!clean.length) return 0;
  clean.sort((a, b) => a - b);
  const pos = Math.min(clean.length - 1, Math.max(0, Math.floor(q * (clean.length - 1))));
  return clean[pos];
}

function valueRange(values, symmetric = false) {
  if (!values || !values.length) return { min: 0, max: 1 };
  if (symmetric) {
    let maxAbs = 0;
    for (let i = 0; i < values.length; i += 1) {
      const value = values[i];
      if (Number.isFinite(value)) maxAbs = Math.max(maxAbs, Math.abs(value));
    }
    return { min: -maxAbs || -1, max: maxAbs || 1 };
  }
  let min = quantile(values, 0.02);
  let max = quantile(values, 0.98);
  if (min === max) {
    min -= 1;
    max += 1;
  }
  return { min, max };
}

function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function rgb(r, g, b) {
  return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`;
}

function sequentialColor(value, min, max) {
  const t = clamp01((value - min) / (max - min || 1));
  const stops = [
    [39, 58, 94],
    [26, 116, 121],
    [56, 161, 105],
    [190, 215, 82],
    [250, 232, 105],
  ];
  const scaled = t * (stops.length - 1);
  const idx = Math.min(stops.length - 2, Math.floor(scaled));
  const local = scaled - idx;
  return rgb(
    lerp(stops[idx][0], stops[idx + 1][0], local),
    lerp(stops[idx][1], stops[idx + 1][1], local),
    lerp(stops[idx][2], stops[idx + 1][2], local),
  );
}

function divergingColor(value, maxAbs) {
  const t = clamp01((value / (maxAbs || 1) + 1) / 2);
  if (t < 0.5) {
    const local = t / 0.5;
    return rgb(31 + 224 * local, 83 + 172 * local, 148 + 107 * local);
  }
  const local = (t - 0.5) / 0.5;
  return rgb(255 - 184 * local, 255 - 198 * local, 255 - 171 * local);
}

function clearCanvas(ctx, width, height) {
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#fbfcfe";
  ctx.fillRect(0, 0, width, height);
}

function drawAxes(ctx, width, height) {
  ctx.strokeStyle = "#d8dee8";
  ctx.lineWidth = 1;
  ctx.strokeRect(28, 28, Math.max(1, width - 56), Math.max(1, height - 56));
}

function adaptivePointSize(width, height, count, mode) {
  const drawableArea = Math.max(1, (width - 56) * (height - 56));
  const spacing = Math.sqrt(drawableArea / Math.max(1, count));
  if (mode === "field") {
    return Math.max(3.5, Math.min(10, spacing * 1.2));
  }
  return Math.max(2.5, Math.min(7, spacing * 0.85));
}

function drawPointCloud({
  canvas,
  points,
  dim,
  values,
  selectedIndex,
  mode = "field",
  titleRangeEl = null,
}) {
  const { ctx, width, height } = prepareCanvas(canvas);
  clearCanvas(ctx, width, height);
  drawAxes(ctx, width, height);
  if (!points || !points.length) return null;

  const bounds = getBounds(points, dim);
  const count = points.length / dim;
  const pointSize = adaptivePointSize(width, height, count, mode);
  const halfPoint = pointSize / 2;
  const range = valueRange(values ?? new Float32Array(count), mode === "diverging");
  if (titleRangeEl) {
    titleRangeEl.textContent = `${formatNumber(range.min)} … ${formatNumber(range.max)}`;
  }

  const screen = new Float32Array(count * 2);
  for (let i = 0; i < count; i += 1) {
    const x = points[i * dim];
    const y = points[i * dim + 1] ?? 0;
    const [sx, sy] = projectPoint(x, y, bounds, width, height);
    screen[i * 2] = sx;
    screen[i * 2 + 1] = sy;
    const value = values ? values[i] : 0;
    ctx.fillStyle =
      mode === "diverging"
        ? divergingColor(value, Math.max(Math.abs(range.min), Math.abs(range.max)))
        : sequentialColor(value, range.min, range.max);
    ctx.globalAlpha = mode === "field" ? 0.9 : 0.78;
    ctx.fillRect(sx - halfPoint, sy - halfPoint, pointSize, pointSize);
  }
  ctx.globalAlpha = 1;

  if (selectedIndex != null && selectedIndex >= 0 && selectedIndex < count) {
    const sx = screen[selectedIndex * 2];
    const sy = screen[selectedIndex * 2 + 1];
    ctx.beginPath();
    ctx.arc(sx, sy, 7, 0, Math.PI * 2);
    ctx.fillStyle = "#f0b429";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "#17202a";
    ctx.stroke();
  }
  return screen;
}

async function loadTop(matrix, sign) {
  const key = `${matrix.id}:${sign}`;
  if (state.topCache.has(key)) return state.topCache.get(key);
  const specs = matrix.top[sign];
  const loaded = {
    indices: await fetchArray(specs.indices, state.manifestUrl),
    values: await fetchArray(specs.values, state.manifestUrl),
    shape: specs.indices.shape,
  };
  state.topCache.set(key, loaded);
  return loaded;
}

async function loadSummary(matrix, name) {
  const key = `${matrix.id}:${name}`;
  if (state.summaryCache.has(key)) return state.summaryCache.get(key);
  const loaded = await fetchArray(matrix.summary[name], state.manifestUrl);
  state.summaryCache.set(key, loaded);
  return loaded;
}

function topRow(top, row, k) {
  const width = top.shape[1];
  const safeRow = clampIndex(row, top.shape[0]);
  const count = Math.min(k, width);
  const indices = top.indices.subarray(safeRow * width, safeRow * width + count);
  const values = top.values.subarray(safeRow * width, safeRow * width + count);
  return { indices, values };
}

async function drawInfluencePlot() {
  const matrix = matrixById(state.selectedMatrixId);
  if (!matrix) return;
  const top = await loadTop(matrix, state.selectedSign);
  const rowIndex = selectedMatrixRowIndex(matrix, top.shape[0]);
  const { indices, values } = topRow(top, rowIndex, state.k);
  const { ctx, width, height } = prepareCanvas(dom.influenceCanvas);
  clearCanvas(ctx, width, height);
  drawAxes(ctx, width, height);

  const display = state.arrays.display_points ?? state.arrays.candidate_points;
  const train = state.arrays.train_points;
  const dim = state.manifest.arrays.train_points.shape[1];
  const bounds = getBounds(display, state.manifest.arrays.display_points?.shape?.[1] ?? state.manifest.arrays.candidate_points.shape[1]);
  const nTrain = train.length / dim;

  ctx.globalAlpha = 0.16;
  ctx.fillStyle = "#647282";
  for (let i = 0; i < nTrain; i += 1) {
    const [sx, sy] = projectPoint(train[i * dim], train[i * dim + 1] ?? 0, bounds, width, height);
    ctx.fillRect(sx - 1, sy - 1, 2, 2);
  }
  ctx.globalAlpha = 1;

  let maxAbs = 0;
  for (let i = 0; i < values.length; i += 1) {
    maxAbs = Math.max(maxAbs, Math.abs(values[i]));
  }

  for (let i = values.length - 1; i >= 0; i -= 1) {
    const trainIndex = indices[i];
    const value = values[i];
    const [sx, sy] = projectPoint(
      train[trainIndex * dim],
      train[trainIndex * dim + 1] ?? 0,
      bounds,
      width,
      height,
    );
    const radius = 4 + 9 * Math.sqrt(Math.abs(value) / (maxAbs || 1));
    ctx.beginPath();
    ctx.arc(sx, sy, radius, 0, Math.PI * 2);
    ctx.fillStyle = divergingColor(value, maxAbs);
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = "#17202a";
    ctx.stroke();
  }

  const rowPoints = matrixRowPoints(matrix);
  const rowDim = matrixRowDim(matrix);
  const [x, y] = pointAt(rowPoints, rowIndex, rowDim);
  const [sx, sy] = projectPoint(x, y, bounds, width, height);
  ctx.beginPath();
  ctx.arc(sx, sy, 8, 0, Math.PI * 2);
  ctx.fillStyle = "#f0b429";
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#17202a";
  ctx.stroke();
  dom.influenceRange.textContent = `max |I| ${formatNumber(maxAbs)}`;
}

async function drawGlobalPlot() {
  const matrix = matrixById(state.selectedMatrixId);
  if (!matrix) return;
  const values = await loadSummary(matrix, state.selectedSummary);
  drawPointCloud({
    canvas: dom.globalCanvas,
    points: state.arrays.train_points,
    dim: state.manifest.arrays.train_points.shape[1],
    values,
    selectedIndex: null,
    mode: state.selectedSummary.includes("signed") || state.selectedSummary.includes("negative")
      ? "diverging"
      : "field",
    titleRangeEl: dom.globalRange,
  });
}

function updateStats() {
  const points = state.arrays.display_points ?? state.arrays.candidate_points;
  const dim = state.manifest.arrays.display_points?.shape?.[1] ?? state.manifest.arrays.candidate_points.shape[1];
  const [x, y] = pointAt(points, state.selectedDisplayIndex, dim);
  const values = state.fields[state.selectedField];
  const value = values?.[state.selectedDisplayIndex];
  dom.selectedPoint.textContent = `(${formatNumber(x)}, ${formatNumber(y)})`;
  dom.selectedValue.textContent = formatNumber(value);
  dom.trainCount.textContent = state.manifest.n_train.toLocaleString();
  dom.candidateCount.textContent = `${(state.manifest.n_display ?? state.manifest.n_candidate).toLocaleString()} display / ${state.manifest.n_candidate.toLocaleString()} influence`;
}

function drawMainPlot() {
  const values = state.fields[state.selectedField];
  const field = fieldById(state.selectedField);
  dom.mainTitle.textContent = field?.label ?? "Field";
  state.screenPoints = drawPointCloud({
    canvas: dom.mainCanvas,
    points: state.arrays.display_points ?? state.arrays.candidate_points,
    dim: state.manifest.arrays.display_points?.shape?.[1] ?? state.manifest.arrays.candidate_points.shape[1],
    values,
    selectedIndex: state.selectedDisplayIndex,
    mode: "field",
    titleRangeEl: dom.mainRange,
  });
  updateStats();
}

async function redrawAll() {
  drawMainPlot();
  await Promise.all([drawInfluencePlot(), drawGlobalPlot()]);
}

function nearestDisplayPoint(clientX, clientY) {
  const rect = dom.mainCanvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  if (!state.screenPoints) return 0;
  let best = state.selectedDisplayIndex ?? 0;
  let bestDist = Infinity;
  for (let i = 0; i < state.screenPoints.length / 2; i += 1) {
    const dx = state.screenPoints[i * 2] - x;
    const dy = state.screenPoints[i * 2 + 1] - y;
    const dist = dx * dx + dy * dy;
    if (dist < bestDist) {
      bestDist = dist;
      best = i;
    }
  }
  return best;
}

function setSelectedDisplayPoint(displayIndex) {
  state.selectedDisplayIndex = displayIndex;
  const candidateMap = state.arrays.display_to_candidate;
  const trainMap = state.arrays.display_to_train;
  state.selectedCandidateIndex = candidateMap ? candidateMap[displayIndex] : displayIndex;
  state.selectedTrainIndex = trainMap ? trainMap[displayIndex] : displayIndex;
}

function setActiveButton(container, attr, value) {
  for (const button of container.querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset[attr] === value);
  }
}

function populateFields() {
  const fields = Object.entries(state.manifest.fields);
  const filtered = fields.filter(([, field]) => field.kind === state.selectedKind);
  const source = filtered.length ? filtered : fields;
  dom.fieldSelect.innerHTML = "";
  for (const [id, field] of source) {
    const option = document.createElement("option");
    option.value = id;
    option.textContent = field.label;
    dom.fieldSelect.append(option);
  }
  if (!source.some(([id]) => id === state.selectedField)) {
    state.selectedField = source[0]?.[0] ?? null;
  }
  dom.fieldSelect.value = state.selectedField ?? "";
  setActiveButton(dom.fieldKindButtons, "kind", state.selectedKind);
}

function populateMatrices() {
  dom.matrixSelect.innerHTML = "";
  for (const matrix of state.manifest.influence_matrices) {
    const option = document.createElement("option");
    option.value = matrix.id;
    option.textContent = matrix.display_label;
    dom.matrixSelect.append(option);
  }
  if (!matrixById(state.selectedMatrixId)) {
    state.selectedMatrixId =
      state.manifest.influence_matrices.find((matrix) => matrix.id === DEFAULT_MATRIX_ID)?.id ??
      state.manifest.influence_matrices.find((matrix) => !matrix.self_influence)?.id ??
      state.manifest.influence_matrices[0]?.id ??
      null;
  }
  const matrix = matrixById(state.selectedMatrixId);
  dom.matrixSelect.value = state.selectedMatrixId ?? "";
  dom.kSlider.max = matrix?.k ?? 1;
  state.k = Math.min(state.k, matrix?.k ?? 1);
  dom.kSlider.value = state.k;
  dom.kOutput.textContent = String(state.k);
  setActiveButton(dom.signButtons, "sign", state.selectedSign);
}

async function loadRun(manifestPath) {
  showMessage("");
  state.topCache.clear();
  state.summaryCache.clear();
  state.manifestUrl = new URL(manifestPath, state.indexUrl);
  state.manifest = await fetchJson(state.manifestUrl);
  state.arrays = {
    candidate_points: await fetchArray(state.manifest.arrays.candidate_points, state.manifestUrl),
    display_points: await fetchArray(
      state.manifest.arrays.display_points ?? state.manifest.arrays.candidate_points,
      state.manifestUrl,
    ),
    display_to_candidate: state.manifest.arrays.display_to_candidate
      ? await fetchArray(state.manifest.arrays.display_to_candidate, state.manifestUrl)
      : null,
    display_to_train: state.manifest.arrays.display_to_train
      ? await fetchArray(state.manifest.arrays.display_to_train, state.manifestUrl)
      : null,
    train_points: await fetchArray(state.manifest.arrays.train_points, state.manifestUrl),
    train_kind: await fetchArray(state.manifest.arrays.train_kind, state.manifestUrl),
    train_bc_id: await fetchArray(state.manifest.arrays.train_bc_id, state.manifestUrl),
  };

  state.fields = {};
  for (const [id, field] of Object.entries(state.manifest.fields)) {
    state.fields[id] = await fetchArray(field.array, state.manifestUrl);
  }

  setSelectedDisplayPoint(0);
  const hasPrediction = Object.values(state.manifest.fields).some((field) => field.kind === "prediction");
  state.selectedKind = hasPrediction ? "prediction" : "loss";
  state.selectedField = null;
  state.selectedMatrixId = null;
  populateFields();
  populateMatrices();

  dom.runMeta.textContent = [
    state.manifest.display_name,
    `${state.manifest.n_candidate.toLocaleString()} candidate`,
    `${(state.manifest.n_display ?? state.manifest.n_candidate).toLocaleString()} display`,
    `${state.manifest.n_train.toLocaleString()} train`,
    `${state.manifest.influence_matrices.length} matrices`,
  ].join(" · ");
  if (state.manifest.errors?.length) {
    showMessage(state.manifest.errors.join(" · "));
  }
  await redrawAll();
}

async function init() {
  try {
    state.index = await fetchJson(state.indexUrl);
    const runs = state.index.runs.filter((run) => run.manifest && run.n_matrices > 0);
    dom.runSelect.innerHTML = "";
    for (const run of runs) {
      const option = document.createElement("option");
      option.value = run.manifest;
      option.textContent = `${run.display_name} · ${run.status}`;
      dom.runSelect.append(option);
    }
    if (!runs.length) {
      showMessage("No static demo runs found in webdemo/data/index.json.");
      return;
    }
    await loadRun(runs[0].manifest);
  } catch (error) {
    showMessage(error.message);
    console.error(error);
  }
}

dom.runSelect.addEventListener("change", async () => {
  await loadRun(dom.runSelect.value);
});

dom.fieldSelect.addEventListener("change", async () => {
  state.selectedField = dom.fieldSelect.value;
  await redrawAll();
});

dom.matrixSelect.addEventListener("change", async () => {
  state.selectedMatrixId = dom.matrixSelect.value;
  populateMatrices();
  await redrawAll();
});

dom.summarySelect.addEventListener("change", async () => {
  state.selectedSummary = dom.summarySelect.value;
  await drawGlobalPlot();
});

dom.kSlider.addEventListener("input", async () => {
  state.k = Number(dom.kSlider.value);
  dom.kOutput.textContent = String(state.k);
  await drawInfluencePlot();
});

dom.fieldKindButtons.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-kind]");
  if (!button) return;
  state.selectedKind = button.dataset.kind;
  populateFields();
  await redrawAll();
});

dom.signButtons.addEventListener("click", async (event) => {
  const button = event.target.closest("button[data-sign]");
  if (!button) return;
  state.selectedSign = button.dataset.sign;
  setActiveButton(dom.signButtons, "sign", state.selectedSign);
  await drawInfluencePlot();
});

dom.resetButton.addEventListener("click", async () => {
  setSelectedDisplayPoint(0);
  await redrawAll();
});

dom.mainCanvas.addEventListener("click", async (event) => {
  setSelectedDisplayPoint(nearestDisplayPoint(event.clientX, event.clientY));
  await redrawAll();
});

window.addEventListener("resize", () => {
  window.requestAnimationFrame(() => {
    redrawAll().catch((error) => {
      showMessage(error.message);
      console.error(error);
    });
  });
});

init();
