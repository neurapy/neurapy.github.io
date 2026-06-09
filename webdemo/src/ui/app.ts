import type { Delaunay } from "d3";
import type {
  DataIndex,
  FieldKind,
  IndexRunEntry,
  InfluenceMatrixManifest,
  InfluenceRow,
  PointArrays,
  RasterData,
  RunManifest,
  SummaryName,
} from "../types";
import { DataRepository } from "../data/arrays";
import { loadIndex, loadRunManifest, resolveIndexUrl } from "../data/manifest";
import { RunPrefetcher, type PrefetchContext } from "../data/prefetcher";
import { Store } from "../state/store";
import type { RasterWorkerRequest, RasterWorkerResponse } from "../worker/rasterWorker";
import { boundsFromAxisMap, clampIndex, inferPointBounds, pointAt } from "../viz/geometry";
import {
  buildDelaunay,
  pointerInDomain,
  rasterSampleAtCoord,
  renderGlobalPlot,
  renderLocalInfluencePlot,
  renderMainPlot,
  type PlotContext,
  type RasterRenderResult,
} from "../viz/plots";
import { formatNumber, getDomRefs, showMessage, type DomRefs } from "./dom";

const DEFAULT_MATRIX_ID = "influences_total_loss_total_loss";
const SUMMARY_LABELS: Record<SummaryName, string> = {
  mean_abs: "Mean |influence|",
  mean_signed: "Mean signed",
  max_abs: "Max |influence|",
  positive_mass: "Positive mass",
  negative_mass: "Negative mass",
};

type PanelName = "main" | "local" | "global";

export class AppController {
  private readonly dom: DomRefs = getDomRefs();
  private readonly store = new Store();
  private readonly indexUrl = resolveIndexUrl();
  private readonly worker = new Worker(new URL("../worker/rasterWorker.ts", import.meta.url), {
    type: "module",
  });
  private index: DataIndex | null = null;
  private manifest: RunManifest | null = null;
  private manifestUrl: URL | null = null;
  private repo: DataRepository | null = null;
  private prefetcher: RunPrefetcher | null = null;
  private points: PointArrays | null = null;
  private candidateDelaunay: Delaunay<number> | null = null;
  private trainDelaunay: Delaunay<number> | null = null;
  private raster: RasterData | null = null;
  private rasterResult: RasterRenderResult | null = null;
  private influenceRow: InfluenceRow | null = null;
  private summaryValues: Float32Array | null = null;
  private mainViewport = null as ReturnType<typeof renderMainPlot> | null;
  private latestRasterRequest = 0;
  private scheduled = new Set<PanelName>();

  async start(): Promise<void> {
    this.bindEvents();
    this.observeLayout();
    showMessage(this.dom.message, null);
    this.dom.runMeta.textContent = "Loading data index";
    try {
      this.index = await loadIndex(this.indexUrl);
      this.populateRunSelect();
      const firstRun = this.index.runs.find((run) => run.manifest);
      if (!firstRun?.manifest) {
        throw new Error("No complete v5 run manifest is available");
      }
      await this.loadRun(firstRun);
    } catch (error) {
      showMessage(this.dom.message, error instanceof Error ? error.message : String(error));
      this.dom.runMeta.textContent = "Data unavailable";
    }
  }

  private bindEvents(): void {
    this.dom.runSelect.addEventListener("change", () => {
      const run = this.index?.runs.find((entry) => entry.run_id === this.dom.runSelect.value);
      if (run) void this.loadRun(run);
    });
    this.dom.fieldSelect.addEventListener("change", () => {
      const fieldId = this.dom.fieldSelect.value;
      this.store.dispatch({ type: "field", fieldId });
      void this.loadRaster(fieldId);
    });
    this.dom.matrixSelect.addEventListener("change", () => {
      this.store.dispatch({ type: "matrix", matrixId: this.dom.matrixSelect.value });
      void Promise.all([this.loadInfluenceRow(), this.loadSummary()]).then(() => {
        this.schedule("local");
        this.schedule("global");
      });
    });
    this.dom.fieldKindButtons.addEventListener("click", (event) => {
      const button = (event.target as Element).closest<HTMLButtonElement>("button[data-kind]");
      if (!button) return;
      const kind = button.dataset.kind === "loss" ? "loss" : "prediction";
      this.store.dispatch({ type: "fieldKind", fieldKind: kind });
      this.populateFieldSelect(kind);
      void this.loadRaster(this.dom.fieldSelect.value);
    });
    this.dom.signButtons.addEventListener("click", (event) => {
      const button = (event.target as Element).closest<HTMLButtonElement>("button[data-sign]");
      if (!button) return;
      const sign = button.dataset.sign === "pos" || button.dataset.sign === "neg" ? button.dataset.sign : "abs";
      this.store.dispatch({ type: "sign", sign });
      this.setActiveButtons(this.dom.signButtons, sign, "sign");
      void this.loadInfluenceRow().then(() => this.schedule("local"));
    });
    this.dom.kSlider.addEventListener("input", () => {
      this.store.dispatch({ type: "k", k: Number(this.dom.kSlider.value) });
      this.dom.kOutput.value = String(this.store.state.k);
      this.schedule("local");
    });
    this.dom.summarySelect.addEventListener("change", () => {
      this.store.dispatch({ type: "summary", summary: this.dom.summarySelect.value as SummaryName });
      void this.loadSummary().then(() => this.schedule("global"));
    });
    this.dom.mobileTabs.addEventListener("click", (event) => {
      const button = (event.target as Element).closest<HTMLButtonElement>("button[data-tab]");
      if (!button) return;
      const tab = button.dataset.tab === "global" ? "global" : "local";
      this.store.dispatch({ type: "mobileTab", mobileTab: tab });
      this.setActiveButtons(this.dom.mobileTabs, tab, "tab");
      this.dom.localPanel.dataset.mobileActive = String(tab === "local");
      this.dom.globalPanel.dataset.mobileActive = String(tab === "global");
      this.schedule(tab);
    });
    this.dom.resetButton.addEventListener("click", () => {
      this.store.dispatch({ type: "resetSelection" });
      this.pickDefaultSelection();
      void this.loadInfluenceRow().then(() => {
        this.schedule("main");
        this.schedule("local");
        this.updateStats();
      });
    });
    this.dom.mainCanvas.addEventListener("pointerdown", (event) => this.handleMainPointer(event));
  }

  private observeLayout(): void {
    const observer = new ResizeObserver(() => {
      this.schedule("main");
      this.schedule("local");
      this.schedule("global");
    });
    observer.observe(this.dom.mainCanvas);
    observer.observe(this.dom.influenceCanvas);
    observer.observe(this.dom.globalCanvas);
  }

  private populateRunSelect(): void {
    if (!this.index) return;
    this.dom.runSelect.replaceChildren(
      ...this.index.runs
        .filter((run) => run.manifest)
        .map((run) => new Option(`${run.display_name} (${run.problem})`, run.run_id)),
    );
  }

  private async loadRun(run: IndexRunEntry): Promise<void> {
    if (!run.manifest) return;
    this.prefetcher?.stop();
    this.prefetcher = null;
    this.repo?.abortBackground();
    showMessage(this.dom.message, null);
    this.dom.runMeta.textContent = `Loading ${run.display_name}`;
    this.store.dispatch({ type: "run", runId: run.run_id });
    this.dom.runSelect.value = run.run_id;
    this.manifestUrl = new URL(run.manifest, this.indexUrl);
    this.manifest = await loadRunManifest(this.indexUrl, run.manifest);
    this.repo = new DataRepository(this.manifestUrl, this.manifest);
    this.points = await this.repo.loadPointArrays();
    this.candidateDelaunay = buildDelaunay(
      this.points.candidate_points,
      this.manifest.arrays.candidate_points.shape[1] ?? 2,
    );
    this.trainDelaunay = buildDelaunay(
      this.points.train_points,
      this.manifest.arrays.train_points.shape[1] ?? 2,
    );
    this.populateControls();
    this.pickDefaultSelection();
    await Promise.all([this.loadRaster(this.store.state.fieldId), this.loadInfluenceRow(), this.loadSummary()]);
    this.dom.runMeta.textContent = `${this.manifest.display_name} · ${this.manifest.n_candidate.toLocaleString()} candidate · ${this.manifest.n_train.toLocaleString()} train`;
    this.schedule("main");
    this.schedule("local");
    this.schedule("global");
    this.updateStats();
    this.prefetcher = new RunPrefetcher(this.repo, this.manifest);
    this.startBackgroundPrefetch();
  }

  private populateControls(): void {
    if (!this.manifest) return;
    const firstField =
      this.manifest.default_field ??
      Object.entries(this.manifest.fields).find(([, field]) => field.kind === "prediction")?.[0] ??
      Object.keys(this.manifest.fields)[0] ??
      null;
    const fieldKind = firstField ? this.manifest.fields[firstField]?.kind ?? "prediction" : "prediction";
    this.store.dispatch({ type: "fieldKind", fieldKind });
    this.populateFieldSelect(fieldKind);
    if (firstField) {
      this.dom.fieldSelect.value = firstField;
      this.store.dispatch({ type: "field", fieldId: firstField });
    }

    const matrices = this.manifest.influence_matrices;
    this.dom.matrixSelect.replaceChildren(
      ...matrices.map((matrix) => new Option(matrix.display_label || matrix.label, matrix.id)),
    );
    const matrixId =
      this.manifest.default_matrix ??
      matrices.find((matrix) => matrix.id === DEFAULT_MATRIX_ID)?.id ??
      matrices[0]?.id ??
      null;
    if (matrixId) {
      this.dom.matrixSelect.value = matrixId;
      this.store.dispatch({ type: "matrix", matrixId });
    }

    this.dom.summarySelect.replaceChildren(
      ...(Object.entries(SUMMARY_LABELS) as [SummaryName, string][]).map(
        ([name, label]) => new Option(label, name),
      ),
    );
    this.dom.summarySelect.value = this.store.state.summary;
    const maxK = Math.max(1, Math.min(64, this.selectedMatrix()?.max_local_influence_points ?? this.manifest.max_local_influence_points));
    this.dom.kSlider.max = String(maxK);
    this.dom.kSlider.value = String(Math.min(this.store.state.k, maxK));
    this.store.dispatch({ type: "k", k: Number(this.dom.kSlider.value) });
    this.dom.kOutput.value = String(this.store.state.k);
    this.setActiveButtons(this.dom.fieldKindButtons, fieldKind, "kind");
    this.setActiveButtons(this.dom.signButtons, this.store.state.sign, "sign");
    this.setActiveButtons(this.dom.mobileTabs, this.store.state.mobileTab, "tab");
  }

  private populateFieldSelect(kind: "prediction" | "loss"): void {
    if (!this.manifest) return;
    const options = Object.entries(this.manifest.fields)
      .filter(([, field]) => field.kind === kind)
      .map(([id, field]) => new Option(field.label, id));
    if (!options.length) {
      options.push(...Object.entries(this.manifest.fields).map(([id, field]) => new Option(field.label, id)));
    }
    this.dom.fieldSelect.replaceChildren(...options);
    const selected = this.store.state.fieldId;
    if (selected && options.some((option) => option.value === selected)) {
      this.dom.fieldSelect.value = selected;
    }
    this.store.dispatch({ type: "field", fieldId: this.dom.fieldSelect.value });
    this.setActiveButtons(this.dom.fieldKindButtons, kind, "kind");
  }

  private setActiveButtons(group: HTMLElement, value: string, datasetName: string): void {
    for (const button of group.querySelectorAll<HTMLButtonElement>("button")) {
      button.classList.toggle("active", button.dataset[datasetName] === value);
    }
  }

  private pickDefaultSelection(): void {
    if (!this.manifest || !this.points) return;
    const candidateDim = this.manifest.arrays.candidate_points.shape[1] ?? 2;
    const trainDim = this.manifest.arrays.train_points.shape[1] ?? 2;
    const candidateIndex = clampIndex(0, this.manifest.n_candidate);
    const coord = pointAt(this.points.candidate_points, candidateIndex, candidateDim);
    const trainIndex = this.trainDelaunay?.find(coord[0], coord[1]) ?? clampIndex(0, this.manifest.n_train);
    this.store.dispatch({ type: "selection", candidateIndex, trainIndex, coord });
  }

  private selectedMatrix(): InfluenceMatrixManifest | null {
    if (!this.manifest || !this.store.state.matrixId) return null;
    return this.manifest.influence_matrices.find((matrix) => matrix.id === this.store.state.matrixId) ?? null;
  }

  private selectedRowIndex(matrix: InfluenceMatrixManifest): number {
    return matrix.row_source === "train_points"
      ? clampIndex(this.store.state.selectedTrainIndex, matrix.row_count)
      : clampIndex(this.store.state.selectedCandidateIndex, matrix.row_count);
  }

  private async loadRaster(fieldId: string | null): Promise<void> {
    if (!this.repo || !this.manifest || !fieldId) return;
    this.raster = await this.repo.loadRaster(fieldId, "foreground");
    await this.renderRasterWithWorker();
    this.dom.mainTitle.textContent = this.manifest.fields[fieldId]?.label ?? "Field";
    const domain = this.manifest.fields[fieldId]?.display_domain;
    this.dom.mainRange.textContent = domain ? `${formatNumber(domain[0])} … ${formatNumber(domain[1])}` : "";
    this.schedule("main");
    this.updateStats();
    this.updatePrefetchPlan();
  }

  private renderRasterWithWorker(): Promise<void> {
    if (!this.raster) return Promise.resolve();
    const requestId = ++this.latestRasterRequest;
    const request: RasterWorkerRequest = {
      requestId,
      width: this.raster.width,
      height: this.raster.height,
      values: this.raster.values,
      mask: this.raster.mask,
      encoding: this.raster.encoding,
      displayDomain: this.raster.displayDomain,
    };
    return new Promise((resolve) => {
      const onResponse = (event: MessageEvent<RasterWorkerResponse>) => {
        if (event.data.requestId !== requestId) return;
        this.worker.removeEventListener("message", onResponse);
        this.handleRasterWorkerResponse(event.data);
        resolve();
      };
      this.worker.addEventListener("message", onResponse);
      this.worker.postMessage(request);
    });
  }

  private handleRasterWorkerResponse(response: RasterWorkerResponse): void {
    if (response.requestId !== this.latestRasterRequest || !this.raster) return;
    const imageCanvas = document.createElement("canvas");
    imageCanvas.width = response.width;
    imageCanvas.height = response.height;
    const imageCtx = imageCanvas.getContext("2d");
    if (!imageCtx) return;
    const imageData = imageCtx.createImageData(response.width, response.height);
    imageData.data.set(response.rgba);
    imageCtx.putImageData(imageData, 0, 0);
    this.rasterResult = {
      image: imageCanvas,
      decoded: response.decoded,
      contourValues: response.contourValues,
      contourPaths: response.contourPaths,
    };
    this.schedule("main");
    this.updateStats();
  }

  private async loadInfluenceRow(): Promise<void> {
    const matrix = this.selectedMatrix();
    if (!this.repo || !matrix) return;
    this.influenceRow = await this.repo.loadInfluenceRow(
      matrix,
      this.store.state.sign,
      this.selectedRowIndex(matrix),
      "foreground",
    );
    this.updatePrefetchPlan();
  }

  private async loadSummary(): Promise<void> {
    const matrix = this.selectedMatrix();
    if (!this.repo || !matrix) return;
    this.summaryValues = await this.repo.loadSummary(matrix, this.store.state.summary, "foreground");
    this.updatePrefetchPlan();
  }

  private startBackgroundPrefetch(): void {
    const prefetcher = this.prefetcher;
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        if (prefetcher !== this.prefetcher) return;
        this.updatePrefetchPlan();
      });
    });
  }

  private updatePrefetchPlan(): void {
    const context = this.prefetchContext();
    if (!context || !this.prefetcher) return;
    this.prefetcher.update(context);
  }

  private prefetchContext(): PrefetchContext | null {
    if (!this.manifest) return null;
    return {
      fieldId: this.store.state.fieldId,
      fieldKind: this.store.state.fieldKind as FieldKind,
      matrixId: this.store.state.matrixId,
      sign: this.store.state.sign,
      summary: this.store.state.summary,
      selectedCandidateIndex: this.store.state.selectedCandidateIndex,
      selectedTrainIndex: this.store.state.selectedTrainIndex,
    };
  }

  private context(): PlotContext | null {
    if (!this.manifest || !this.points) return null;
    const candidateDim = this.manifest.arrays.candidate_points.shape[1] ?? 2;
    const trainDim = this.manifest.arrays.train_points.shape[1] ?? 2;
    const bounds = Object.keys(this.manifest.bounds ?? {}).length
      ? boundsFromAxisMap(this.manifest.bounds, this.manifest.axes)
      : inferPointBounds(this.points.candidate_points, candidateDim);
    return {
      manifest: this.manifest,
      points: this.points,
      bounds,
      candidateDim,
      trainDim,
    };
  }

  private schedule(panel: PanelName): void {
    if (this.scheduled.has(panel)) return;
    this.scheduled.add(panel);
    requestAnimationFrame(() => {
      this.scheduled.delete(panel);
      this.render(panel);
    });
  }

  private render(panel: PanelName): void {
    const context = this.context();
    if (!context) return;
    if (panel === "main") {
      this.mainViewport = renderMainPlot({
        canvas: this.dom.mainCanvas,
        svg: this.dom.mainSvg,
        context,
        raster: this.raster,
        rasterResult: this.rasterResult,
        selectedCoord: this.store.state.selectedCoord,
        showCandidatePoints: true,
        showTrainPoints: true,
      });
      return;
    }
    if (panel === "local") {
      const matrix = this.selectedMatrix();
      if (!matrix) return;
      const maxAbs = renderLocalInfluencePlot({
        canvas: this.dom.influenceCanvas,
        svg: this.dom.influenceSvg,
        context,
        matrix,
        row: this.influenceRow,
        selectedCandidateIndex: this.store.state.selectedCandidateIndex,
        selectedTrainIndex: this.store.state.selectedTrainIndex,
        k: this.store.state.k,
      });
      this.dom.influenceRange.textContent = maxAbs ? `max |I| ${formatNumber(maxAbs)}` : "";
      return;
    }
    const domain = renderGlobalPlot({
      canvas: this.dom.globalCanvas,
      svg: this.dom.globalSvg,
      context,
      values: this.summaryValues,
      diverging:
        this.store.state.summary === "mean_signed" ||
        this.store.state.summary === "negative_mass",
    });
    this.dom.globalRange.textContent = `${formatNumber(domain[0])} … ${formatNumber(domain[1])}`;
  }

  private handleMainPointer(event: PointerEvent): void {
    const context = this.context();
    if (!context || !this.mainViewport) return;
    const domain = pointerInDomain(event, this.dom.mainCanvas, context.bounds, this.mainViewport);
    if (!domain) return;
    const matrix = this.selectedMatrix();
    const candidateIndex =
      matrix?.row_source === "train_points"
        ? this.store.state.selectedCandidateIndex
        : (this.candidateDelaunay?.find(domain[0], domain[1]) ?? 0);
    const trainIndex =
      matrix?.row_source === "train_points"
        ? (this.trainDelaunay?.find(domain[0], domain[1]) ?? 0)
        : (this.trainDelaunay?.find(domain[0], domain[1]) ?? this.store.state.selectedTrainIndex);
    const coord =
      matrix?.row_source === "train_points"
        ? pointAt(context.points.train_points, trainIndex, context.trainDim)
        : pointAt(context.points.candidate_points, candidateIndex, context.candidateDim);
    this.store.dispatch({ type: "selection", candidateIndex, trainIndex, coord });
    void this.loadInfluenceRow().then(() => {
      this.schedule("main");
      this.schedule("local");
      this.updateStats();
    });
  }

  private updateStats(): void {
    if (!this.manifest) return;
    const context = this.context();
    const rasterBounds = this.manifest.field_raster
      ? boundsFromAxisMap(this.manifest.field_raster.bounds, this.manifest.field_raster.axes)
      : context?.bounds;
    const sample = rasterBounds
      ? rasterSampleAtCoord(this.raster, this.rasterResult?.decoded ?? null, rasterBounds, this.store.state.selectedCoord)
      : null;
    const [x, y] = sample
      ? [sample.x, sample.y]
      : (this.store.state.selectedCoord ?? [Number.NaN, Number.NaN]);
    this.dom.selectedPoint.textContent = `(${formatNumber(x)}, ${formatNumber(y)})`;
    this.dom.selectedValue.textContent = formatNumber(sample?.value);
    this.dom.trainCount.textContent = this.manifest.n_train.toLocaleString();
    this.dom.candidateCount.textContent = this.manifest.n_candidate.toLocaleString();
  }
}
