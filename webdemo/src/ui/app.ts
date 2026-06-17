import type { Delaunay } from "d3";
import type {
  AppView,
  BackgroundMode,
  Bounds,
  DataIndex,
  IndexVariantEntry,
  InfluenceAggregate,
  InfluenceMatrixManifest,
  InfluenceRow,
  ModelQuality,
  PointArrays,
  PlotViewport,
  RasterData,
  ResultsData,
  RunManifest,
  TypedArray,
} from "../types";
import { DataRepository } from "../data/arrays";
import { LruCache } from "../data/cache";
import { loadResultsData } from "../data/results";
import {
  firstAvailableProblem,
  formatProblemLabel,
  loadIndex,
  loadRunManifest,
  qualityLabel,
  resolveIndexUrl,
  resolveProblemVariant,
} from "../data/manifest";
import { RunPrefetcher, type PrefetchContext } from "../data/prefetcher";
import { MAX_TOP_K, Store } from "../state/store";
import {
  captureVariantState as captureRestorableVariantState,
  denormalizeSelection,
  orderedFieldEntries,
  resolveRestoredFieldId,
  resolveRestoredMatrixId,
  type VariantStateSnapshot,
} from "../state/variantRestore";
import type { RasterWorkerRequest, RasterWorkerResponse } from "../worker/rasterWorker";
import type { PlotColorbarPlacement } from "../viz/chrome";
import {
  boundsFromAxisMap,
  clampIndex,
  containsViewportPoint,
  domainAspectRatio,
  inferPointBounds,
  pointAt,
  selectPointIndicesInBounds,
} from "../viz/geometry";
import { chooseAdaptivePlotLayout } from "../viz/layout";
import {
  buildDelaunay,
  pointerInDomain,
  rasterSampleAtCoord,
  renderLocalInfluencePlot,
  renderMainPlot,
  renderRegionalInfluencePlot,
  selectionPulseProgress,
  type PlotContext,
  type RasterRenderResult,
} from "../viz/plots";
import {
  plotProjectionForManifest,
  projectPointToDisplay,
  regionBoundsFromProjectedViewportDrag,
} from "../viz/projection";
import {
  formatFieldSelectLabel,
  formatInfluenceMatrixLabel,
  formatNumber,
  formatRegionReadoutNumber,
  formatReadoutNumber,
  getDomRefs,
  showMessage,
  type DomRefs,
} from "./dom";
import { ResultsDashboard } from "./results";

const DEFAULT_MATRIX_ID = "influences_total_loss_output_0";
const BACKGROUND_MODE_LABELS: Record<BackgroundMode, string> = {
  points: "Points",
  smooth: "Smooth",
  cell: "Cells",
};
const DRAG_THRESHOLD_PX = 8;
const DOUBLE_TAP_MS = 350;
const DOUBLE_TAP_DISTANCE_PX = 36;
const MODEL_INTERACTION_HINT_VISIBLE_MS = 2500;
const MODEL_INTERACTION_HINT_HIDE_MS = 440;
const MODEL_INTERACTION_HINT_REDUCED_HIDE_MS = 1;

type PanelName = "main" | "train";
type ModelGesture = {
  pointerId: number;
  pointerType: string;
  start: [number, number];
  current: [number, number];
  mode: "pending" | "region";
};
type PlotTarget = "main" | "train";
type PlotClickGesture = {
  pointerId: number;
  start: [number, number];
};

export class AppController {
  private readonly dom: DomRefs = getDomRefs();
  private readonly store = new Store();
  private readonly resultsDashboard = new ResultsDashboard(this.dom.resultsWorkspace);
  private readonly indexUrl = resolveIndexUrl();
  private readonly worker = new Worker(new URL("../worker/rasterWorker.ts", import.meta.url), {
    type: "module",
  });
  private readonly arrayCache = new LruCache<TypedArray>();
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
  private influenceAggregate: InfluenceAggregate | null = null;
  private mainViewport: PlotViewport | null = null;
  private trainViewport: PlotViewport | null = null;
  private draftRegion: Bounds | null = null;
  private modelGesture: ModelGesture | null = null;
  private trainGesture: PlotClickGesture | null = null;
  private touchRegionArmed = false;
  private lastTouchTap: { time: number; point: [number, number] } | null = null;
  private latestRasterRequest = 0;
  private latestAggregateRequest = 0;
  private scheduled = new Set<PanelName>();
  private resultsData: ResultsData | null = null;
  private resultsLoadPromise: Promise<ResultsData> | null = null;
  private scheduledResultsRender = 0;
  private lastLayoutSignature = "";
  private selectionPulseStartedAt = 0;
  private selectionPulseAnimation = 0;
  private modelInteractionHintShown = false;
  private modelInteractionHintAutoTimer = 0;
  private modelInteractionHintHideTimer = 0;
  private modelInteractionHintShowFrame = 0;

  async start(): Promise<void> {
    this.bindEvents();
    this.observeLayout();
    showMessage(this.dom.message, null);
    this.setActiveButtons(this.dom.viewButtons, this.store.state.appView, "appView");
    this.applyActiveView();
    this.dom.runMeta.textContent = "Loading data index";
    try {
      this.index = await loadIndex(this.indexUrl);
      this.populateProblemSelect();
      const firstProblem = firstAvailableProblem(this.index);
      if (!firstProblem) {
        throw new Error("No complete v7 problem manifest is available");
      }
      this.store.dispatch({ type: "problem", problem: firstProblem.problem });
      this.dom.problemSelect.value = firstProblem.problem;
      this.setActiveButtons(this.dom.qualityButtons, this.store.state.modelQuality, "modelQuality");
      await this.loadActiveVariant();
      if (this.store.state.appView === "results") {
        await this.ensureResultsData();
        this.renderResultsDashboard();
      }
    } catch (error) {
      showMessage(this.dom.message, error instanceof Error ? error.message : String(error));
      this.dom.runMeta.textContent = "Data unavailable";
    }
  }

  private bindEvents(): void {
    this.dom.viewButtons.addEventListener("click", (event) => {
      const button = (event.target as Element).closest<HTMLButtonElement>("button[data-app-view]");
      if (!button) return;
      const appView: AppView = button.dataset.appView === "results" ? "results" : "playground";
      this.store.dispatch({ type: "view", appView });
      this.applyActiveView();
    });
    this.dom.problemSelect.addEventListener("change", () => {
      this.store.dispatch({ type: "problem", problem: this.dom.problemSelect.value });
      if (this.store.state.appView === "results") {
        this.renderResultsDashboard();
        return;
      }
      void this.loadActiveVariant();
    });
    this.dom.qualityButtons.addEventListener("click", (event) => {
      const button = (event.target as Element).closest<HTMLButtonElement>("button[data-model-quality]");
      if (!button) return;
      const quality: ModelQuality = button.dataset.modelQuality === "bad" ? "bad" : "good";
      this.store.dispatch({ type: "modelQuality", modelQuality: quality });
      this.setActiveButtons(this.dom.qualityButtons, quality, "modelQuality");
      if (this.store.state.appView === "results") {
        this.updateToplineMeta();
        return;
      }
      void this.loadActiveVariant();
    });
    this.dom.fieldSelect.addEventListener("change", () => {
      const fieldId = this.dom.fieldSelect.value;
      this.store.dispatch({ type: "field", fieldId });
      void this.loadRaster(fieldId);
    });
    this.dom.matrixSelect.addEventListener("change", () => {
      this.store.dispatch({ type: "matrix", matrixId: this.dom.matrixSelect.value });
      void this.loadInfluenceForSelection().then(() => {
        this.schedule("train");
        this.updateStats();
      });
    });
    this.dom.signButtons.addEventListener("click", (event) => {
      const button = (event.target as Element).closest<HTMLButtonElement>("button[data-sign]");
      if (!button) return;
      const sign = button.dataset.sign === "pos" || button.dataset.sign === "neg" ? button.dataset.sign : "abs";
      this.store.dispatch({ type: "sign", sign });
      this.setActiveButtons(this.dom.signButtons, sign, "sign");
      void this.loadInfluenceForSelection().then(() => {
        this.schedule("train");
        this.updateStats();
      });
    });
    this.dom.kSlider.addEventListener("input", () => {
      this.store.dispatch({ type: "k", k: Number(this.dom.kSlider.value) });
      this.dom.kOutput.value = String(this.store.state.k);
      this.updateRangeProgress();
      this.schedule("train");
    });
    this.dom.backgroundButtons.addEventListener("click", (event) => {
      const button = (event.target as Element).closest<HTMLButtonElement>("button[data-background-mode]");
      if (!button) return;
      const value = button.dataset.backgroundMode;
      const backgroundMode: BackgroundMode =
        value === "smooth" || value === "cell" ? value : "points";
      this.store.dispatch({ type: "backgroundMode", backgroundMode });
      this.setActiveButtons(this.dom.backgroundButtons, backgroundMode, "backgroundMode");
      this.refreshResponsiveLayout();
      this.schedule("train");
    });
    this.dom.resetButton.addEventListener("click", () => {
      this.store.dispatch({ type: "resetSelection" });
      this.draftRegion = null;
      this.modelGesture = null;
      this.trainGesture = null;
      this.touchRegionArmed = false;
      this.lastTouchTap = null;
      this.influenceAggregate = null;
      this.pickDefaultSelection();
      this.triggerSelectionPulse();
      void this.loadInfluenceForSelection().then(() => {
        this.schedule("main");
        this.schedule("train");
        this.updateStats();
      });
    });
    this.dom.mainCanvas.addEventListener("pointerdown", (event) => this.handleMainPointerDown(event));
    this.dom.mainCanvas.addEventListener("pointermove", (event) => this.handleMainPointerMove(event));
    this.dom.mainCanvas.addEventListener("pointerup", (event) => this.handleMainPointerUp(event));
    this.dom.mainCanvas.addEventListener("pointercancel", (event) => this.handleMainPointerCancel(event));
    this.dom.mainCanvas.addEventListener("contextmenu", (event) => event.preventDefault());
    this.dom.trainCanvas.addEventListener("pointerdown", (event) => this.handleTrainPointerDown(event));
    this.dom.trainCanvas.addEventListener("pointerup", (event) => this.handleTrainPointerUp(event));
    this.dom.trainCanvas.addEventListener("pointercancel", (event) => this.handleTrainPointerCancel(event));
    this.dom.trainCanvas.addEventListener("contextmenu", (event) => event.preventDefault());
    window.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      this.clearRegionSelection();
    });
    window.addEventListener("resize", () => this.handleViewportScaleChange());
    window.visualViewport?.addEventListener("resize", () => this.handleViewportScaleChange());
    this.observeDevicePixelRatio();
  }

  private setPanelLoading(panel: HTMLElement, loading: boolean): void {
    panel.dataset.loading = loading ? "true" : "false";
    panel.setAttribute("aria-busy", String(loading));
  }

  private updateRangeProgress(): void {
    const min = Number(this.dom.kSlider.min);
    const max = Number(this.dom.kSlider.max);
    const value = Number(this.dom.kSlider.value);
    const span = Number.isFinite(max - min) && max > min ? max - min : 1;
    const progress = Math.max(0, Math.min(100, ((value - min) / span) * 100));
    this.dom.kSlider.style.setProperty("--range-progress", `${progress}%`);
  }

  private applyActiveView(): void {
    const isResults = this.store.state.appView === "results";
    this.dom.playgroundWorkspace.hidden = isResults;
    this.dom.resultsWorkspace.hidden = !isResults;
    this.dom.resetButton.hidden = isResults;
    this.setActiveButtons(this.dom.viewButtons, this.store.state.appView, "appView");
    this.updateToplineMeta();
    if (isResults) {
      this.dismissModelInteractionHint();
      void this.ensureResultsData()
        .then(() => this.renderResultsDashboard())
        .catch(() => undefined);
      return;
    }
    if (this.index && !this.activeVariantMatchesState()) {
      void this.loadActiveVariant();
      return;
    }
    this.refreshResponsiveLayout();
    this.schedule("main");
    this.schedule("train");
  }

  private async ensureResultsData(): Promise<ResultsData | null> {
    if (this.resultsData) return this.resultsData;
    if (!this.resultsLoadPromise) {
      this.resultsDashboard.setLoading();
      this.resultsLoadPromise = loadResultsData(this.indexUrl)
        .then((data) => {
          this.resultsData = data;
          return data;
        })
        .catch((error) => {
          this.resultsLoadPromise = null;
          const message = error instanceof Error ? error.message : String(error);
          this.resultsDashboard.setError(message);
          throw error;
        });
    }
    return this.resultsLoadPromise;
  }

  private renderResultsDashboard(): void {
    if (this.store.state.appView !== "results" || !this.resultsData || !this.index) return;
    this.resultsDashboard.render(this.resultsData, this.store.state.problem ?? this.dom.problemSelect.value);
    this.updateToplineMeta();
  }

  private scheduleResultsRender(): void {
    if (this.scheduledResultsRender) return;
    this.scheduledResultsRender = requestAnimationFrame(() => {
      this.scheduledResultsRender = 0;
      this.resultsDashboard.rerender();
    });
  }

  private updateToplineMeta(): void {
    if (this.store.state.appView === "results") {
      const problemLabel = this.dom.problemSelect.selectedOptions[0]?.textContent ?? "Selected problem";
      this.dom.runMeta.textContent = `Results · ${problemLabel}`;
      return;
    }
    if (!this.manifest) return;
    this.dom.runMeta.textContent = `${formatProblemLabel(this.manifest.display_name)} · ${qualityLabel(this.manifest.model_quality)} · ${this.manifest.n_candidate.toLocaleString()} candidate · ${this.manifest.n_train.toLocaleString()} train`;
  }

  private activeVariantMatchesState(): boolean {
    const problemId = this.store.state.problem ?? this.dom.problemSelect.value;
    return Boolean(
      this.manifest &&
        problemId &&
        this.manifest.problem === problemId &&
        this.manifest.model_quality === this.store.state.modelQuality,
    );
  }

  private prefersReducedMotion(): boolean {
    return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
  }

  private currentSelectionPulse(): number {
    return this.prefersReducedMotion()
      ? 0
      : selectionPulseProgress(this.selectionPulseStartedAt);
  }

  private triggerSelectionPulse(): void {
    if (this.prefersReducedMotion()) return;
    this.selectionPulseStartedAt = performance.now();
    if (this.selectionPulseAnimation) return;

    const tick = () => {
      const pulse = this.currentSelectionPulse();
      this.schedule("main");
      this.schedule("train");
      if (pulse > 0) {
        this.selectionPulseAnimation = requestAnimationFrame(tick);
        return;
      }
      this.selectionPulseAnimation = 0;
      this.selectionPulseStartedAt = 0;
      this.schedule("main");
      this.schedule("train");
    };
    this.selectionPulseAnimation = requestAnimationFrame(tick);
  }

  private maybeShowModelInteractionHint(): void {
    if (this.modelInteractionHintShown || !this.mainViewport || !this.rasterResult) return;
    this.modelInteractionHintShown = true;
    this.showModelInteractionHint();
  }

  private showModelInteractionHint(): void {
    const hint = this.dom.modelInteractionHint;
    this.clearModelInteractionHintTimers();
    hint.hidden = false;
    hint.setAttribute("aria-hidden", "false");
    hint.dataset.state = "hidden";

    this.modelInteractionHintShowFrame = requestAnimationFrame(() => {
      this.modelInteractionHintShowFrame = 0;
      hint.dataset.state = "visible";
      this.modelInteractionHintAutoTimer = window.setTimeout(
        () => this.dismissModelInteractionHint(),
        MODEL_INTERACTION_HINT_VISIBLE_MS,
      );
    });
  }

  private dismissModelInteractionHint(): void {
    const hint = this.dom.modelInteractionHint;
    if (hint.hidden && !this.modelInteractionHintShowFrame) return;
    this.clearModelInteractionHintTimers();
    hint.dataset.state = "hidden";
    hint.setAttribute("aria-hidden", "true");
    this.modelInteractionHintHideTimer = window.setTimeout(
      () => {
        this.modelInteractionHintHideTimer = 0;
        hint.hidden = true;
      },
      this.prefersReducedMotion()
        ? MODEL_INTERACTION_HINT_REDUCED_HIDE_MS
        : MODEL_INTERACTION_HINT_HIDE_MS,
    );
  }

  private clearModelInteractionHintTimers(): void {
    if (this.modelInteractionHintAutoTimer) {
      window.clearTimeout(this.modelInteractionHintAutoTimer);
      this.modelInteractionHintAutoTimer = 0;
    }
    if (this.modelInteractionHintHideTimer) {
      window.clearTimeout(this.modelInteractionHintHideTimer);
      this.modelInteractionHintHideTimer = 0;
    }
    if (this.modelInteractionHintShowFrame) {
      window.cancelAnimationFrame(this.modelInteractionHintShowFrame);
      this.modelInteractionHintShowFrame = 0;
    }
  }

  private updateTrainControlVisibility(): void {
    this.dom.kControl.hidden = false;
    this.setActiveButtons(this.dom.backgroundButtons, this.store.state.backgroundMode, "backgroundMode");
  }

  private refreshResponsiveLayout(): boolean {
    return this.applyAdaptiveLayout();
  }

  private observeLayout(): void {
    const observer = new ResizeObserver(() => {
      this.handleViewportScaleChange();
    });
    observer.observe(this.dom.plotGrid);
    observer.observe(this.dom.modelPanel);
    observer.observe(this.dom.trainPanel);
  }

  private handleViewportScaleChange(): void {
    if (this.store.state.appView === "results") {
      this.scheduleResultsRender();
      return;
    }
    this.refreshResponsiveLayout();
    this.schedule("main");
    this.schedule("train");
  }

  private observeDevicePixelRatio(): void {
    if (!window.matchMedia) return;
    let query: MediaQueryList | null = null;
    const handleChange = () => {
      query?.removeEventListener("change", handleChange);
      this.handleViewportScaleChange();
      bindQuery();
    };
    const bindQuery = () => {
      query = window.matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
      query.addEventListener("change", handleChange);
    };
    bindQuery();
  }

  private populateProblemSelect(): void {
    if (!this.index) return;
    this.dom.problemSelect.replaceChildren(
      ...this.index.problems.map(
        (problem) => new Option(formatProblemLabel(problem.display_name || problem.problem), problem.problem),
      ),
    );
  }

  private async loadActiveVariant(): Promise<void> {
    if (!this.index) return;
    const problemId = this.store.state.problem ?? this.dom.problemSelect.value;
    const variant = resolveProblemVariant(this.index, problemId, this.store.state.modelQuality);
    if (!variant?.manifest) {
      const label = formatProblemLabel(problemId ?? "selected problem");
      throw new Error(`${label} has no ${qualityLabel(this.store.state.modelQuality)} manifest`);
    }
    await this.loadVariant(variant);
  }

  private async loadVariant(variant: IndexVariantEntry): Promise<void> {
    if (!variant.manifest) return;
    this.setPanelLoading(this.dom.modelPanel, true);
    this.setPanelLoading(this.dom.trainPanel, true);
    this.dismissModelInteractionHint();
    const snapshot = this.captureVariantStateSnapshot();
    this.prefetcher?.stop();
    this.prefetcher = null;
    this.repo?.abortBackground();
    this.raster = null;
    this.rasterResult = null;
    this.influenceRow = null;
    this.influenceAggregate = null;
    this.mainViewport = null;
    this.trainViewport = null;
    this.draftRegion = null;
    this.modelGesture = null;
    this.trainGesture = null;
    this.touchRegionArmed = false;
    this.lastTouchTap = null;
    this.latestAggregateRequest += 1;
    showMessage(this.dom.message, null);
    this.dom.runMeta.textContent = `Loading ${variant.display_name} · ${qualityLabel(variant.model_quality)}`;
    this.dom.problemSelect.value = variant.problem;
    this.setActiveButtons(this.dom.qualityButtons, variant.model_quality, "modelQuality");
    try {
      this.manifestUrl = new URL(variant.manifest, this.indexUrl);
      this.manifest = await loadRunManifest(this.indexUrl, variant.manifest);
      this.repo = new DataRepository(this.manifestUrl, this.manifest, this.arrayCache);
      this.points = await this.repo.loadPointArrays();
      const context = this.context();
      const projection = context ? this.mainProjection(context) : undefined;
      this.candidateDelaunay = buildDelaunay(
        this.points.candidate_points,
        this.manifest.arrays.candidate_points.shape[1] ?? 2,
        projection,
      );
      this.trainDelaunay = buildDelaunay(
        this.points.train_points,
        this.manifest.arrays.train_points.shape[1] ?? 2,
        projection,
      );
      this.populateControls(snapshot);
      this.restoreSelection(snapshot);
      await Promise.all([
        this.loadRaster(this.store.state.fieldId),
        this.loadInfluenceForSelection(),
      ]);
      this.updateToplineMeta();
      this.refreshResponsiveLayout();
      if (this.store.state.selectionMode === "point") this.triggerSelectionPulse();
      this.schedule("main");
      this.schedule("train");
      this.updateStats();
      this.prefetcher = new RunPrefetcher(this.repo, this.manifest);
      this.startBackgroundPrefetch();
    } finally {
      this.setPanelLoading(this.dom.modelPanel, false);
      this.setPanelLoading(this.dom.trainPanel, false);
    }
  }

  private captureVariantStateSnapshot(): VariantStateSnapshot | null {
    const context = this.context();
    const mainBounds = context ? this.mainPlotBounds(context) : null;
    return captureRestorableVariantState(this.store.state, this.manifest, mainBounds);
  }

  private populateControls(snapshot: VariantStateSnapshot | null): void {
    if (!this.manifest) return;
    const fieldId = this.populateFieldSelect(resolveRestoredFieldId(this.manifest, snapshot));
    this.store.dispatch({ type: "field", fieldId });

    const matrices = this.manifest.influence_matrices;
    this.dom.matrixSelect.replaceChildren(
      ...matrices.map(
        (matrix) => new Option(formatInfluenceMatrixLabel(matrix, this.manifest?.term_labels), matrix.id),
      ),
    );
    const matrixId = resolveRestoredMatrixId(this.manifest, snapshot, DEFAULT_MATRIX_ID);
    if (matrixId) {
      this.dom.matrixSelect.value = matrixId;
    }
    this.store.dispatch({ type: "matrix", matrixId });

    this.dom.kSlider.min = "0";
    this.dom.kSlider.max = String(MAX_TOP_K);
    this.dom.kSlider.value = String(Math.min(this.store.state.k, MAX_TOP_K));
    this.store.dispatch({ type: "k", k: Number(this.dom.kSlider.value) });
    this.dom.kOutput.value = String(this.store.state.k);
    this.updateRangeProgress();
    this.setActiveButtons(this.dom.signButtons, this.store.state.sign, "sign");
    this.setActiveButtons(this.dom.backgroundButtons, this.store.state.backgroundMode, "backgroundMode");
    this.updateTrainControlVisibility();
  }

  private populateFieldSelect(selectedFieldId: string | null): string | null {
    if (!this.manifest) return null;
    const orderedEntries = orderedFieldEntries(this.manifest);
    const options = orderedEntries.map(([id, field]) => new Option(formatFieldSelectLabel(field.label), id));
    this.dom.fieldSelect.replaceChildren(...options);
    if (selectedFieldId && options.some((option) => option.value === selectedFieldId)) {
      this.dom.fieldSelect.value = selectedFieldId;
    }
    return this.dom.fieldSelect.value || null;
  }

  private setActiveButtons(group: HTMLElement, value: string, datasetName: string): void {
    for (const button of group.querySelectorAll<HTMLButtonElement>("button")) {
      button.classList.toggle("active", button.dataset[datasetName] === value);
    }
  }

  private pickDefaultSelection(): void {
    if (!this.manifest || !this.points) return;
    const context = this.context();
    if (!context) return;
    const candidateDim = this.manifest.arrays.candidate_points.shape[1] ?? 2;
    const trainDim = this.manifest.arrays.train_points.shape[1] ?? 2;
    const candidateIndex = clampIndex(0, this.manifest.n_candidate);
    const coord = pointAt(this.points.candidate_points, candidateIndex, candidateDim);
    const displayCoord = this.projectedMainPoint(coord, context);
    const trainIndex =
      this.trainDelaunay?.find(displayCoord[0], displayCoord[1]) ??
      clampIndex(0, this.manifest.n_train);
    this.store.dispatch({ type: "selection", candidateIndex, trainIndex, coord });
  }

  private restoreSelection(snapshot: VariantStateSnapshot | null): void {
    if (!this.manifest || !this.points) return;
    const context = this.context();
    if (!context) return;
    const selection = denormalizeSelection(snapshot?.selection ?? null, this.mainPlotBounds(context));
    if (!selection) {
      this.pickDefaultSelection();
      return;
    }
    if (selection.mode === "region") {
      this.store.dispatch({ type: "regionSelection", region: selection.region, candidateIndices: [] });
      const matrix = this.selectedMatrix();
      if (matrix) this.refreshRegionRowSelection(matrix);
      return;
    }

    const coord = selection.coord;
    const displayCoord = this.projectedMainPoint(coord, context);
    const candidateIndex =
      this.candidateDelaunay?.find(displayCoord[0], displayCoord[1]) ??
      clampIndex(this.store.state.selectedCandidateIndex, this.manifest.n_candidate);
    const trainIndex =
      this.trainDelaunay?.find(displayCoord[0], displayCoord[1]) ??
      clampIndex(this.store.state.selectedTrainIndex, this.manifest.n_train);
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
    this.setPanelLoading(this.dom.modelPanel, true);
    try {
      this.raster = await this.repo.loadRaster(fieldId, "foreground");
      await this.renderRasterWithWorker();
      this.dom.mainTitle.textContent = "Model";
      this.dom.mainRange.textContent = "";
      if (this.refreshResponsiveLayout()) this.schedule("train");
      this.schedule("main");
      this.updateStats();
      this.updatePrefetchPlan();
    } finally {
      this.setPanelLoading(this.dom.modelPanel, false);
    }
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
    this.schedule("train");
    this.updateStats();
  }

  private async loadInfluenceRow(): Promise<void> {
    const matrix = this.selectedMatrix();
    if (!this.repo || !matrix) return;
    this.setPanelLoading(this.dom.trainPanel, true);
    try {
      this.influenceRow = await this.repo.loadInfluenceRow(
        matrix,
        this.store.state.sign,
        this.selectedRowIndex(matrix),
        "foreground",
      );
      this.updatePrefetchPlan();
    } finally {
      this.setPanelLoading(this.dom.trainPanel, false);
    }
  }

  private async loadInfluenceForSelection(): Promise<void> {
    if (this.store.state.selectionMode === "region") {
      await this.loadInfluenceAggregate();
      return;
    }
    this.latestAggregateRequest += 1;
    await this.loadInfluenceRow();
  }

  private async loadInfluenceAggregate(): Promise<void> {
    const matrix = this.selectedMatrix();
    if (!this.repo || !matrix) return;
    this.setPanelLoading(this.dom.trainPanel, true);
    const requestId = ++this.latestAggregateRequest;
    const rowIndices = this.refreshRegionRowSelection(matrix);
    if (!this.store.state.selectedRegion) {
      this.influenceAggregate = null;
      this.updatePrefetchPlan();
      this.setPanelLoading(this.dom.trainPanel, false);
      return;
    }
    this.influenceAggregate = null;
    this.schedule("train");
    try {
      const aggregate = await this.repo.loadInfluenceAggregate(
        matrix,
        this.store.state.sign,
        rowIndices,
        "foreground",
      );
      if (requestId !== this.latestAggregateRequest) return;
      this.influenceAggregate = aggregate;
      this.updatePrefetchPlan();
    } finally {
      this.setPanelLoading(this.dom.trainPanel, false);
    }
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
      matrixId: this.store.state.matrixId,
      sign: this.store.state.sign,
      selectedCandidateIndex: this.store.state.selectedCandidateIndex,
      selectedTrainIndex: this.store.state.selectedTrainIndex,
      selectionMode: this.store.state.selectionMode,
      selectedRegionCandidateIndices: this.store.state.selectedRegionCandidateIndices,
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

  private mainProjection(context: PlotContext) {
    return plotProjectionForManifest(context.manifest, this.mainPlotBounds(context));
  }

  private trainProjection(context: PlotContext) {
    return plotProjectionForManifest(context.manifest, context.bounds);
  }

  private projectedMainPoint(coord: [number, number], context: PlotContext): [number, number] {
    return projectPointToDisplay(coord, this.mainProjection(context));
  }

  private applyAdaptiveLayout(): boolean {
    const context = this.context();
    if (!context) return false;
    const rect = this.dom.plotGrid.getBoundingClientRect();
    if (rect.width <= 1 || rect.height <= 1) return false;
    const style = getComputedStyle(this.dom.plotGrid);
    const gap = Number.parseFloat(style.gap || style.columnGap) || 0;
    const headerHeight =
      Math.max(
        this.dom.mainTitle.closest(".plot-header")?.getBoundingClientRect().height ?? 0,
        this.dom.trainTitle.closest(".plot-header")?.getBoundingClientRect().height ?? 0,
      ) || 44;
    const layout = chooseAdaptivePlotLayout({
      width: rect.width,
      height: rect.height,
      gap,
      headerHeight,
      modelAspect: domainAspectRatio(this.mainProjection(context).displayBounds),
      trainAspect: domainAspectRatio(this.trainProjection(context).displayBounds),
    });
    const modelTrack = `${Math.max(1, Math.round(layout.modelTrackPx))}px`;
    const trainTrack = `${Math.max(1, Math.round(layout.trainTrackPx))}px`;
    const columns =
      layout.orientation === "row" ? `${modelTrack} ${trainTrack}` : "minmax(0, 1fr)";
    const rows =
      layout.orientation === "column" ? `${modelTrack} ${trainTrack}` : "minmax(0, 1fr)";
    const signature = `${layout.orientation}|${columns}|${rows}`;
    if (signature === this.lastLayoutSignature) return false;
    this.lastLayoutSignature = signature;
    this.dom.plotGrid.dataset.layout = layout.orientation;
    this.dom.plotGrid.style.gridTemplateColumns = columns;
    this.dom.plotGrid.style.gridTemplateRows = rows;
    return true;
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
    const selectionPulse = this.currentSelectionPulse();
    if (panel === "main") {
      this.mainViewport = renderMainPlot({
        canvas: this.dom.mainCanvas,
        svg: this.dom.mainSvg,
        context,
        raster: this.raster,
        rasterResult: this.rasterResult,
        selectedCoord: this.store.state.selectionMode === "point" ? this.store.state.selectedCoord : null,
        selectedRegion:
          this.store.state.selectionMode === "region" ? this.store.state.selectedRegion : null,
        draftRegion: this.draftRegion,
        showCandidatePoints: true,
        showTrainPoints: false,
        selectionPulse,
      });
      this.maybeShowModelInteractionHint();
      return;
    }
    this.updateTrainControlVisibility();
    const matrix = this.selectedMatrix();
    if (!matrix) {
      this.trainViewport = null;
      this.setTrainSummary("", null);
      return;
    }
    const backgroundMode = this.store.state.backgroundMode;
    const backgroundLabel = BACKGROUND_MODE_LABELS[backgroundMode];
    this.dom.trainTitle.textContent = "Training";
    if (this.store.state.selectionMode === "region") {
      const selectedCount = this.store.state.selectedRegionCandidateIndices.length;
      const stats = renderRegionalInfluencePlot({
        canvas: this.dom.trainCanvas,
        svg: this.dom.trainSvg,
        context,
        raster: this.raster,
        rasterResult: this.rasterResult,
        aggregate: this.influenceAggregate,
        k: this.store.state.k,
        backgroundMode,
      });
      this.trainViewport = stats.viewport;
      const meanValue = this.influenceAggregate?.meanValue;
      this.setTrainSummary(this.store.state.selectedRegion
        ? `Local region · ${backgroundLabel} · average over ${selectedCount.toLocaleString()} candidates${Number.isFinite(meanValue) ? ` · mean I ${formatNumber(meanValue)}` : ""}`
        : "", stats.viewport, stats.colorbarPlacement);
      return;
    }
    const stats = renderLocalInfluencePlot({
      canvas: this.dom.trainCanvas,
      svg: this.dom.trainSvg,
      context,
      raster: this.raster,
      rasterResult: this.rasterResult,
      matrix,
      row: this.influenceRow,
      selectedCandidateIndex: this.store.state.selectedCandidateIndex,
      selectedTrainIndex: this.store.state.selectedTrainIndex,
      k: this.store.state.k,
      sign: this.store.state.sign,
      backgroundMode,
      selectionPulse,
    });
    this.trainViewport = stats.viewport;
    this.setTrainSummary(stats.maxAbs
      ? `Local · ${backgroundLabel} · max |I| ${formatNumber(stats.maxAbs)}`
      : `Local · ${backgroundLabel}`, stats.viewport, stats.colorbarPlacement);
  }

  private setTrainSummary(
    text: string,
    viewport: PlotViewport | null,
    colorbarPlacement: PlotColorbarPlacement = "right",
  ): void {
    const badge = this.dom.trainRange;
    badge.textContent = text;
    if (!text || !viewport) {
      badge.hidden = true;
      badge.style.removeProperty("left");
      badge.style.removeProperty("top");
      badge.style.removeProperty("max-width");
      return;
    }

    badge.hidden = false;
    const body = badge.closest<HTMLElement>(".plot-body");
    const bodyWidth = body?.getBoundingClientRect().width ?? viewport.right;
    const bodyHeight = body?.getBoundingClientRect().height ?? viewport.bottom + 36;
    const preferredLeft = Math.round(viewport.x);
    badge.style.left = `${preferredLeft}px`;
    badge.style.maxWidth = `${Math.round(Math.max(80, bodyWidth - preferredLeft - 6))}px`;
    if (badge.scrollWidth > badge.clientWidth + 1) {
      badge.style.left = "8px";
      badge.style.maxWidth = `${Math.round(Math.max(80, bodyWidth - 16))}px`;
    }
    const fitBadgeAtLeft = (left: number): number => {
      badge.style.left = `${Math.round(left)}px`;
      badge.style.maxWidth = `${Math.round(Math.max(80, bodyWidth - left - 6))}px`;
      if (badge.scrollWidth > badge.clientWidth + 1) {
        badge.style.left = "8px";
        badge.style.maxWidth = `${Math.round(Math.max(80, bodyWidth - 16))}px`;
        return 8;
      }
      return left;
    };

    const badgeHeight = badge.getBoundingClientRect().height || 16;
    const colorbar = body?.querySelector<SVGRectElement>(".colorbar-frame");
    const bodyRect = body?.getBoundingClientRect();
    const colorbarRect = colorbar?.getBoundingClientRect();
    const svgDecorationBounds = bodyRect
      ? Array.from(
          body?.querySelectorAll<SVGGraphicsElement>(
            "#trainSvg .axis text, #trainSvg .axis-label, #trainSvg .colorbar-ticks text",
          ) ?? [],
        ).map((element) => {
          const rect = element.getBoundingClientRect();
          return {
            left: rect.left - bodyRect.left,
            right: rect.right - bodyRect.left,
            top: rect.top - bodyRect.top,
            bottom: rect.bottom - bodyRect.top,
          };
        })
      : [];
    const colorbarBounds =
      bodyRect && colorbarRect
        ? {
            left: colorbarRect.left - bodyRect.left,
            right: colorbarRect.right - bodyRect.left,
            top: colorbarRect.top - bodyRect.top,
            bottom: colorbarRect.bottom - bodyRect.top,
          }
        : null;
    const badgeIntersectsBounds = (
      left: number,
      top: number,
      bounds: { left: number; right: number; top: number; bottom: number },
    ): boolean => {
      const badgeWidth = badge.getBoundingClientRect().width || badge.clientWidth || 0;
      return (
        left < bounds.right - 1 &&
        left + badgeWidth > bounds.left + 1 &&
        top < bounds.bottom - 1 &&
        top + badgeHeight > bounds.top + 1
      );
    };
    const badgeIntersectsFrame = (left: number, top: number): boolean =>
      badgeIntersectsBounds(left, top, {
        left: viewport.x,
        right: viewport.right,
        top: viewport.y,
        bottom: viewport.bottom,
      });
    const badgeIntersectsColorbar = (left: number, top: number): boolean =>
      colorbarBounds ? badgeIntersectsBounds(left, top, colorbarBounds) : false;
    const badgeIntersectsSvgDecorations = (left: number, top: number): boolean => {
      const badgeWidth = badge.getBoundingClientRect().width || badge.clientWidth || 0;
      const badgeRight = left + badgeWidth;
      return svgDecorationBounds.some(
        (decoration) =>
          left < decoration.right + 1 &&
          badgeRight > decoration.left - 1 &&
          top < decoration.bottom + 1 &&
          top + badgeHeight > decoration.top - 1,
      );
    };
    const inBody = (top: number): boolean => top >= 2 && top + badgeHeight <= bodyHeight - 2;
    const topCandidates =
      colorbarPlacement === "bottom"
        ? [
            viewport.y - badgeHeight - 4,
            viewport.bottom + 2,
            viewport.bottom + 6,
            colorbarBounds ? colorbarBounds.top - badgeHeight - 4 : viewport.bottom + 2,
            bodyHeight - badgeHeight - 2,
          ]
        : [
            viewport.y - badgeHeight - 4,
            viewport.bottom + 2,
            Math.min(
              viewport.bottom + 18,
              Math.max(viewport.bottom + 2, bodyHeight - badgeHeight - 2),
            ),
            bodyHeight - badgeHeight - 2,
          ];
    const leftCandidates = [preferredLeft, 8, Math.max(8, viewport.right - 180)];
    const findPlacement = (
      requireDecorationClearance: boolean,
    ): { left: number; top: number } | null => {
      for (const leftCandidate of leftCandidates) {
        const fittedLeft = fitBadgeAtLeft(leftCandidate);
        const topCandidate = topCandidates.find(
          (candidate) =>
            inBody(candidate) &&
            !badgeIntersectsFrame(fittedLeft, candidate) &&
            !badgeIntersectsColorbar(fittedLeft, candidate) &&
            (!requireDecorationClearance ||
              !badgeIntersectsSvgDecorations(fittedLeft, candidate)),
        );
        if (topCandidate !== undefined) return { left: fittedLeft, top: topCandidate };
      }
      return null;
    };

    const placement = findPlacement(true) ?? findPlacement(false);
    const left = placement?.left ?? fitBadgeAtLeft(preferredLeft);
    const top =
      placement?.top ??
      Math.max(
        2,
        Math.min(
          bodyHeight - badgeHeight - 2,
          viewport.y >= badgeHeight + 6 ? viewport.y - badgeHeight - 4 : viewport.bottom + 2,
        ),
      );
    badge.style.left = `${Math.round(left)}px`;
    badge.style.top = `${Math.round(top)}px`;
  }

  private handleMainPointerDown(event: PointerEvent): void {
    const context = this.context();
    if (!context || !this.mainViewport) return;
    const point = this.canvasPointer(event);
    if (!containsViewportPoint(point[0], point[1], this.mainViewport)) return;
    this.dismissModelInteractionHint();
    event.preventDefault();
    if (event.pointerType === "touch" && this.touchRegionArmed) {
      this.touchRegionArmed = false;
      this.lastTouchTap = null;
      this.startRegionGesture(event, point);
      return;
    }
    this.captureMainPointer(event.pointerId);
    this.modelGesture = {
      pointerId: event.pointerId,
      pointerType: event.pointerType,
      start: point,
      current: point,
      mode: "pending",
    };
  }

  private handleMainPointerMove(event: PointerEvent): void {
    if (!this.modelGesture || event.pointerId !== this.modelGesture.pointerId) return;
    const context = this.context();
    if (!context || !this.mainViewport) return;
    this.modelGesture.current = this.canvasPointer(event);
    const distance = this.gestureDistance(this.modelGesture, this.modelGesture.current);
    if (this.modelGesture.mode === "pending") {
      if (distance < DRAG_THRESHOLD_PX) return;
      this.touchRegionArmed = false;
      this.lastTouchTap = null;
      this.modelGesture.mode = "region";
    }
    event.preventDefault();
    this.updateDraftRegion(context);
  }

  private handleMainPointerUp(event: PointerEvent): void {
    if (!this.modelGesture || event.pointerId !== this.modelGesture.pointerId) return;
    const context = this.context();
    const gesture = this.modelGesture;
    const end = this.canvasPointer(event);
    gesture.current = end;
    this.modelGesture = null;
    this.releaseMainPointer(event.pointerId);
    const distance = this.gestureDistance(gesture, end);
    if (gesture.mode === "region") {
      const region = this.draftRegion;
      this.draftRegion = null;
      if (distance < DRAG_THRESHOLD_PX || !region) {
        this.schedule("main");
        return;
      }
      this.finalizeRegionSelection(region);
      return;
    }
    this.draftRegion = null;
    if (gesture.pointerType === "touch") {
      if (distance >= DRAG_THRESHOLD_PX) {
        if (context && this.mainViewport) {
          const region = regionBoundsFromProjectedViewportDrag(
            gesture.start,
            end,
            this.mainProjection(context),
            this.mainViewport,
          );
          this.finalizeRegionSelection(region);
          return;
        }
        this.schedule("main");
        return;
      }
      if (this.isDoubleTap(end)) {
        this.touchRegionArmed = true;
        this.lastTouchTap = null;
        this.schedule("main");
        return;
      }
      this.lastTouchTap = { time: performance.now(), point: end };
      this.selectPointFromPointer(event);
      return;
    }
    if (distance >= DRAG_THRESHOLD_PX && context && this.mainViewport) {
      const region = regionBoundsFromProjectedViewportDrag(
        gesture.start,
        end,
        this.mainProjection(context),
        this.mainViewport,
      );
      this.finalizeRegionSelection(region);
      return;
    }
    this.selectPointFromPointer(event);
  }

  private handleMainPointerCancel(event: PointerEvent): void {
    if (!this.modelGesture || event.pointerId !== this.modelGesture.pointerId) return;
    this.modelGesture = null;
    this.draftRegion = null;
    this.releaseMainPointer(event.pointerId);
    this.schedule("main");
  }

  private handleTrainPointerDown(event: PointerEvent): void {
    if (!this.context() || !this.trainViewport) return;
    const point = this.canvasPointer(event, this.dom.trainCanvas);
    if (!containsViewportPoint(point[0], point[1], this.trainViewport)) return;
    event.preventDefault();
    this.capturePointer(this.dom.trainCanvas, event.pointerId);
    this.trainGesture = {
      pointerId: event.pointerId,
      start: point,
    };
  }

  private handleTrainPointerUp(event: PointerEvent): void {
    if (!this.trainGesture || event.pointerId !== this.trainGesture.pointerId) return;
    const gesture = this.trainGesture;
    const end = this.canvasPointer(event, this.dom.trainCanvas);
    this.trainGesture = null;
    this.releasePointer(this.dom.trainCanvas, event.pointerId);
    if (Math.hypot(end[0] - gesture.start[0], end[1] - gesture.start[1]) >= DRAG_THRESHOLD_PX) {
      return;
    }
    this.selectPointFromPointer(event, "train");
  }

  private handleTrainPointerCancel(event: PointerEvent): void {
    if (!this.trainGesture || event.pointerId !== this.trainGesture.pointerId) return;
    this.trainGesture = null;
    this.releasePointer(this.dom.trainCanvas, event.pointerId);
  }

  private startRegionGesture(event: PointerEvent, point: [number, number]): void {
    const context = this.context();
    if (!context || !this.mainViewport) return;
    this.captureMainPointer(event.pointerId);
    this.modelGesture = {
      pointerId: event.pointerId,
      pointerType: event.pointerType,
      start: point,
      current: point,
      mode: "region",
    };
    this.updateDraftRegion(context);
  }

  private updateDraftRegion(context: PlotContext): void {
    if (!this.modelGesture || !this.mainViewport) return;
    this.draftRegion = regionBoundsFromProjectedViewportDrag(
      this.modelGesture.start,
      this.modelGesture.current,
      this.mainProjection(context),
      this.mainViewport,
    );
    this.schedule("main");
  }

  private gestureDistance(
    gesture: Pick<ModelGesture, "start">,
    point: [number, number],
  ): number {
    return Math.hypot(point[0] - gesture.start[0], point[1] - gesture.start[1]);
  }

  private isDoubleTap(point: [number, number]): boolean {
    if (!this.lastTouchTap) return false;
    const elapsed = performance.now() - this.lastTouchTap.time;
    const distance = Math.hypot(
      point[0] - this.lastTouchTap.point[0],
      point[1] - this.lastTouchTap.point[1],
    );
    return elapsed <= DOUBLE_TAP_MS && distance <= DOUBLE_TAP_DISTANCE_PX;
  }

  private captureMainPointer(pointerId: number): void {
    this.capturePointer(this.dom.mainCanvas, pointerId);
  }

  private releaseMainPointer(pointerId: number): void {
    this.releasePointer(this.dom.mainCanvas, pointerId);
  }

  private capturePointer(canvas: HTMLCanvasElement, pointerId: number): void {
    try {
      canvas.setPointerCapture(pointerId);
    } catch {
      // Synthetic pointer events in tests do not always create an active pointer capture target.
    }
  }

  private releasePointer(canvas: HTMLCanvasElement, pointerId: number): void {
    try {
      if (canvas.hasPointerCapture(pointerId)) {
        canvas.releasePointerCapture(pointerId);
      }
    } catch {
      // Ignore capture state mismatches from synthetic events.
    }
  }

  private selectPointFromPointer(event: PointerEvent, target: PlotTarget = "main"): void {
    const context = this.context();
    const viewport = target === "train" ? this.trainViewport : this.mainViewport;
    if (!context || !viewport) return;
    const canvas = target === "train" ? this.dom.trainCanvas : this.dom.mainCanvas;
    const projection = target === "train" ? this.trainProjection(context) : this.mainProjection(context);
    const domain = pointerInDomain(event, canvas, projection, viewport);
    if (!domain) return;
    const displayDomain = projectPointToDisplay(domain, projection);
    const matrix = this.selectedMatrix();
    const candidateIndex =
      matrix?.row_source === "train_points"
        ? this.store.state.selectedCandidateIndex
        : (this.candidateDelaunay?.find(displayDomain[0], displayDomain[1]) ?? 0);
    const trainIndex =
      matrix?.row_source === "train_points"
        ? (this.trainDelaunay?.find(displayDomain[0], displayDomain[1]) ?? 0)
        : (this.trainDelaunay?.find(displayDomain[0], displayDomain[1]) ??
          this.store.state.selectedTrainIndex);
    const coord =
      matrix?.row_source === "train_points"
        ? pointAt(context.points.train_points, trainIndex, context.trainDim)
        : pointAt(context.points.candidate_points, candidateIndex, context.candidateDim);
    this.touchRegionArmed = false;
    this.draftRegion = null;
    this.influenceAggregate = null;
    this.store.dispatch({ type: "selection", candidateIndex, trainIndex, coord });
    this.triggerSelectionPulse();
    void this.loadInfluenceForSelection().then(() => {
      this.schedule("main");
      this.schedule("train");
      this.updateStats();
    });
  }

  private finalizeRegionSelection(region: Bounds): void {
    this.store.dispatch({ type: "regionSelection", region, candidateIndices: [] });
    const matrix = this.selectedMatrix();
    const rowIndices = matrix ? this.refreshRegionRowSelection(matrix) : [];
    this.influenceAggregate = null;
    this.schedule("main");
    this.schedule("train");
    this.updateStats();
    if (!rowIndices.length) {
      this.updatePrefetchPlan();
    }
    void this.loadInfluenceForSelection().then(() => {
      this.schedule("train");
      this.updateStats();
    });
  }

  private clearRegionSelection(): void {
    if (!this.store.state.selectedRegion && !this.draftRegion) return;
    this.draftRegion = null;
    this.modelGesture = null;
    this.touchRegionArmed = false;
    this.influenceAggregate = null;
    this.store.dispatch({ type: "regionSelection", region: null, candidateIndices: [] });
    this.latestAggregateRequest += 1;
    this.schedule("main");
    this.schedule("train");
    this.updateStats();
    this.updatePrefetchPlan();
  }

  private refreshRegionRowSelection(matrix: InfluenceMatrixManifest): number[] {
    const context = this.context();
    const region = this.store.state.selectedRegion;
    if (!context || !region) {
      this.store.dispatch({ type: "regionSelection", region, candidateIndices: [] });
      return [];
    }
    const rowSourcePoints =
      matrix.row_source === "train_points"
        ? context.points.train_points
        : context.points.candidate_points;
    const rowDim = matrix.row_source === "train_points" ? context.trainDim : context.candidateDim;
    const rowIndices = selectPointIndicesInBounds(
      rowSourcePoints,
      rowDim,
      region,
      matrix.row_count,
    );
    this.store.dispatch({ type: "regionSelection", region, candidateIndices: rowIndices });
    return rowIndices;
  }

  private mainPlotBounds(context: PlotContext): Bounds {
    return this.manifest?.field_raster
      ? boundsFromAxisMap(this.manifest.field_raster.bounds, this.manifest.field_raster.axes)
      : context.bounds;
  }

  private canvasPointer(event: PointerEvent, canvas: HTMLCanvasElement = this.dom.mainCanvas): [number, number] {
    const rect = canvas.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }

  private updateStats(): void {
    if (!this.manifest) return;
    if (this.store.state.selectionMode === "region") {
      const region = this.store.state.selectedRegion;
      this.dom.selectedPointLabel.textContent = "Region";
      this.dom.selectedValueLabel.textContent = "Value";
      this.dom.selectedPoint.textContent = region
        ? `x[${formatRegionReadoutNumber(region.minX)},${formatRegionReadoutNumber(region.maxX)}] y[${formatRegionReadoutNumber(region.minY)},${formatRegionReadoutNumber(region.maxY)}]`
        : "-";
      this.dom.selectedValue.textContent = "-";
      this.refreshResponsiveLayout();
      return;
    }
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
    this.dom.selectedPointLabel.textContent = "Point";
    this.dom.selectedValueLabel.textContent = "Value";
    this.dom.selectedPoint.textContent = `(${formatReadoutNumber(x)},${formatReadoutNumber(y)})`;
    this.dom.selectedValue.textContent = formatReadoutNumber(sample?.value);
    this.refreshResponsiveLayout();
  }
}
