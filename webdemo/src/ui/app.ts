import type { Delaunay } from "d3";
import type {
  BackgroundMode,
  Bounds,
  DataIndex,
  IndexVariantEntry,
  InfluenceAggregate,
  InfluenceMatrixManifest,
  InfluenceRow,
  ModelQuality,
  PointArrays,
  RasterData,
  RunManifest,
  TypedArray,
} from "../types";
import { DataRepository } from "../data/arrays";
import { LruCache } from "../data/cache";
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
  PLOT_DECORATION_INSETS,
  pointerInDomain,
  rasterSampleAtCoord,
  renderLocalInfluencePlot,
  renderMainPlot,
  renderRegionalInfluencePlot,
  type PlotContext,
  type RasterRenderResult,
} from "../viz/plots";
import {
  plotProjectionForManifest,
  projectPointToDisplay,
  regionBoundsFromProjectedViewportDrag,
} from "../viz/projection";
import {
  formatDisplayLabel,
  formatInfluenceMatrixLabel,
  formatNumber,
  getDomRefs,
  showMessage,
  type DomRefs,
} from "./dom";

const DEFAULT_MATRIX_ID = "influences_total_loss_total_loss";
const BACKGROUND_MODE_LABELS: Record<BackgroundMode, string> = {
  points: "Points",
  smooth: "Smooth",
  cell: "Cells",
};
const DRAG_THRESHOLD_PX = 8;
const DOUBLE_TAP_MS = 350;
const DOUBLE_TAP_DISTANCE_PX = 36;

type PanelName = "main" | "train";
type ControlLayout = "inline" | "bar" | "menu";
type ModelGesture = {
  pointerId: number;
  pointerType: string;
  start: [number, number];
  current: [number, number];
  mode: "pending" | "region";
};

export class AppController {
  private readonly dom: DomRefs = getDomRefs();
  private readonly store = new Store();
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
  private mainViewport = null as ReturnType<typeof renderMainPlot> | null;
  private draftRegion: Bounds | null = null;
  private modelGesture: ModelGesture | null = null;
  private touchRegionArmed = false;
  private lastTouchTap: { time: number; point: [number, number] } | null = null;
  private latestRasterRequest = 0;
  private latestAggregateRequest = 0;
  private scheduled = new Set<PanelName>();
  private lastLayoutSignature = "";
  private lastControlLayoutSignature = "";

  async start(): Promise<void> {
    this.bindEvents();
    this.observeLayout();
    showMessage(this.dom.message, null);
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
    } catch (error) {
      showMessage(this.dom.message, error instanceof Error ? error.message : String(error));
      this.dom.runMeta.textContent = "Data unavailable";
    }
  }

  private bindEvents(): void {
    this.dom.problemSelect.addEventListener("change", () => {
      this.store.dispatch({ type: "problem", problem: this.dom.problemSelect.value });
      void this.loadActiveVariant();
    });
    this.dom.qualityButtons.addEventListener("click", (event) => {
      const button = (event.target as Element).closest<HTMLButtonElement>("button[data-model-quality]");
      if (!button) return;
      const quality: ModelQuality = button.dataset.modelQuality === "bad" ? "bad" : "good";
      this.store.dispatch({ type: "modelQuality", modelQuality: quality });
      this.setActiveButtons(this.dom.qualityButtons, quality, "modelQuality");
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
    this.dom.modelMenuButton.addEventListener("click", (event) => {
      event.stopPropagation();
      this.toggleMenu("model");
    });
    this.dom.trainMenuButton.addEventListener("click", (event) => {
      event.stopPropagation();
      this.toggleMenu("train");
    });
    this.dom.modelMenu.addEventListener("click", (event) => event.stopPropagation());
    this.dom.trainMenu.addEventListener("click", (event) => event.stopPropagation());
    document.addEventListener("click", () => this.closeMenus());
    this.dom.resetButton.addEventListener("click", () => {
      this.store.dispatch({ type: "resetSelection" });
      this.draftRegion = null;
      this.modelGesture = null;
      this.touchRegionArmed = false;
      this.lastTouchTap = null;
      this.influenceAggregate = null;
      this.pickDefaultSelection();
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
    window.addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      this.closeMenus();
      this.clearRegionSelection();
    });
    window.addEventListener("resize", () => this.handleViewportScaleChange());
    window.visualViewport?.addEventListener("resize", () => this.handleViewportScaleChange());
    this.observeDevicePixelRatio();
  }

  private toggleMenu(menu: "model" | "train"): void {
    const actions = menu === "model" ? this.dom.modelActions : this.dom.trainActions;
    const isOpen = actions?.dataset.open === "true";
    this.closeMenus();
    this.setMenuOpen(menu, !isOpen);
  }

  private setMenuOpen(menu: "model" | "train", open: boolean): void {
    const button = menu === "model" ? this.dom.modelMenuButton : this.dom.trainMenuButton;
    const actions = menu === "model" ? this.dom.modelActions : this.dom.trainActions;
    actions.dataset.open = open ? "true" : "false";
    button.setAttribute("aria-expanded", String(open));
  }

  private closeMenus(): void {
    this.setMenuOpen("model", false);
    this.setMenuOpen("train", false);
  }

  private updateTrainControlVisibility(): void {
    this.dom.kControl.hidden = false;
    this.setActiveButtons(this.dom.backgroundButtons, this.store.state.backgroundMode, "backgroundMode");
  }

  private refreshResponsiveLayout(): boolean {
    const controlsChanged = this.applyAdaptiveControlLayouts();
    const layoutChanged = this.applyAdaptiveLayout();
    const controlsChangedAfterLayout = layoutChanged ? this.applyAdaptiveControlLayouts() : false;
    return controlsChanged || layoutChanged || controlsChangedAfterLayout;
  }

  private applyAdaptiveControlLayouts(): boolean {
    this.updatePanelWidthVar(this.dom.modelPanel);
    this.updatePanelWidthVar(this.dom.trainPanel);

    const modelLayout = this.chooseControlLayout({
      panel: this.dom.modelPanel,
      actions: this.dom.modelActions,
      menu: this.dom.modelMenu,
      button: this.dom.modelMenuButton,
    });
    const trainLayout = this.chooseControlLayout({
      panel: this.dom.trainPanel,
      actions: this.dom.trainActions,
      menu: this.dom.trainMenu,
      button: this.dom.trainMenuButton,
    });
    const signature = `${modelLayout}|${trainLayout}`;
    const changed = signature !== this.lastControlLayoutSignature;
    this.lastControlLayoutSignature = signature;
    this.setControlLayout(this.dom.modelActions, this.dom.modelMenuButton, modelLayout);
    this.setControlLayout(this.dom.trainActions, this.dom.trainMenuButton, trainLayout);
    return changed;
  }

  private chooseControlLayout({
    panel,
    actions,
    menu,
    button,
  }: {
    panel: HTMLElement;
    actions: HTMLElement;
    menu: HTMLElement;
    button: HTMLButtonElement;
  }): ControlLayout {
    const previousLayout = this.controlLayout(actions);
    const wasOpen = actions.dataset.open === "true";
    const candidates: ControlLayout[] = ["inline", "bar", "menu"];
    for (const layout of candidates) {
      this.setControlLayout(actions, button, layout, false);
      if (this.controlLayoutFits(panel, actions, menu, layout)) {
        this.setControlLayout(actions, button, previousLayout, wasOpen && previousLayout === "menu");
        return layout;
      }
    }
    this.setControlLayout(actions, button, previousLayout, wasOpen && previousLayout === "menu");
    return "menu";
  }

  private controlLayoutFits(
    panel: HTMLElement,
    actions: HTMLElement,
    menu: HTMLElement,
    layout: ControlLayout,
  ): boolean {
    if (layout === "menu") return true;
    const header = actions.closest<HTMLElement>(".plot-header");
    if (!header) return false;
    const panelRect = panel.getBoundingClientRect();
    const headerRect = header.getBoundingClientRect();
    if (panelRect.width <= 1 || panelRect.height <= 1) return false;

    const bodyHeight = panelRect.height - headerRect.height;
    const minBodyHeight = panelRect.height < 360 ? 110 : 180;
    if (bodyHeight < minBodyHeight) return false;
    if (layout === "inline" && headerRect.height > 84) return false;
    if (layout === "bar" && headerRect.height > Math.min(150, panelRect.height * 0.45)) return false;
    return (
      !this.hasHorizontalOverflow(header) &&
      !this.hasHorizontalOverflow(actions) &&
      !this.hasHorizontalOverflow(menu)
    );
  }

  private hasHorizontalOverflow(element: HTMLElement): boolean {
    return element.scrollWidth > element.clientWidth + 3;
  }

  private controlLayout(actions: HTMLElement): ControlLayout {
    const layout = actions.dataset.controlLayout;
    return layout === "inline" || layout === "bar" ? layout : "menu";
  }

  private setControlLayout(
    actions: HTMLElement,
    button: HTMLButtonElement,
    layout: ControlLayout,
    keepOpen = actions.dataset.open === "true",
  ): void {
    actions.dataset.controlLayout = layout;
    if (layout !== "menu") {
      actions.dataset.open = "false";
      button.setAttribute("aria-expanded", "false");
      return;
    }
    actions.dataset.open = keepOpen ? "true" : "false";
    button.setAttribute("aria-expanded", String(keepOpen));
  }

  private updatePanelWidthVar(panel: HTMLElement): void {
    const width = Math.max(120, panel.getBoundingClientRect().width);
    panel.style.setProperty("--panel-width", `${Math.round(width)}px`);
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
    const snapshot = this.captureVariantStateSnapshot();
    this.prefetcher?.stop();
    this.prefetcher = null;
    this.repo?.abortBackground();
    this.raster = null;
    this.rasterResult = null;
    this.influenceRow = null;
    this.influenceAggregate = null;
    this.draftRegion = null;
    this.modelGesture = null;
    this.touchRegionArmed = false;
    this.lastTouchTap = null;
    this.latestAggregateRequest += 1;
    showMessage(this.dom.message, null);
    this.dom.runMeta.textContent = `Loading ${variant.display_name} · ${qualityLabel(variant.model_quality)}`;
    this.dom.problemSelect.value = variant.problem;
    this.setActiveButtons(this.dom.qualityButtons, variant.model_quality, "modelQuality");
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
    this.dom.runMeta.textContent = `${formatProblemLabel(this.manifest.display_name)} · ${qualityLabel(this.manifest.model_quality)} · ${this.manifest.n_candidate.toLocaleString()} candidate · ${this.manifest.n_train.toLocaleString()} train`;
    this.refreshResponsiveLayout();
    this.schedule("main");
    this.schedule("train");
    this.updateStats();
    this.prefetcher = new RunPrefetcher(this.repo, this.manifest);
    this.startBackgroundPrefetch();
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
        (matrix) => new Option(formatInfluenceMatrixLabel(matrix), matrix.id),
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
    this.setActiveButtons(this.dom.signButtons, this.store.state.sign, "sign");
    this.setActiveButtons(this.dom.backgroundButtons, this.store.state.backgroundMode, "backgroundMode");
    this.updateTrainControlVisibility();
  }

  private populateFieldSelect(selectedFieldId: string | null): string | null {
    if (!this.manifest) return null;
    const orderedEntries = orderedFieldEntries(this.manifest);
    const options = orderedEntries.map(([id, field]) => new Option(formatDisplayLabel(field.label), id));
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
    this.raster = await this.repo.loadRaster(fieldId, "foreground");
    await this.renderRasterWithWorker();
    this.dom.mainTitle.textContent = "Model";
    this.dom.mainRange.textContent = "";
    if (this.refreshResponsiveLayout()) this.schedule("train");
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
    this.schedule("train");
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
    const requestId = ++this.latestAggregateRequest;
    const rowIndices = this.refreshRegionRowSelection(matrix);
    if (!this.store.state.selectedRegion) {
      this.influenceAggregate = null;
      this.updatePrefetchPlan();
      return;
    }
    this.influenceAggregate = null;
    this.schedule("train");
    const aggregate = await this.repo.loadInfluenceAggregate(
      matrix,
      this.store.state.sign,
      rowIndices,
      "foreground",
    );
    if (requestId !== this.latestAggregateRequest) return;
    this.influenceAggregate = aggregate;
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
      padding: PLOT_DECORATION_INSETS,
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
        showTrainPoints: true,
      });
      return;
    }
    this.updateTrainControlVisibility();
    const matrix = this.selectedMatrix();
    if (!matrix) return;
    const backgroundMode = this.store.state.backgroundMode;
    const backgroundLabel = BACKGROUND_MODE_LABELS[backgroundMode];
    this.dom.trainTitle.textContent = "Train";
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
      this.dom.trainRange.textContent = this.store.state.selectedRegion
        ? `Local region · ${backgroundLabel} · sum over ${selectedCount.toLocaleString()} candidates${stats.maxAbs ? ` · max |sum I| ${formatNumber(stats.maxAbs)}` : ""}`
        : "";
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
    });
    this.dom.trainRange.textContent = stats.maxAbs
      ? `Local · ${backgroundLabel} · max |I| ${formatNumber(stats.maxAbs)}`
      : `Local · ${backgroundLabel}`;
  }

  private handleMainPointerDown(event: PointerEvent): void {
    const context = this.context();
    if (!context || !this.mainViewport) return;
    const point = this.canvasPointer(event);
    if (!containsViewportPoint(point[0], point[1], this.mainViewport)) return;
    this.closeMenus();
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
      if (this.modelGesture.pointerType === "touch" || distance < DRAG_THRESHOLD_PX) return;
      this.modelGesture.mode = "region";
    }
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
    try {
      this.dom.mainCanvas.setPointerCapture(pointerId);
    } catch {
      // Synthetic pointer events in tests do not always create an active pointer capture target.
    }
  }

  private releaseMainPointer(pointerId: number): void {
    try {
      if (this.dom.mainCanvas.hasPointerCapture(pointerId)) {
        this.dom.mainCanvas.releasePointerCapture(pointerId);
      }
    } catch {
      // Ignore capture state mismatches from synthetic events.
    }
  }

  private selectPointFromPointer(event: PointerEvent): void {
    const context = this.context();
    if (!context || !this.mainViewport) return;
    const projection = this.mainProjection(context);
    const domain = pointerInDomain(event, this.dom.mainCanvas, projection, this.mainViewport);
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

  private canvasPointer(event: PointerEvent): [number, number] {
    const rect = this.dom.mainCanvas.getBoundingClientRect();
    return [event.clientX - rect.left, event.clientY - rect.top];
  }

  private updateStats(): void {
    if (!this.manifest) return;
    if (this.store.state.selectionMode === "region") {
      const region = this.store.state.selectedRegion;
      this.dom.selectedPointLabel.textContent = "Region";
      this.dom.selectedValueLabel.textContent = "Value";
      this.dom.selectedPoint.textContent = region
        ? `x ${formatNumber(region.minX)} … ${formatNumber(region.maxX)}, y ${formatNumber(region.minY)} … ${formatNumber(region.maxY)}`
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
    this.dom.selectedPoint.textContent = `(${formatNumber(x)}, ${formatNumber(y)})`;
    this.dom.selectedValue.textContent = formatNumber(sample?.value);
    this.refreshResponsiveLayout();
  }
}
