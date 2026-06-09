import { DEFAULT_PLOT_PADDING, plotViewport } from "./geometry";

export type PlotLayoutOrientation = "row" | "column";

export interface AdaptivePlotLayoutInput {
  width: number;
  height: number;
  gap: number;
  headerHeight: number;
  modelAspect: number;
  trainAspect: number;
  padding?: number;
}

export interface AdaptivePlotLayoutCandidate {
  orientation: PlotLayoutOrientation;
  modelTrackPx: number;
  trainTrackPx: number;
  modelViewportArea: number;
  trainViewportArea: number;
  score: number;
}

const EPS = 1e-6;

function positiveNumber(value: number, fallback: number): number {
  return Number.isFinite(value) && value > 0 ? value : fallback;
}

function fittedViewportArea(aspect: number, width: number, height: number, padding: number): number {
  if (width <= 0 || height <= 0) return 0;
  const viewport = plotViewport(
    { minX: 0, maxX: aspect, minY: 0, maxY: 1 },
    width,
    height,
    padding,
  );
  return viewport.width * viewport.height;
}

export function adaptivePlotLayoutCandidates(
  input: AdaptivePlotLayoutInput,
): [AdaptivePlotLayoutCandidate, AdaptivePlotLayoutCandidate] {
  const width = positiveNumber(input.width, 1);
  const height = positiveNumber(input.height, 1);
  const gap = Math.max(0, Number.isFinite(input.gap) ? input.gap : 0);
  const headerHeight = Math.max(0, Number.isFinite(input.headerHeight) ? input.headerHeight : 0);
  const modelAspect = positiveNumber(input.modelAspect, 1);
  const trainAspect = positiveNumber(input.trainAspect, 1);
  const padding = Math.max(
    0,
    Number.isFinite(input.padding ?? DEFAULT_PLOT_PADDING)
      ? (input.padding ?? DEFAULT_PLOT_PADDING)
      : DEFAULT_PLOT_PADDING,
  );

  const rowTrackTotal = Math.max(0, width - gap);
  const rowWeightTotal = modelAspect + trainAspect;
  const rowModelTrack = rowTrackTotal * (modelAspect / rowWeightTotal);
  const rowTrainTrack = rowTrackTotal - rowModelTrack;
  const rowBodyHeight = Math.max(0, height - headerHeight);
  const rowModelArea = fittedViewportArea(modelAspect, rowModelTrack, rowBodyHeight, padding);
  const rowTrainArea = fittedViewportArea(trainAspect, rowTrainTrack, rowBodyHeight, padding);

  const columnBodyTotal = Math.max(0, height - gap - headerHeight * 2);
  const modelInverseAspect = 1 / modelAspect;
  const trainInverseAspect = 1 / trainAspect;
  const columnWeightTotal = modelInverseAspect + trainInverseAspect;
  const columnModelBody = columnBodyTotal * (modelInverseAspect / columnWeightTotal);
  const columnTrainBody = columnBodyTotal - columnModelBody;
  const columnModelTrack = headerHeight + columnModelBody;
  const columnTrainTrack = headerHeight + columnTrainBody;
  const columnModelArea = fittedViewportArea(modelAspect, width, columnModelBody, padding);
  const columnTrainArea = fittedViewportArea(trainAspect, width, columnTrainBody, padding);

  return [
    {
      orientation: "row",
      modelTrackPx: rowModelTrack,
      trainTrackPx: rowTrainTrack,
      modelViewportArea: rowModelArea,
      trainViewportArea: rowTrainArea,
      score: rowModelArea + rowTrainArea,
    },
    {
      orientation: "column",
      modelTrackPx: columnModelTrack,
      trainTrackPx: columnTrainTrack,
      modelViewportArea: columnModelArea,
      trainViewportArea: columnTrainArea,
      score: columnModelArea + columnTrainArea,
    },
  ];
}

export function chooseAdaptivePlotLayout(input: AdaptivePlotLayoutInput): AdaptivePlotLayoutCandidate {
  const [row, column] = adaptivePlotLayoutCandidates(input);
  if (row.score > column.score + EPS) return row;
  if (column.score > row.score + EPS) return column;
  return input.width >= input.height ? row : column;
}
