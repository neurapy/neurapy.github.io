import type { AppView, BackgroundMode, Bounds, InfluenceSign, ModelQuality, SelectionMode } from "../types";

export const MAX_TOP_K = 256;

export interface AppState {
  appView: AppView;
  problem: string | null;
  modelQuality: ModelQuality;
  fieldId: string | null;
  matrixId: string | null;
  sign: InfluenceSign;
  k: number;
  selectionMode: SelectionMode;
  selectedCandidateIndex: number;
  selectedTrainIndex: number;
  selectedCoord: [number, number] | null;
  selectedRegion: Bounds | null;
  selectedRegionCandidateIndices: number[];
  backgroundMode: BackgroundMode;
}

export type AppAction =
  | { type: "view"; appView: AppView }
  | { type: "problem"; problem: string }
  | { type: "modelQuality"; modelQuality: ModelQuality }
  | { type: "field"; fieldId: string | null }
  | { type: "matrix"; matrixId: string | null }
  | { type: "sign"; sign: InfluenceSign }
  | { type: "k"; k: number }
  | { type: "selectionMode"; selectionMode: SelectionMode }
  | {
      type: "selection";
      candidateIndex: number;
      trainIndex?: number;
      coord: [number, number];
    }
  | {
      type: "regionSelection";
      region: Bounds | null;
      candidateIndices: number[];
    }
  | { type: "backgroundMode"; backgroundMode: BackgroundMode }
  | { type: "resetSelection" };

export const initialState: AppState = {
  appView: "playground",
  problem: null,
  modelQuality: "good",
  fieldId: null,
  matrixId: null,
  sign: "abs",
  k: 25,
  selectionMode: "point",
  selectedCandidateIndex: 0,
  selectedTrainIndex: 0,
  selectedCoord: null,
  selectedRegion: null,
  selectedRegionCandidateIndices: [],
  backgroundMode: "points",
};

export function reduceState(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "view":
      return { ...state, appView: action.appView };
    case "problem":
      return { ...state, problem: action.problem };
    case "modelQuality":
      return { ...state, modelQuality: action.modelQuality };
    case "field":
      return { ...state, fieldId: action.fieldId };
    case "matrix":
      return { ...state, matrixId: action.matrixId, selectedTrainIndex: 0 };
    case "sign":
      return { ...state, sign: action.sign };
    case "k": {
      const k = Math.trunc(action.k);
      return { ...state, k: Number.isFinite(k) ? Math.max(0, Math.min(MAX_TOP_K, k)) : state.k };
    }
    case "selectionMode":
      return { ...state, selectionMode: action.selectionMode };
    case "selection":
      return {
        ...state,
        selectionMode: "point",
        selectedCandidateIndex: Math.max(0, Math.trunc(action.candidateIndex)),
        selectedTrainIndex: Math.max(0, Math.trunc(action.trainIndex ?? state.selectedTrainIndex)),
        selectedCoord: action.coord,
        selectedRegion: null,
        selectedRegionCandidateIndices: [],
      };
    case "regionSelection":
      return {
        ...state,
        selectionMode: action.region ? "region" : "point",
        selectedRegion: action.region,
        selectedRegionCandidateIndices: action.candidateIndices.map((index) =>
          Math.max(0, Math.trunc(index)),
        ),
      };
    case "backgroundMode":
      return { ...state, backgroundMode: action.backgroundMode };
    case "resetSelection":
      return {
        ...state,
        selectionMode: "point",
        selectedCandidateIndex: 0,
        selectedTrainIndex: 0,
        selectedCoord: null,
        selectedRegion: null,
        selectedRegionCandidateIndices: [],
      };
    default:
      return state;
  }
}

export class Store {
  private stateValue: AppState;
  private readonly listeners = new Set<(state: AppState, previous: AppState) => void>();

  constructor(initial: AppState = initialState) {
    this.stateValue = initial;
  }

  get state(): AppState {
    return this.stateValue;
  }

  dispatch(action: AppAction): void {
    const previous = this.stateValue;
    const next = reduceState(previous, action);
    this.stateValue = next;
    for (const listener of this.listeners) listener(next, previous);
  }

  subscribe(listener: (state: AppState, previous: AppState) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }
}
