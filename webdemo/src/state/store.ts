import type { Bounds, InfluenceSign, SelectionMode, SummaryName } from "../types";

export interface AppState {
  runId: string | null;
  fieldId: string | null;
  fieldKind: "prediction" | "loss";
  matrixId: string | null;
  sign: InfluenceSign;
  summary: SummaryName;
  k: number;
  selectionMode: SelectionMode;
  selectedCandidateIndex: number;
  selectedTrainIndex: number;
  selectedCoord: [number, number] | null;
  selectedRegion: Bounds | null;
  selectedRegionCandidateIndices: number[];
  mobileTab: "local" | "global";
}

export type AppAction =
  | { type: "run"; runId: string }
  | { type: "field"; fieldId: string }
  | { type: "fieldKind"; fieldKind: AppState["fieldKind"] }
  | { type: "matrix"; matrixId: string }
  | { type: "sign"; sign: InfluenceSign }
  | { type: "summary"; summary: SummaryName }
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
  | { type: "mobileTab"; mobileTab: AppState["mobileTab"] }
  | { type: "resetSelection" };

export const initialState: AppState = {
  runId: null,
  fieldId: null,
  fieldKind: "prediction",
  matrixId: null,
  sign: "abs",
  summary: "mean_abs",
  k: 25,
  selectionMode: "point",
  selectedCandidateIndex: 0,
  selectedTrainIndex: 0,
  selectedCoord: null,
  selectedRegion: null,
  selectedRegionCandidateIndices: [],
  mobileTab: "local",
};

export function reduceState(state: AppState, action: AppAction): AppState {
  switch (action.type) {
    case "run":
      return {
        ...state,
        runId: action.runId,
        selectionMode: "point",
        selectedCandidateIndex: 0,
        selectedTrainIndex: 0,
        selectedRegion: null,
        selectedRegionCandidateIndices: [],
      };
    case "field":
      return { ...state, fieldId: action.fieldId };
    case "fieldKind":
      return { ...state, fieldKind: action.fieldKind };
    case "matrix":
      return { ...state, matrixId: action.matrixId, selectedTrainIndex: 0 };
    case "sign":
      return { ...state, sign: action.sign };
    case "summary":
      return { ...state, summary: action.summary };
    case "k":
      return { ...state, k: Math.max(1, Math.trunc(action.k)) };
    case "selectionMode":
      return { ...state, selectionMode: action.selectionMode };
    case "selection":
      return {
        ...state,
        selectedCandidateIndex: Math.max(0, Math.trunc(action.candidateIndex)),
        selectedTrainIndex: Math.max(0, Math.trunc(action.trainIndex ?? state.selectedTrainIndex)),
        selectedCoord: action.coord,
      };
    case "regionSelection":
      return {
        ...state,
        selectedRegion: action.region,
        selectedRegionCandidateIndices: action.candidateIndices.map((index) =>
          Math.max(0, Math.trunc(index)),
        ),
      };
    case "mobileTab":
      return { ...state, mobileTab: action.mobileTab };
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
