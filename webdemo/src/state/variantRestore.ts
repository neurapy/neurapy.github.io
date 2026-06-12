import type {
  Bounds,
  FieldKind,
  InfluenceMatrixManifest,
  RasterFieldManifest,
  RunManifest,
} from "../types";
import { normalizeRegionBounds, normalizedBounds } from "../viz/geometry";
import type { AppState } from "./store";

export interface MatrixRestoreSignature {
  method: string;
  left_term: string;
  right_term: string;
  row_source: InfluenceMatrixManifest["row_source"];
}

export type SelectionSnapshot =
  | { mode: "point"; coord: [number, number] }
  | { mode: "region"; region: Bounds };

export interface VariantStateSnapshot {
  fieldId: string | null;
  fieldKind: FieldKind | null;
  matrixId: string | null;
  matrixSignature: MatrixRestoreSignature | null;
  selection: SelectionSnapshot | null;
}

type FieldEntry = [string, RasterFieldManifest];

export function orderedFieldEntries(manifest: RunManifest): FieldEntry[] {
  const entries = Object.entries(manifest.fields);
  return [
    ...entries.filter(([, field]) => field.kind === "prediction"),
    ...entries.filter(([id]) => id === "loss_total"),
    ...entries.filter(([id, field]) => field.kind !== "prediction" && id !== "loss_total"),
  ];
}

export function captureVariantState(
  state: AppState,
  manifest: RunManifest | null,
  mainBounds: Bounds | null,
): VariantStateSnapshot {
  const field = state.fieldId && manifest ? manifest.fields[state.fieldId] : null;
  const matrix =
    state.matrixId && manifest
      ? manifest.influence_matrices.find((candidate) => candidate.id === state.matrixId)
      : null;
  return {
    fieldId: state.fieldId,
    fieldKind: field?.kind ?? null,
    matrixId: state.matrixId,
    matrixSignature: matrix ? matrixRestoreSignature(matrix) : null,
    selection: mainBounds ? captureSelection(state, mainBounds) : null,
  };
}

export function resolveRestoredFieldId(
  manifest: RunManifest,
  snapshot: VariantStateSnapshot | null,
): string | null {
  if (snapshot?.fieldId && manifest.fields[snapshot.fieldId]) return snapshot.fieldId;

  const ordered = orderedFieldEntries(manifest);
  if (snapshot?.fieldKind) {
    const sameKind = ordered.find(([, field]) => field.kind === snapshot.fieldKind);
    if (sameKind) return sameKind[0];
  }

  return defaultFieldId(manifest, ordered);
}

export function resolveRestoredMatrixId(
  manifest: RunManifest,
  snapshot: VariantStateSnapshot | null,
  preferredDefaultMatrixId: string,
): string | null {
  const matrices = manifest.influence_matrices;
  if (snapshot?.matrixId && matrices.some((matrix) => matrix.id === snapshot.matrixId)) {
    return snapshot.matrixId;
  }

  const signature = snapshot?.matrixSignature;
  if (signature) {
    const sameFullSignature = matrices.find(
      (matrix) =>
        matrix.method === signature.method &&
        matrix.left_term === signature.left_term &&
        matrix.right_term === signature.right_term &&
        matrix.row_source === signature.row_source,
    );
    if (sameFullSignature) return sameFullSignature.id;

    const sameTermsAndSource = matrices.find(
      (matrix) =>
        matrix.left_term === signature.left_term &&
        matrix.right_term === signature.right_term &&
        matrix.row_source === signature.row_source,
    );
    if (sameTermsAndSource) return sameTermsAndSource.id;

    const sameTerms = matrices.find(
      (matrix) =>
        matrix.left_term === signature.left_term && matrix.right_term === signature.right_term,
    );
    if (sameTerms) return sameTerms.id;
  }

  return defaultMatrixId(manifest, preferredDefaultMatrixId);
}

export function denormalizeSelection(
  selection: SelectionSnapshot | null,
  bounds: Bounds,
): SelectionSnapshot | null {
  if (!selection) return null;
  if (selection.mode === "point") {
    return { mode: "point", coord: denormalizePoint(selection.coord, bounds) };
  }
  return { mode: "region", region: denormalizeRegion(selection.region, bounds) };
}

function captureSelection(state: AppState, bounds: Bounds): SelectionSnapshot | null {
  if (state.selectionMode === "region" && state.selectedRegion) {
    return { mode: "region", region: normalizeRegion(state.selectedRegion, bounds) };
  }
  if (state.selectedCoord) {
    return { mode: "point", coord: normalizePoint(state.selectedCoord, bounds) };
  }
  return null;
}

function defaultFieldId(manifest: RunManifest, ordered = orderedFieldEntries(manifest)): string | null {
  if (manifest.default_field && manifest.fields[manifest.default_field]) {
    return manifest.default_field;
  }
  return ordered.find(([, field]) => field.kind === "prediction")?.[0] ?? ordered[0]?.[0] ?? null;
}

function defaultMatrixId(manifest: RunManifest, preferredDefaultMatrixId: string): string | null {
  const matrices = manifest.influence_matrices;
  const preferredDefault = matrices.find((matrix) => matrix.id === preferredDefaultMatrixId)?.id;
  if (preferredDefault) {
    return preferredDefault;
  }
  if (
    manifest.default_matrix &&
    matrices.some((matrix) => matrix.id === manifest.default_matrix)
  ) {
    return manifest.default_matrix;
  }
  return matrices[0]?.id ?? null;
}

function matrixRestoreSignature(matrix: InfluenceMatrixManifest): MatrixRestoreSignature {
  return {
    method: matrix.method,
    left_term: matrix.left_term,
    right_term: matrix.right_term,
    row_source: matrix.row_source,
  };
}

function normalizePoint(coord: [number, number], bounds: Bounds): [number, number] {
  const safeBounds = normalizedBounds(bounds);
  return [
    clamp01((coord[0] - safeBounds.minX) / positiveSpan(safeBounds.minX, safeBounds.maxX)),
    clamp01((coord[1] - safeBounds.minY) / positiveSpan(safeBounds.minY, safeBounds.maxY)),
  ];
}

function denormalizePoint(coord: [number, number], bounds: Bounds): [number, number] {
  const safeBounds = normalizedBounds(bounds);
  return [
    safeBounds.minX + clamp01(coord[0]) * positiveSpan(safeBounds.minX, safeBounds.maxX),
    safeBounds.minY + clamp01(coord[1]) * positiveSpan(safeBounds.minY, safeBounds.maxY),
  ];
}

function normalizeRegion(region: Bounds, bounds: Bounds): Bounds {
  const normalized = normalizeRegionBounds(region);
  const min = normalizePoint([normalized.minX, normalized.minY], bounds);
  const max = normalizePoint([normalized.maxX, normalized.maxY], bounds);
  return normalizeRegionBounds({
    minX: min[0],
    maxX: max[0],
    minY: min[1],
    maxY: max[1],
  });
}

function denormalizeRegion(region: Bounds, bounds: Bounds): Bounds {
  const normalized = normalizeRegionBounds(region);
  const min = denormalizePoint([normalized.minX, normalized.minY], bounds);
  const max = denormalizePoint([normalized.maxX, normalized.maxY], bounds);
  return normalizeRegionBounds({
    minX: min[0],
    maxX: max[0],
    minY: min[1],
    maxY: max[1],
  });
}

function positiveSpan(min: number, max: number): number {
  const span = max - min;
  return Number.isFinite(span) && span > 0 ? span : 1;
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.max(0, Math.min(1, value));
}
