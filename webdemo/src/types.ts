export type DType = "float32" | "uint32" | "uint16" | "uint8" | "int16";
export type TypedArray =
  | Float32Array
  | Uint32Array
  | Uint16Array
  | Uint8Array
  | Int16Array;

export type Priority = "foreground" | "background";
export type FieldKind = "prediction" | "loss";
export type InfluenceSign = "abs" | "pos" | "neg";
export type BackgroundMode = "points" | "smooth" | "cell";
export type SelectionMode = "point" | "region";
export type ModelQuality = "good" | "bad";
export type AppView = "playground" | "results";

export interface ArraySpec {
  path: string;
  dtype: DType;
  shape: number[];
  bytes?: number;
}

export interface Bounds {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

export type AxisBounds = Record<string, [number, number]>;

export interface LinearEncoding {
  kind: "linear";
  min: number;
  max: number;
  missing?: number;
}

export interface RasterFieldManifest {
  label: string;
  kind: FieldKind;
  raster: ArraySpec;
  encoding: LinearEncoding;
  display_domain: [number, number];
}

export interface FieldRasterManifest {
  width: number;
  height: number;
  shape: [number, number];
  bounds: AxisBounds;
  axes: string[];
  max_axis_resolution?: number;
  coordinate_order?: Record<string, string>;
  mask: ArraySpec;
}

export interface DenseScoreLayout {
  kind: "dense_row_major";
  row_stride_bytes: number;
  data_offset_bytes: number;
}

export interface InfluenceMatrixManifest {
  id: string;
  method: string;
  left_term: string;
  right_term: string;
  num_pdes: number;
  num_bcs: number;
  n_outputs: number;
  self_influence: boolean;
  scores_shape: number[];
  source_scores_shape?: number[];
  candidate_points_shape?: number[];
  source_candidate_points_shape?: number[];
  row_source: "candidate_points" | "train_points";
  row_count: number;
  k: number;
  max_local_influence_points: number;
  label: string;
  display_label: string;
  scores: ArraySpec;
  score_layout: DenseScoreLayout;
}

export interface RunManifest {
  schema_version: 9;
  problem: string;
  display_name: string;
  model_quality: ModelQuality;
  folder: string;
  run_id: string;
  status: "complete" | "partial" | "incomplete" | "failed";
  errors: string[];
  generated_at: string;
  matrix_mode?: string;
  max_local_influence_points: number;
  axes: string[];
  bounds: AxisBounds;
  n_candidate: number;
  n_train: number;
  source_n_candidate?: number;
  source_n_train?: number;
  point_selection?: "deterministic_spread";
  n_outputs: number;
  num_pdes: number;
  num_bcs: number;
  available_terms: string[];
  term_labels: Record<string, string>;
  default_field: string | null;
  default_matrix: string | null;
  arrays: {
    candidate_points: ArraySpec;
    train_points: ArraySpec;
  };
  field_raster: FieldRasterManifest | null;
  fields: Record<string, RasterFieldManifest>;
  influence_matrices: InfluenceMatrixManifest[];
  validation?: Record<string, unknown>;
}

export interface IndexVariantEntry {
  run_id: string;
  display_name: string;
  problem: string;
  model_quality: ModelQuality;
  folder: string;
  n_candidate: number;
  n_train: number;
  source_n_candidate?: number;
  source_n_train?: number;
  status: RunManifest["status"];
  manifest: string | null;
  default_field: string | null;
  default_matrix: string | null;
  n_matrices?: number;
  n_fields?: number;
  errors?: string[];
}

export interface IndexProblemEntry {
  problem: string;
  display_name: string;
  variants: Record<ModelQuality, IndexVariantEntry>;
}

export interface DataIndex {
  schema_version: 9;
  generated_at: string;
  matrix_mode?: string;
  max_local_influence_points?: number;
  n_candidate?: number | null;
  n_train?: number | null;
  point_selection?: "deterministic_spread";
  raster_max_resolution?: number;
  bundle_report?: string;
  problems: IndexProblemEntry[];
}

export interface PointArrays {
  candidate_points: Float32Array;
  train_points: Float32Array;
}

export interface RasterData {
  fieldId: string;
  values: Uint16Array;
  mask: Uint8Array;
  encoding: LinearEncoding;
  displayDomain: [number, number];
  width: number;
  height: number;
}

export interface InfluenceRow {
  rowIndex: number;
  indices: Uint16Array | Uint32Array;
  values: Float32Array;
}

export interface InfluenceAggregate {
  rowIndices: number[];
  indices: Uint32Array;
  values: Float32Array;
  meanValue: number;
}

export interface PlotViewport {
  x: number;
  y: number;
  width: number;
  height: number;
  right: number;
  bottom: number;
}

export interface LossDecompositionTermSeries {
  id: string;
  label: string;
  mean_fraction: number;
  std_fraction: number;
  binned_fraction: number[];
  binned_fraction_std: number[];
}

export interface LossDecompositionOutput {
  id: string;
  label: string;
  mean_coherence: number;
  std_coherence: number;
  binned_coherence: number[];
  bin_centers: number[];
  terms: LossDecompositionTermSeries[];
}

export interface LossDecompositionData {
  problem: string;
  display_name: string;
  quality: ModelQuality;
  source_kind: "aggregate_summary";
  axis: {
    id: string;
    label: string;
  };
  outputs: LossDecompositionOutput[];
}

export interface PaperIndicatorValue {
  mean: number;
  std: number;
}

export interface TemporalIndicatorEntry {
  problem: string;
  display_name: string;
  baseline: number;
  bad_baseline?: number;
  values: Record<ModelQuality, PaperIndicatorValue>;
  note?: string;
}

export interface DirectionalityIndicatorEntry {
  problem: string;
  display_name: string;
  output_id: string;
  output_label: string;
  baseline: number;
  values: Record<ModelQuality, PaperIndicatorValue>;
}

export interface ResultsData {
  schema_version: 1;
  generated_at: string;
  sources: string[];
  loss_decompositions: LossDecompositionData[];
  indicators: {
    temporal: TemporalIndicatorEntry[];
    directionality: DirectionalityIndicatorEntry[];
  };
}
