import type {
  ArraySpec,
  LossDecompositionArrayBundle,
  LossDecompositionData,
  LossDecompositionOutput,
  LossDecompositionOutputMetadata,
  ResultsData,
  ResultsIndexData,
} from "../types";
import { assertDType, typedArrayFromBuffer } from "./dtypes";
import { fetchJson } from "./manifest";

const RESULTS_SCHEMA_VERSION = 2;
const ARRAY_KEYS = [
  "bin_centers",
  "binned_fractions",
  "binned_fractions_std",
  "binned_coherence",
  "binned_coherence_std",
] as const satisfies readonly (keyof LossDecompositionArrayBundle)[];

export function resolveResultsUrl(indexUrl: URL): URL {
  return new URL("results/index.json", indexUrl);
}

export async function loadResultsData(indexUrl: URL, signal?: AbortSignal): Promise<ResultsData> {
  const resultsUrl = resolveResultsUrl(indexUrl);
  const index = assertResultsData(await fetchJson<ResultsIndexData>(resultsUrl, signal));
  return materializeResultsData(index, resultsUrl, signal);
}

export function assertResultsData(data: ResultsIndexData): ResultsIndexData {
  if (data.schema_version !== RESULTS_SCHEMA_VERSION) {
    throw new Error(
      `Unsupported results schema ${String(data.schema_version)}; expected ${RESULTS_SCHEMA_VERSION}`,
    );
  }
  if (!Array.isArray(data.loss_decompositions)) {
    throw new Error("Results data is missing loss decomposition summaries");
  }
  if (
    !data.indicators ||
    !Array.isArray(data.indicators.temporal) ||
    !Array.isArray(data.indicators.directionality)
  ) {
    throw new Error("Results data is missing paper indicator summaries");
  }
  for (const entry of data.loss_decompositions) {
    if (!entry.problem || !entry.display_name || (entry.quality !== "good" && entry.quality !== "bad")) {
      throw new Error("Results data has invalid problem/quality metadata");
    }
    if (entry.source_kind !== "full_matrix") {
      throw new Error(`${entry.problem}/${entry.quality}: expected full_matrix results source`);
    }
    if (!entry.axis?.id || !entry.axis.label) {
      throw new Error(`${entry.problem}/${entry.quality}: missing axis metadata`);
    }
    if (!Array.isArray(entry.outputs)) {
      throw new Error(`${entry.problem}/${entry.quality}: missing outputs[]`);
    }
    for (const output of entry.outputs) {
      validateOutputMetadata(entry.problem, entry.quality, output);
    }
  }
  return data;
}

async function materializeResultsData(
  index: ResultsIndexData,
  resultsUrl: URL,
  signal?: AbortSignal,
): Promise<ResultsData> {
  return {
    schema_version: RESULTS_SCHEMA_VERSION,
    generated_at: index.generated_at,
    sources: index.sources,
    indicators: index.indicators,
    loss_decompositions: await Promise.all(
      index.loss_decompositions.map(async (entry): Promise<LossDecompositionData> => ({
        problem: entry.problem,
        display_name: entry.display_name,
        quality: entry.quality,
        source_kind: entry.source_kind,
        axis: entry.axis,
        outputs: await Promise.all(
          entry.outputs.map((output) => materializeOutput(output, resultsUrl, signal)),
        ),
      })),
    ),
  };
}

async function materializeOutput(
  output: LossDecompositionOutputMetadata,
  resultsUrl: URL,
  signal?: AbortSignal,
): Promise<LossDecompositionOutput> {
  const [
    binCenters,
    binnedFractions,
    binnedFractionsStd,
    binnedCoherence,
    binnedCoherenceStd,
  ] = await Promise.all([
    loadFloat32Array(output.arrays.bin_centers, resultsUrl, signal),
    loadFloat32Array(output.arrays.binned_fractions, resultsUrl, signal),
    loadFloat32Array(output.arrays.binned_fractions_std, resultsUrl, signal),
    loadFloat32Array(output.arrays.binned_coherence, resultsUrl, signal),
    loadFloat32Array(output.arrays.binned_coherence_std, resultsUrl, signal),
  ]);
  const nBins = output.n_bins;
  return {
    id: output.id,
    label: output.label,
    mean_coherence: output.mean_coherence,
    std_coherence: output.std_coherence,
    binned_coherence: Array.from(binnedCoherence),
    binned_coherence_std: Array.from(binnedCoherenceStd),
    bin_centers: Array.from(binCenters),
    n_candidate: output.n_candidate,
    n_train: output.n_train,
    source_matrix_ids: [...output.source_matrix_ids],
    terms: output.terms.map((term, termIndex) => ({
      ...term,
      binned_fraction: Array.from(binnedFractions.slice(termIndex * nBins, (termIndex + 1) * nBins)),
      binned_fraction_std: Array.from(
        binnedFractionsStd.slice(termIndex * nBins, (termIndex + 1) * nBins),
      ),
    })),
  };
}

async function loadFloat32Array(
  spec: ArraySpec,
  resultsUrl: URL,
  signal?: AbortSignal,
): Promise<Float32Array> {
  assertDType(spec, "float32");
  const response = await fetch(new URL(spec.path, resultsUrl), { signal });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${new URL(spec.path, resultsUrl).toString()}`);
  }
  const array = typedArrayFromBuffer(spec, await response.arrayBuffer());
  if (!(array instanceof Float32Array)) {
    throw new Error(`${spec.path}: expected Float32Array`);
  }
  return array;
}

function validateOutputMetadata(problem: string, quality: string, output: LossDecompositionOutputMetadata): void {
  const context = `${problem}/${quality}/${output.id || "output"}`;
  if (!output.id || !output.label) {
    throw new Error(`${context}: missing output metadata`);
  }
  if (!Number.isInteger(output.n_bins) || output.n_bins <= 0) {
    throw new Error(`${context}: n_bins must be a positive integer`);
  }
  if (!Array.isArray(output.terms) || output.terms.length !== output.n_terms) {
    throw new Error(`${context}: terms[] length must match n_terms`);
  }
  if (!Array.isArray(output.source_matrix_ids) || output.source_matrix_ids.length !== output.n_terms) {
    throw new Error(`${context}: source_matrix_ids length must match n_terms`);
  }
  validateArraySpec(context, "bin_centers", output.arrays?.bin_centers, [output.n_bins]);
  validateArraySpec(context, "binned_fractions", output.arrays?.binned_fractions, [
    output.n_terms,
    output.n_bins,
  ]);
  validateArraySpec(context, "binned_fractions_std", output.arrays?.binned_fractions_std, [
    output.n_terms,
    output.n_bins,
  ]);
  validateArraySpec(context, "binned_coherence", output.arrays?.binned_coherence, [output.n_bins]);
  validateArraySpec(context, "binned_coherence_std", output.arrays?.binned_coherence_std, [
    output.n_bins,
  ]);
}

function validateArraySpec(
  context: string,
  key: (typeof ARRAY_KEYS)[number],
  spec: ArraySpec | undefined,
  shape: number[],
): void {
  if (!spec) {
    throw new Error(`${context}: missing ${key} array spec`);
  }
  assertDType(spec, "float32");
  if (!spec.path || !Array.isArray(spec.shape) || spec.shape.length !== shape.length) {
    throw new Error(`${context}: invalid ${key} array spec`);
  }
  if (!shape.every((value, index) => spec.shape[index] === value)) {
    throw new Error(`${context}: ${key} shape must be [${shape.join(", ")}]`);
  }
  const expectedBytes = shape.reduce((product, value) => product * value, 1) * Float32Array.BYTES_PER_ELEMENT;
  if (spec.bytes != null && spec.bytes !== expectedBytes) {
    throw new Error(`${context}: ${key} expected ${expectedBytes} bytes, got ${spec.bytes}`);
  }
}
