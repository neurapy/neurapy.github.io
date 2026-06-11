import type {
  DataIndex,
  IndexProblemEntry,
  IndexVariantEntry,
  ModelQuality,
  RunManifest,
} from "../types";

export function resolveIndexUrl(): URL {
  const params = new URLSearchParams(window.location.search);
  const override = params.get("data");
  const configured = import.meta.env.VITE_PINNFLUENCE_INDEX_URL as string | undefined;
  return new URL(override || configured || "data/index.json", window.location.href);
}

export async function fetchJson<T>(url: URL, signal?: AbortSignal): Promise<T> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}: ${url.toString()}`);
  }
  return (await response.json()) as T;
}

export function assertV7Index(index: DataIndex): DataIndex {
  if (index.schema_version !== 7) {
    throw new Error(`Unsupported index schema ${String(index.schema_version)}; expected 7`);
  }
  if (!Array.isArray(index.problems)) {
    throw new Error("index.json is missing problems[]");
  }
  if ("runs" in index) {
    throw new Error("index.json uses deprecated runs[]; expected grouped problems[]");
  }
  for (const problem of index.problems) {
    if (!problem.problem || !problem.display_name || !problem.variants) {
      throw new Error("index.json has an invalid problem entry");
    }
    for (const quality of MODEL_QUALITIES) {
      const variant = problem.variants[quality];
      if (!variant?.manifest) {
        throw new Error(`${formatProblemLabel(problem.display_name)} is missing a ${quality} manifest`);
      }
      if (variant.problem !== problem.problem || variant.model_quality !== quality) {
        throw new Error(`${problem.problem}: invalid ${quality} variant metadata`);
      }
    }
  }
  return index;
}

export function assertV7RunManifest(manifest: RunManifest): RunManifest {
  if (manifest.schema_version !== 7) {
    throw new Error(`Unsupported run schema ${String(manifest.schema_version)}; expected 7`);
  }
  if (!manifest.problem || !manifest.display_name || !manifest.folder || !manifest.run_id) {
    throw new Error("Run manifest is missing required problem/run metadata");
  }
  if (!isModelQuality(manifest.model_quality)) {
    throw new Error(`Run manifest has invalid model_quality ${String(manifest.model_quality)}`);
  }
  if (!manifest.arrays?.candidate_points || !manifest.arrays?.train_points) {
    throw new Error("Run manifest is missing required point arrays");
  }
  if (!Array.isArray(manifest.influence_matrices)) {
    throw new Error("Run manifest is missing influence_matrices[]");
  }
  return manifest;
}

export async function loadIndex(indexUrl = resolveIndexUrl()): Promise<DataIndex> {
  return assertV7Index(await fetchJson<DataIndex>(indexUrl));
}

export async function loadRunManifest(indexUrl: URL, manifestPath: string): Promise<RunManifest> {
  return assertV7RunManifest(await fetchJson<RunManifest>(new URL(manifestPath, indexUrl)));
}

export const MODEL_QUALITIES: ModelQuality[] = ["good", "bad"];

export function isModelQuality(value: unknown): value is ModelQuality {
  return value === "good" || value === "bad";
}

export function qualityLabel(quality: ModelQuality): string {
  return quality === "good" ? "Good" : "Bad";
}

export function formatProblemLabel(problem: string): string {
  const normalized = problem
    .trim()
    .replace(/_float64(?:_(?:good|bad))?$/i, "")
    .replace(/_nd$/i, "")
    .replace(/\s+ND$/i, "")
    .replace(/_/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (!normalized) return "";
  return normalized
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1).toLowerCase())
    .join(" ");
}

export function firstAvailableProblem(index: DataIndex): IndexProblemEntry | null {
  return index.problems.find((problem) => problem.variants.good?.manifest) ?? null;
}

export function resolveProblemVariant(
  index: DataIndex,
  problemId: string | null,
  quality: ModelQuality,
): IndexVariantEntry | null {
  if (!problemId) return null;
  const problem = index.problems.find((entry) => entry.problem === problemId);
  return problem?.variants[quality] ?? null;
}
