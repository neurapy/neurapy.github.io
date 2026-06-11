import type { DataIndex, RunManifest } from "../types";

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

export function assertV6Index(index: DataIndex): DataIndex {
  if (index.schema_version !== 6) {
    throw new Error(`Unsupported index schema ${String(index.schema_version)}; expected 6`);
  }
  if (!Array.isArray(index.runs)) {
    throw new Error("index.json is missing runs[]");
  }
  return index;
}

export function assertV6RunManifest(manifest: RunManifest): RunManifest {
  if (manifest.schema_version !== 6) {
    throw new Error(`Unsupported run schema ${String(manifest.schema_version)}; expected 6`);
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
  return assertV6Index(await fetchJson<DataIndex>(indexUrl));
}

export async function loadRunManifest(indexUrl: URL, manifestPath: string): Promise<RunManifest> {
  return assertV6RunManifest(await fetchJson<RunManifest>(new URL(manifestPath, indexUrl)));
}
