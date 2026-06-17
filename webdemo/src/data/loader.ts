import type { Priority } from "../types";
import { LruCache } from "./cache";

const DEFAULT_FULL_RESPONSE_CACHE_BYTES = 64 * 1024 * 1024;

export type PriorityLoaderOptions = Partial<Record<Priority, number>> & {
  fullResponseCacheBytes?: number;
};

interface Subscriber {
  id: number;
  resolve: (buffer: ArrayBuffer) => void;
  reject: (error: unknown) => void;
  abortHandler?: () => void;
}

interface ByteRange {
  start: number;
  endExclusive: number;
}

interface QueuedRequest {
  id: number;
  url: URL;
  key: string;
  range?: ByteRange;
  priority: Priority;
  state: "queued" | "active";
  activePriority?: Priority;
  controller?: AbortController;
  requeueOnAbort: boolean;
  subscribers: Map<number, Subscriber>;
}

export class PriorityLoader {
  private readonly limits: Record<Priority, number>;
  private readonly active: Record<Priority, Set<QueuedRequest>> = {
    foreground: new Set(),
    background: new Set(),
  };
  private readonly queues: Record<Priority, QueuedRequest[]> = {
    foreground: [],
    background: [],
  };
  private readonly requestsByKey = new Map<string, QueuedRequest>();
  private readonly fullResponsesByUrl: LruCache<ArrayBuffer>;
  private readonly fullResponsePromisesByUrl = new Map<string, Promise<ArrayBuffer>>();
  private readonly brokenRangeUrls = new Set<string>();
  private requestId = 0;
  private subscriberId = 0;

  constructor(options: PriorityLoaderOptions = {}) {
    this.limits = {
      foreground: options.foreground ?? 2,
      background: options.background ?? 1,
    };
    this.fullResponsesByUrl = new LruCache<ArrayBuffer>(
      options.fullResponseCacheBytes ?? DEFAULT_FULL_RESPONSE_CACHE_BYTES,
    );
  }

  load(url: URL, priority: Priority = "foreground", externalSignal?: AbortSignal): Promise<ArrayBuffer> {
    return this.enqueue(url, undefined, priority, externalSignal);
  }

  loadRange(
    url: URL,
    start: number,
    endExclusive: number,
    priority: Priority = "foreground",
    externalSignal?: AbortSignal,
  ): Promise<ArrayBuffer> {
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(endExclusive) || start < 0 || endExclusive <= start) {
      return Promise.reject(new Error(`Invalid byte range ${start}-${endExclusive}`));
    }
    const full = this.fullResponsesByUrl.get(url.toString());
    if (full) {
      return Promise.resolve(sliceFullResponse(url, full, start, endExclusive));
    }
    return this.enqueue(url, { start, endExclusive }, priority, externalSignal);
  }

  private enqueue(
    url: URL,
    range: ByteRange | undefined,
    priority: Priority,
    externalSignal?: AbortSignal,
  ): Promise<ArrayBuffer> {
    return new Promise((resolve, reject) => {
      const key = requestKey(url, range);
      let request = this.requestsByKey.get(key);
      if (!request) {
        request = {
          id: ++this.requestId,
          url,
          key,
          range,
          priority,
          state: "queued",
          requeueOnAbort: false,
          subscribers: new Map(),
        };
        this.requestsByKey.set(key, request);
        this.queues[priority].push(request);
      }

      const subscriber: Subscriber = {
        id: ++this.subscriberId,
        resolve,
        reject,
      };
      request.subscribers.set(subscriber.id, subscriber);

      if (externalSignal) {
        if (externalSignal.aborted) {
          this.abortSubscriber(request, subscriber);
          return;
        }
        subscriber.abortHandler = () => this.abortSubscriber(request, subscriber);
        externalSignal.addEventListener("abort", subscriber.abortHandler, { once: true });
      }

      if (priority === "foreground") {
        this.promoteToForeground(request);
        this.preemptActiveBackground();
      }
      this.pump();
    });
  }

  abortBackground(): void {
    const queued = this.queues.background.splice(0);
    for (const request of queued) this.failRequest(request, new DOMException("Background request aborted", "AbortError"));
    for (const request of this.active.background) {
      request.requeueOnAbort = false;
      request.controller?.abort();
    }
  }

  private pump(): void {
    this.startAvailable("foreground");
    if (this.queues.foreground.length || this.active.foreground.size) return;
    this.startAvailable("background");
  }

  private startAvailable(priority: Priority): void {
    while (this.active[priority].size < this.limits[priority] && this.queues[priority].length) {
      const request = this.queues[priority].shift();
      if (!request) return;
      if (!request.subscribers.size) {
        this.requestsByKey.delete(request.key);
        continue;
      }
      request.priority = priority;
      request.state = "active";
      request.activePriority = priority;
      request.controller = new AbortController();
      request.requeueOnAbort = false;
      this.active[priority].add(request);
      void this.run(request);
    }
  }

  private async run(request: QueuedRequest): Promise<void> {
    const activePriority = request.activePriority;
    const controller = request.controller;
    if (!activePriority || !controller) return;
    let buffer: ArrayBuffer | null = null;
    let error: unknown = null;
    try {
      const headers = new Headers({
        Accept: "application/octet-stream",
      });
      const init: RequestInit = { signal: controller.signal, headers };
      if (request.range) {
        const urlKey = request.url.toString();
        if (this.brokenRangeUrls.has(urlKey)) {
          buffer = await this.loadFullResponseSliceForRange(
            request.url,
            request.range,
            controller.signal,
            activePriority,
          );
        } else {
          headers.set("Range", `bytes=${request.range.start}-${request.range.endExclusive - 1}`);
          const response = await fetch(request.url, init);
          if (response.status !== 206 && response.status !== 200) {
            throw new Error(`${response.status} ${response.statusText}: ${request.url.toString()}`);
          }
          if (response.status === 200) {
            const responseBuffer = await response.arrayBuffer();
            assertBinaryFullResponse(request.url, response, responseBuffer, request.range);
            this.fullResponsesByUrl.set(request.url.toString(), responseBuffer, responseBuffer.byteLength);
            buffer = sliceFullResponse(
              request.url,
              responseBuffer,
              request.range.start,
              request.range.endExclusive,
            );
          } else {
            buffer = await this.readPartialResponseOrFallback(
              request.url,
              request.range,
              response,
              controller.signal,
              activePriority,
            );
          }
        }
      } else {
        const response = await fetch(request.url, init);
        if (!response.ok) {
          throw new Error(`${response.status} ${response.statusText}: ${request.url.toString()}`);
        }
        buffer = await response.arrayBuffer();
      }
    } catch (caught) {
      error = caught;
    } finally {
      this.active[activePriority].delete(request);
      request.activePriority = undefined;
      request.controller = undefined;
    }

    if (buffer) {
      this.resolveRequest(request, buffer);
    } else if (isAbortError(error) && request.requeueOnAbort && request.subscribers.size) {
      request.state = "queued";
      request.requeueOnAbort = false;
      this.queues[request.priority].push(request);
    } else {
      this.failRequest(request, error ?? new DOMException("Request aborted", "AbortError"));
    }

    this.pump();
  }

  private promoteToForeground(request: QueuedRequest): void {
    if (request.priority === "foreground") return;
    request.priority = "foreground";
    if (request.state === "queued") {
      this.removeQueued(request, "background");
      this.queues.foreground.push(request);
      return;
    }
    if (request.activePriority === "background") {
      request.requeueOnAbort = true;
      request.controller?.abort();
    }
  }

  private preemptActiveBackground(): void {
    for (const request of this.active.background) {
      if (!request.subscribers.size) continue;
      request.requeueOnAbort = true;
      request.controller?.abort();
    }
  }

  private abortSubscriber(request: QueuedRequest, subscriber: Subscriber): void {
    if (!request.subscribers.delete(subscriber.id)) return;
    subscriber.reject(new DOMException("Request aborted", "AbortError"));
    if (request.subscribers.size) return;
    if (request.state === "queued") {
      this.removeQueued(request, request.priority);
      this.requestsByKey.delete(request.key);
      return;
    }
    request.requeueOnAbort = false;
    request.controller?.abort();
  }

  private resolveRequest(request: QueuedRequest, buffer: ArrayBuffer): void {
    this.requestsByKey.delete(request.key);
    for (const subscriber of request.subscribers.values()) {
      subscriber.resolve(buffer);
    }
    request.subscribers.clear();
  }

  private failRequest(request: QueuedRequest, error: unknown): void {
    this.requestsByKey.delete(request.key);
    request.state = "queued";
    this.removeQueued(request, "foreground");
    this.removeQueued(request, "background");
    for (const subscriber of request.subscribers.values()) {
      subscriber.reject(error);
    }
    request.subscribers.clear();
  }

  private removeQueued(request: QueuedRequest, priority: Priority): void {
    const queue = this.queues[priority];
    const index = queue.indexOf(request);
    if (index >= 0) queue.splice(index, 1);
  }

  private async readPartialResponseOrFallback(
    url: URL,
    range: ByteRange,
    response: Response,
    signal: AbortSignal,
    priority: Priority,
  ): Promise<ArrayBuffer> {
    const expectedBytes = range.endExclusive - range.start;
    try {
      const responseBuffer = await response.arrayBuffer();
      if (responseBuffer.byteLength === expectedBytes) return responseBuffer;
    } catch (caught) {
      if (isAbortError(caught)) throw caught;
      if (!isDecodingFailedError(caught)) throw caught;
    }
    return this.loadFullResponseSliceForRange(url, range, signal, priority);
  }

  private async loadFullResponseSliceForRange(
    url: URL,
    range: ByteRange,
    signal: AbortSignal,
    priority: Priority,
  ): Promise<ArrayBuffer> {
    const key = url.toString();
    this.brokenRangeUrls.add(key);
    if (
      priority === "background" &&
      !this.fullResponsesByUrl.has(key) &&
      !this.fullResponsePromisesByUrl.has(key)
    ) {
      throw new Error(`${url.toString()}: skipping background full response fallback for broken range response`);
    }
    return this.loadFullResponseSlice(url, range, signal);
  }

  private async loadFullResponseSlice(
    url: URL,
    range: ByteRange,
    signal: AbortSignal,
  ): Promise<ArrayBuffer> {
    const cached = this.fullResponsesByUrl.get(url.toString());
    if (cached) return sliceFullResponse(url, cached, range.start, range.endExclusive);

    const buffer = await this.loadFullResponse(url, signal);
    return sliceFullResponse(url, buffer, range.start, range.endExclusive);
  }

  private async loadFullResponse(url: URL, signal: AbortSignal): Promise<ArrayBuffer> {
    const key = url.toString();
    const cached = this.fullResponsesByUrl.get(key);
    if (cached) return cached;

    let pending = this.fullResponsePromisesByUrl.get(key);
    if (!pending) {
      pending = this.fetchFullResponse(url, signal);
      this.fullResponsePromisesByUrl.set(key, pending);
      pending.then(
        () => this.fullResponsePromisesByUrl.delete(key),
        () => this.fullResponsePromisesByUrl.delete(key),
      );
    }
    return pending;
  }

  private async fetchFullResponse(url: URL, signal: AbortSignal): Promise<ArrayBuffer> {
    const response = await fetch(url, {
      signal,
      headers: new Headers({ Accept: "application/octet-stream" }),
    });
    if (!response.ok) {
      throw new Error(`${response.status} ${response.statusText}: ${url.toString()}`);
    }
    if (response.status !== 200) {
      throw new Error(`${url.toString()}: expected full binary response, got HTTP ${response.status}`);
    }
    const buffer = await response.arrayBuffer();
    assertBinaryResponse(url, response);
    this.fullResponsesByUrl.set(url.toString(), buffer, buffer.byteLength);
    return buffer;
  }
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException
    ? error.name === "AbortError"
    : error instanceof Error && error.name === "AbortError";
}

function isDecodingFailedError(error: unknown): boolean {
  return error instanceof TypeError && /decoding failed/i.test(error.message);
}

function requestKey(url: URL, range?: ByteRange): string {
  if (!range) return url.toString();
  return `${url.toString()}#bytes=${range.start}-${range.endExclusive}`;
}

function sliceFullResponse(
  url: URL,
  buffer: ArrayBuffer,
  start: number,
  endExclusive: number,
): ArrayBuffer {
  if (endExclusive > buffer.byteLength) {
    throw new Error(
      `${url.toString()}: range ${start}-${endExclusive} exceeds ${buffer.byteLength} response bytes`,
    );
  }
  return buffer.slice(start, endExclusive);
}

function assertBinaryFullResponse(
  url: URL,
  response: Response,
  buffer: ArrayBuffer,
  range: ByteRange,
): void {
  assertBinaryResponse(url, response);
  if (range.endExclusive > buffer.byteLength) {
    throw new Error(
      `${url.toString()}: server ignored Range but full response is only ${buffer.byteLength} bytes; expected at least ${range.endExclusive}`,
    );
  }
}

function assertBinaryResponse(url: URL, response: Response): void {
  const contentType = response.headers.get("Content-Type")?.toLowerCase() ?? "";
  if (contentType.includes("text/html")) {
    throw new Error(`${url.toString()}: expected binary response, got HTML fallback`);
  }
}
