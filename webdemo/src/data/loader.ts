import type { Priority } from "../types";

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
  private readonly fullResponsesByUrl = new Map<string, ArrayBuffer>();
  private requestId = 0;
  private subscriberId = 0;

  constructor(limits: Partial<Record<Priority, number>> = {}) {
    this.limits = {
      foreground: limits.foreground ?? 2,
      background: limits.background ?? 1,
    };
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

  pendingCount(priority?: Priority): number {
    if (priority) return this.queues[priority].length + this.active[priority].size;
    return this.pendingCount("foreground") + this.pendingCount("background");
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
        headers.set("Range", `bytes=${request.range.start}-${request.range.endExclusive - 1}`);
      }
      const response = await fetch(request.url, init);
      if (request.range && response.status !== 206 && response.status !== 200) {
        throw new Error(`${response.status} ${response.statusText}: ${request.url.toString()}`);
      }
      if (!request.range && !response.ok) {
        throw new Error(`${response.status} ${response.statusText}: ${request.url.toString()}`);
      }
      if (request.range) {
        if (response.status === 200) {
          const responseBuffer = await response.arrayBuffer();
          assertBinaryFullResponse(request.url, response, responseBuffer, request.range);
          this.fullResponsesByUrl.set(request.url.toString(), responseBuffer);
          buffer = sliceFullResponse(
            request.url,
            responseBuffer,
            request.range.start,
            request.range.endExclusive,
          );
        } else {
          const expectedBytes = request.range.endExclusive - request.range.start;
          let responseBuffer: ArrayBuffer | null = null;
          try {
            responseBuffer = await response.arrayBuffer();
          } catch (caught) {
            if (isAbortError(caught)) throw caught;
          }
          if (responseBuffer?.byteLength === expectedBytes) {
            buffer = responseBuffer;
          } else {
            buffer = await this.loadFullResponseSlice(request.url, request.range, controller.signal);
          }
        }
      } else {
        const responseBuffer = await response.arrayBuffer();
        buffer = responseBuffer;
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

  private async loadFullResponseSlice(
    url: URL,
    range: ByteRange,
    signal: AbortSignal,
  ): Promise<ArrayBuffer> {
    const cached = this.fullResponsesByUrl.get(url.toString());
    if (cached) return sliceFullResponse(url, cached, range.start, range.endExclusive);

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
    assertBinaryFullResponse(url, response, buffer, range);
    this.fullResponsesByUrl.set(url.toString(), buffer);
    return sliceFullResponse(url, buffer, range.start, range.endExclusive);
  }
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException
    ? error.name === "AbortError"
    : error instanceof Error && error.name === "AbortError";
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
  const contentType = response.headers.get("Content-Type")?.toLowerCase() ?? "";
  if (contentType.includes("text/html")) {
    throw new Error(`${url.toString()}: expected binary range response, got HTML fallback`);
  }
  if (range.endExclusive > buffer.byteLength) {
    throw new Error(
      `${url.toString()}: server ignored Range but full response is only ${buffer.byteLength} bytes; expected at least ${range.endExclusive}`,
    );
  }
}
