import type { Priority } from "../types";

interface Subscriber {
  id: number;
  resolve: (buffer: ArrayBuffer) => void;
  reject: (error: unknown) => void;
  abortHandler?: () => void;
}

interface QueuedRequest {
  id: number;
  url: URL;
  key: string;
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
  private requestId = 0;
  private subscriberId = 0;

  constructor(limits: Partial<Record<Priority, number>> = {}) {
    this.limits = {
      foreground: limits.foreground ?? 2,
      background: limits.background ?? 1,
    };
  }

  load(url: URL, priority: Priority = "foreground", externalSignal?: AbortSignal): Promise<ArrayBuffer> {
    return new Promise((resolve, reject) => {
      const key = url.toString();
      let request = this.requestsByKey.get(key);
      if (!request) {
        request = {
          id: ++this.requestId,
          url,
          key,
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
    if (!activePriority || !request.controller) return;
    let buffer: ArrayBuffer | null = null;
    let error: unknown = null;
    try {
      const response = await fetch(request.url, { signal: request.controller.signal });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}: ${request.url.toString()}`);
      }
      buffer = await response.arrayBuffer();
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
}

function isAbortError(error: unknown): boolean {
  return error instanceof DOMException
    ? error.name === "AbortError"
    : error instanceof Error && error.name === "AbortError";
}
