import type { Priority } from "../types";

interface QueuedRequest {
  id: number;
  url: URL;
  priority: Priority;
  controller: AbortController;
  resolve: (buffer: ArrayBuffer) => void;
  reject: (error: unknown) => void;
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
  private requestId = 0;

  constructor(limits: Partial<Record<Priority, number>> = {}) {
    this.limits = {
      foreground: limits.foreground ?? 2,
      background: limits.background ?? 1,
    };
  }

  load(url: URL, priority: Priority = "foreground", externalSignal?: AbortSignal): Promise<ArrayBuffer> {
    const controller = new AbortController();
    if (externalSignal) {
      if (externalSignal.aborted) controller.abort();
      externalSignal.addEventListener("abort", () => controller.abort(), { once: true });
    }

    return new Promise((resolve, reject) => {
      const request: QueuedRequest = {
        id: ++this.requestId,
        url,
        priority,
        controller,
        resolve,
        reject,
      };
      this.queues[priority].push(request);
      this.pump();
    });
  }

  abortBackground(): void {
    for (const request of this.queues.background.splice(0)) {
      request.controller.abort();
      request.reject(new DOMException("Background request aborted", "AbortError"));
    }
    for (const request of this.active.background) {
      request.controller.abort();
    }
  }

  pendingCount(priority?: Priority): number {
    if (priority) return this.queues[priority].length + this.active[priority].size;
    return this.pendingCount("foreground") + this.pendingCount("background");
  }

  private pump(): void {
    this.startAvailable("foreground");
    this.startAvailable("background");
  }

  private startAvailable(priority: Priority): void {
    while (this.active[priority].size < this.limits[priority] && this.queues[priority].length) {
      const request = this.queues[priority].shift();
      if (!request) return;
      if (request.controller.signal.aborted) {
        request.reject(new DOMException("Request aborted", "AbortError"));
        continue;
      }
      this.active[priority].add(request);
      void this.run(request);
    }
  }

  private async run(request: QueuedRequest): Promise<void> {
    try {
      const response = await fetch(request.url, { signal: request.controller.signal });
      if (!response.ok) {
        throw new Error(`${response.status} ${response.statusText}: ${request.url.toString()}`);
      }
      request.resolve(await response.arrayBuffer());
    } catch (error) {
      request.reject(error);
    } finally {
      this.active[request.priority].delete(request);
      this.pump();
    }
  }
}
