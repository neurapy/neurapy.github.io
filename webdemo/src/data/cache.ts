export interface CacheEntry<T> {
  value: T;
  bytes: number;
}

export class LruCache<T> {
  readonly maxBytes: number;
  private readonly entries = new Map<string, CacheEntry<T>>();
  private usedBytesValue = 0;

  constructor(maxBytes = 128 * 1024 * 1024) {
    this.maxBytes = maxBytes;
  }

  get usedBytes(): number {
    return this.usedBytesValue;
  }

  get size(): number {
    return this.entries.size;
  }

  get(key: string): T | undefined {
    const entry = this.entries.get(key);
    if (!entry) return undefined;
    this.entries.delete(key);
    this.entries.set(key, entry);
    return entry.value;
  }

  set(key: string, value: T, bytes: number): void {
    const existing = this.entries.get(key);
    if (existing) {
      this.usedBytesValue -= existing.bytes;
      this.entries.delete(key);
    }
    this.entries.set(key, { value, bytes });
    this.usedBytesValue += bytes;
    this.evict();
  }

  has(key: string): boolean {
    return this.entries.has(key);
  }

  clear(): void {
    this.entries.clear();
    this.usedBytesValue = 0;
  }

  private evict(): void {
    while (this.usedBytesValue > this.maxBytes && this.entries.size > 1) {
      const oldestKey = this.entries.keys().next().value as string | undefined;
      if (!oldestKey) return;
      const oldest = this.entries.get(oldestKey);
      this.entries.delete(oldestKey);
      this.usedBytesValue -= oldest?.bytes ?? 0;
    }
  }
}
