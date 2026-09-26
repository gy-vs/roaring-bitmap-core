import {
  ArrayContainer,
  BitmapContainer,
  RunContainer,
  Container,
  TYPE_ARRAY,
  TYPE_BITMAP,
  TYPE_RUN,
  ARRAY_LIMIT,
  canonicalizeArray,
  canonicalizeBitmap,
  containerEquals,
  runEnd,
} from "./containers";
import { containerOp, SetOp } from "./ops";
import { encodeSet, decodeSet, DeserializationError, SERIALIZATION_VERSION } from "./serialize";
import { ctz32, wordRangeMask } from "./bits";

const MAX_UINT32 = 0xffffffff;

function checkUint32(v: number, name: string): void {
  if (!Number.isInteger(v) || v < 0 || v > MAX_UINT32) {
    throw new RangeError(`${name} must be an integer in [0, 2^32 - 1], got ${v}`);
  }
}

/** Container for a fresh shard fully covered by [lo, hi]. */
function newShardFromRange(lo: number, hi: number): Container {
  const card = hi - lo + 1;
  if (card <= ARRAY_LIMIT) {
    const values = new Uint16Array(Math.max(4, card));
    for (let i = 0; i < card; i++) values[i] = lo + i;
    return new ArrayContainer(values, card);
  }
  return new RunContainer([{ start: lo, length: card }], card);
}

export interface ContainerStats {
  shards: number;
  array: number;
  bitmap: number;
  run: number;
  cardinality: number;
}

/**
 * A sorted set of unsigned 32-bit integers.
 *
 * Values are sharded by their high 16 bits; each shard stores its low 16
 * bits in an array, bitmap, or run container that is converted based on
 * density. All iteration is in ascending order.
 */
export class Roaring32 implements Iterable<number> {
  private shards: Map<number, Container> = new Map();
  private cardinality = 0;
  private keysCache: number[] | null = null;
  private activeBuffers = 0;

  constructor(values?: Iterable<number>) {
    if (values !== undefined) this.addAll(values);
  }

  static from(values: Iterable<number>): Roaring32 {
    return new Roaring32(values);
  }

  /** Number of elements in the set. */
  get size(): number {
    return this.cardinality;
  }

  get isEmpty(): boolean {
    return this.cardinality === 0;
  }

  /** Number of non-empty high-16-bit shards. */
  get shardCount(): number {
    return this.shards.size;
  }

  /**
   * Number of temporary iterator buffers currently checked out.
   * Diagnostic hook: a cancelled range iterator must release its buffer,
   * bringing this back to zero.
   */
  get pendingIteratorBuffers(): number {
    return this.activeBuffers;
  }

  /** Container mix, for introspection and tests. */
  stats(): ContainerStats {
    let array = 0;
    let bitmap = 0;
    let run = 0;
    for (const c of this.shards.values()) {
      if (c.type === TYPE_ARRAY) array++;
      else if (c.type === TYPE_BITMAP) bitmap++;
      else run++;
    }
    return { shards: this.shards.size, array, bitmap, run, cardinality: this.cardinality };
  }

  /* ------------------------------ mutation ------------------------------ */

  /** Add a value; returns true if it was not already present. */
  add(value: number): boolean {
    checkUint32(value, "value");
    const key = value >>> 16;
    const low = value & 0xffff;
    const c = this.shards.get(key);
    if (c === undefined) {
      this.shards.set(key, ArrayContainer.of(low));
      this.cardinality++;
      this.keysCache = null;
      return true;
    }
    const before = c.cardinality;
    const next = c.add(low);
    if (next !== c) this.shards.set(key, next);
    const delta = next.cardinality - before;
    this.cardinality += delta;
    return delta > 0;
  }

  /** Add every value in the inclusive range [start, end]. */
  addRange(start: number, end: number): void {
    checkUint32(start, "start");
    checkUint32(end, "end");
    if (end < start) throw new RangeError(`end (${end}) must be >= start (${start})`);
    const k0 = start >>> 16;
    const k1 = end >>> 16;
    for (let k = k0; k <= k1; k++) {
      const lo = k === k0 ? start & 0xffff : 0;
      const hi = k === k1 ? end & 0xffff : 0xffff;
      const c = this.shards.get(k);
      const before = c === undefined ? 0 : c.cardinality;
      const next = c === undefined ? newShardFromRange(lo, hi) : c.addRange(lo, hi);
      if (c === undefined) this.keysCache = null;
      if (next !== c) this.shards.set(k, next);
      this.cardinality += next.cardinality - before;
    }
  }

  /** Remove a value; returns true if it was present. */
  delete(value: number): boolean {
    checkUint32(value, "value");
    const key = value >>> 16;
    const c = this.shards.get(key);
    if (c === undefined) return false;
    const before = c.cardinality;
    const next = c.delete(value & 0xffff);
    this.cardinality += next.cardinality - before;
    if (next.cardinality === 0) {
      this.shards.delete(key);
      this.keysCache = null;
    } else if (next !== c) {
      this.shards.set(key, next);
    }
    return next.cardinality < before;
  }

  /** Remove every value in the inclusive range [start, end]. */
  deleteRange(start: number, end: number): void {
    checkUint32(start, "start");
    checkUint32(end, "end");
    if (end < start) throw new RangeError(`end (${end}) must be >= start (${start})`);
    const k0 = start >>> 16;
    const k1 = end >>> 16;
    for (let k = k0; k <= k1; k++) {
      const c = this.shards.get(k);
      if (c === undefined) continue;
      const lo = k === k0 ? start & 0xffff : 0;
      const hi = k === k1 ? end & 0xffff : 0xffff;
      const before = c.cardinality;
      const next = c.deleteRange(lo, hi);
      this.cardinality += next.cardinality - before;
      if (next.cardinality === 0) {
        this.shards.delete(k);
        this.keysCache = null;
      } else if (next !== c) {
        this.shards.set(k, next);
      }
    }
  }

  clear(): void {
    this.shards.clear();
    this.cardinality = 0;
    this.keysCache = null;
  }

  /* ------------------------------ queries ------------------------------- */

  has(value: number): boolean {
    checkUint32(value, "value");
    const c = this.shards.get(value >>> 16);
    return c !== undefined && c.has(value & 0xffff);
  }

  /** Number of elements <= value. */
  rank(value: number): number {
    checkUint32(value, "value");
    const key = value >>> 16;
    const low = value & 0xffff;
    let r = 0;
    for (const k of this.sortedKeys()) {
      if (k > key) break;
      const c = this.shards.get(k)!;
      if (k === key) {
        r += c.rank(low);
        break;
      }
      r += c.cardinality;
    }
    return r;
  }

  /** The element at sorted position `index` (0-based). Throws if out of bounds. */
  select(index: number): number {
    if (!Number.isInteger(index) || index < 0 || index >= this.cardinality) {
      throw new RangeError(`index ${index} out of bounds for set of size ${this.cardinality}`);
    }
    let i = index;
    for (const k of this.sortedKeys()) {
      const c = this.shards.get(k)!;
      if (i < c.cardinality) return k * 0x10000 + c.select(i);
      i -= c.cardinality;
    }
    /* istanbul ignore next */
    throw new RangeError("unreachable");
  }

  min(): number {
    if (this.cardinality === 0) throw new RangeError("min of empty set");
    const k = this.sortedKeys()[0];
    return k * 0x10000 + this.shards.get(k)!.select(0);
  }

  max(): number {
    if (this.cardinality === 0) throw new RangeError("max of empty set");
    const keys = this.sortedKeys();
    const k = keys[keys.length - 1];
    const c = this.shards.get(k)!;
    return k * 0x10000 + c.select(c.cardinality - 1);
  }

  /* ----------------------------- iteration ------------------------------ */

  /** Iterate the values in the inclusive range [start, end], ascending. */
  range(start: number, end: number): IterableIterator<number> {
    checkUint32(start, "start");
    checkUint32(end, "end");
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const self = this;
    return (function* (): Generator<number, void, undefined> {
      if (end < start) return;
      const k0 = start >>> 16;
      const k1 = end >>> 16;
      for (const k of self.sortedKeys()) {
        if (k < k0) continue;
        if (k > k1) break;
        const c = self.shard(k);
        const lo = k === k0 ? start & 0xffff : 0;
        const hi = k === k1 ? end & 0xffff : 0xffff;
        yield* self.iterateContainer(c, lo, hi, k * 0x10000);
      }
    })();
  }

  [Symbol.iterator](): IterableIterator<number> {
    return this.range(0, MAX_UINT32);
  }

  keys(): IterableIterator<number> {
    return this[Symbol.iterator]();
  }

  values(): IterableIterator<number> {
    return this[Symbol.iterator]();
  }

  *entries(): IterableIterator<[number, number]> {
    for (const v of this) yield [v, v];
  }

  forEach(callback: (value: number, key: number, set: Roaring32) => void, thisArg?: unknown): void {
    for (const v of this) callback.call(thisArg, v, v, this);
  }

  toArray(): number[] {
    const out: number[] = [];
    for (const v of this) out.push(v);
    return out;
  }

  /**
   * Iterate one container's values in [lo, hi] (low 16 bits), emitting
   * base + low. Bitmap shards are walked through a small reusable scratch
   * buffer that is released in a `finally` block, so breaking out of the
   * iteration (caller cancellation) cannot leak it.
   */
  private *iterateContainer(c: Container, lo: number, hi: number, base: number): Generator<number> {
    if (c.type === TYPE_ARRAY) {
      let i = c.lowerBound(lo);
      while (i < c.cardinality) {
        const v = c.values[i];
        if (v > hi) break;
        yield base + v;
        i++;
      }
      return;
    }
    if (c.type === TYPE_RUN) {
      // first run whose end >= lo
      let loIdx = 0;
      let hiIdx = c.runs.length;
      while (loIdx < hiIdx) {
        const mid = (loIdx + hiIdx) >>> 1;
        if (runEnd(c.runs[mid]) < lo) loIdx = mid + 1;
        else hiIdx = mid;
      }
      let idx = loIdx;
      while (idx < c.runs.length) {
        const r = c.runs[idx];
        if (r.start > hi) break;
        const s = Math.max(r.start, lo);
        const e = Math.min(runEnd(r), hi);
        for (let v = s; v <= e; v++) yield base + v;
        idx++;
      }
      return;
    }
    // bitmap container: chunked through a temporary scratch buffer
    const CHUNK = 256;
    const scratch = new Uint32Array(CHUNK);
    this.activeBuffers++;
    try {
      let count = 0;
      const firstWord = lo >>> 5;
      const lastWord = hi >>> 5;
      for (let w = firstWord; w <= lastWord; w++) {
        let bits = c.words[w] | 0;
        if (w === firstWord) bits &= wordRangeMask(lo & 31, 31);
        if (w === lastWord) bits &= wordRangeMask(0, hi & 31);
        while (bits !== 0) {
          scratch[count++] = w * 32 + ctz32(bits);
          bits &= bits - 1;
          if (count === CHUNK) {
            for (let i = 0; i < CHUNK; i++) yield base + scratch[i];
            count = 0;
          }
        }
      }
      for (let i = 0; i < count; i++) yield base + scratch[i];
    } finally {
      // Released on normal completion AND on caller cancellation (return/throw).
      scratch.fill(0);
      this.activeBuffers--;
    }
  }

  /* ---------------------------- set operations -------------------------- */

  /** New set: this ∪ other. Inputs are not modified. */
  union(other: Roaring32): Roaring32 {
    return this.combine(other, "union");
  }

  /** New set: this ∩ other. Inputs are not modified. */
  intersect(other: Roaring32): Roaring32 {
    return this.combine(other, "intersect");
  }

  /** New set: this − other. Inputs are not modified. */
  difference(other: Roaring32): Roaring32 {
    return this.combine(other, "difference");
  }

  /** New set: this △ other. Inputs are not modified. */
  xor(other: Roaring32): Roaring32 {
    return this.combine(other, "xor");
  }

  private combine(other: Roaring32, op: SetOp): Roaring32 {
    const out = new Roaring32();
    const ka = this.sortedKeys();
    const kb = other.sortedKeys();
    let i = 0;
    let j = 0;
    let card = 0;
    while (i < ka.length || j < kb.length) {
      let key: number;
      let c: Container | null;
      if (j >= kb.length || (i < ka.length && ka[i] < kb[j])) {
        key = ka[i++];
        if (op === "intersect") continue;
        c = this.shard(key).clone();
      } else if (i >= ka.length || kb[j] < ka[i]) {
        key = kb[j++];
        if (op === "intersect" || op === "difference") continue;
        c = other.shard(key).clone();
      } else {
        key = ka[i];
        i++;
        j++;
        c = containerOp(this.shard(key), other.shard(key), op);
      }
      if (c !== null && c.cardinality > 0) {
        out.shards.set(key, c);
        card += c.cardinality;
      }
    }
    out.cardinality = card;
    return out;
  }

  /** In-place union: this = this ∪ other. Accepts a set or any iterable of values. */
  addAll(other: Roaring32 | Iterable<number>): this {
    if (other instanceof Roaring32) {
      for (const [k, cb] of other.shards) {
        const ca = this.shards.get(k);
        if (ca === undefined) {
          this.shards.set(k, cb.clone());
          this.keysCache = null;
        } else {
          const merged = containerOp(ca, cb, "union");
          if (merged === null || merged.cardinality === 0) this.shards.delete(k);
          else this.shards.set(k, merged);
        }
      }
      this.recomputeCardinality();
    } else {
      for (const v of other) this.add(v);
    }
    return this;
  }

  /** In-place intersection: this = this ∩ other. */
  intersectAll(other: Roaring32): this {
    for (const k of Array.from(this.shards.keys())) {
      const cb = other.shards.get(k);
      const ca = this.shards.get(k)!;
      const c = cb === undefined ? null : containerOp(ca, cb, "intersect");
      if (c === null || c.cardinality === 0) {
        this.shards.delete(k);
        this.keysCache = null;
      } else {
        this.shards.set(k, c);
      }
    }
    this.recomputeCardinality();
    return this;
  }

  /** In-place difference: this = this − other. Accepts a set or any iterable of values. */
  deleteAll(other: Roaring32 | Iterable<number>): this {
    if (other instanceof Roaring32) {
      for (const [k, cb] of other.shards) {
        const ca = this.shards.get(k);
        if (ca === undefined) continue;
        const c = containerOp(ca, cb, "difference");
        if (c === null || c.cardinality === 0) {
          this.shards.delete(k);
          this.keysCache = null;
        } else {
          this.shards.set(k, c);
        }
      }
      this.recomputeCardinality();
    } else {
      for (const v of other) this.delete(v);
    }
    return this;
  }

  /** In-place symmetric difference: this = this △ other. */
  xorAll(other: Roaring32): this {
    for (const [k, cb] of other.shards) {
      const ca = this.shards.get(k);
      if (ca === undefined) {
        this.shards.set(k, cb.clone());
        this.keysCache = null;
        continue;
      }
      const c = containerOp(ca, cb, "xor");
      if (c === null || c.cardinality === 0) {
        this.shards.delete(k);
        this.keysCache = null;
      } else {
        this.shards.set(k, c);
      }
    }
    this.recomputeCardinality();
    return this;
  }

  /** Logical equality: same elements, regardless of container encodings. */
  equals(other: Roaring32): boolean {
    if (this.cardinality !== other.cardinality) return false;
    if (this.shards.size !== other.shards.size) return false;
    const ka = this.sortedKeys();
    const kb = other.sortedKeys();
    for (let i = 0; i < ka.length; i++) {
      if (ka[i] !== kb[i]) return false;
      if (!containerEquals(this.shard(ka[i]), other.shard(kb[i]))) return false;
    }
    return true;
  }

  clone(): Roaring32 {
    const out = new Roaring32();
    for (const [k, c] of this.shards) out.shards.set(k, c.clone());
    out.cardinality = this.cardinality;
    return out;
  }

  /**
   * Re-run container selection on every shard: dense-enough shards become
   * run containers, sparse bitmaps shrink back to arrays. Useful after a
   * batch of point insertions.
   */
  optimize(): void {
    for (const [k, c] of this.shards) {
      let next: Container = c;
      if (c.type === TYPE_BITMAP) next = canonicalizeBitmap(c);
      else if (c.type === TYPE_ARRAY) next = canonicalizeArray(c);
      if (next !== c) this.shards.set(k, next);
    }
  }

  /* ---------------------------- serialization --------------------------- */

  /** Encode as a versioned binary buffer. */
  serialize(): Buffer {
    return encodeSet(this.shards, this.cardinality);
  }

  /**
   * Decode a buffer produced by serialize(). Offsets, lengths and
   * cardinalities are validated before any container is materialised;
   * malformed input throws DeserializationError.
   */
  static deserialize(data: Buffer | Uint8Array): Roaring32 {
    const { shards, cardinality } = decodeSet(data);
    const out = new Roaring32();
    out.shards = shards;
    out.cardinality = cardinality;
    return out;
  }

  /* ------------------------------ internals ----------------------------- */

  private shard(key: number): Container {
    return this.shards.get(key)!;
  }

  private sortedKeys(): number[] {
    if (this.keysCache === null) {
      this.keysCache = Array.from(this.shards.keys()).sort((a, b) => a - b);
    }
    return this.keysCache;
  }

  private recomputeCardinality(): void {
    let sum = 0;
    for (const c of this.shards.values()) sum += c.cardinality;
    this.cardinality = sum;
  }
}

export {
  ArrayContainer,
  BitmapContainer,
  RunContainer,
  DeserializationError,
  SERIALIZATION_VERSION,
  ARRAY_LIMIT,
};
export type { Container, ContainerStats as Stats };
